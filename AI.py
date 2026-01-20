#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Neural Text Generator (Gradio GUI)
V4.4: Original logic preserved + stable training + stable generation + optional KV-cache

Original logic kept:
LM probs -> focus(LateralInhibition) -> top-k -> boosts -> GELU bias -> ResonantGate -> sample
"""

from __future__ import annotations
import re
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


# ----------------------------
# Neural Components (same logic, safer shapes)
# ----------------------------
class LateralInhibition(nn.Module):
    def __init__(self, kernel_size=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        # (keep your original kernel)
        k = torch.tensor([-0.95, -0.9, -0.1, 0.3, -1.4, -1.2, -1.05], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = int(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (V,) or (B,V)
        if x.dim() == 1:
            x_ = x.view(1, 1, -1)
        elif x.dim() == 2:
            x_ = x.unsqueeze(1)  # (B,1,V)
        else:
            raise ValueError(f"LateralInhibition expects 1D/2D, got {tuple(x.shape)}")

        modulation = F.conv1d(x_, self.kernel, padding=self.pad)
        out = x_ + self.strength * modulation
        out = F.relu(out)
        out = out / (out.sum(dim=-1, keepdim=True) + 1e-12)

        if x.dim() == 1:
            return out.view(-1)
        return out.squeeze(1)  # (B,V)


class ResonantGate(nn.Module):
    def __init__(self, steer_strength=1.35):
        super().__init__()
        self.steer_strength = float(steer_strength)
        self.noise_injector = nn.Dropout(p=0.05)

    def forward(self, lm_probs: torch.Tensor, token_boosts: torch.Tensor, temp=0.95) -> torch.Tensor:
        # lm_probs, token_boosts: (K,) or (B,K)
        if lm_probs.dim() == 1:
            lm_probs = lm_probs.unsqueeze(0)
        if token_boosts.dim() == 1:
            token_boosts = token_boosts.unsqueeze(0)

        potentials = torch.log(lm_probs.clamp_min(1e-12))
        potentials = potentials + self.steer_strength * token_boosts
        potentials = potentials / max(float(temp), 1e-9)
        potentials = self.noise_injector(potentials)
        return F.softmax(potentials, dim=-1)  # (B,K)


class SyntheticGELUBias(nn.Module):
    def __init__(self, vocab_size=50257, hidden=32, approximate="tanh"):
        super().__init__()
        self.fc1 = nn.Linear(2, int(hidden))
        self.act = nn.GELU(approximate=approximate)
        self.fc2 = nn.Linear(int(hidden), 1)

    def reset_seed(self, seed: int):
        g = torch.Generator()
        g.manual_seed(int(seed))
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.15, generator=g)
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.15, generator=g)
            nn.init.zeros_(self.fc2.bias)

    def freeze_(self, frozen: bool = True):
        for p in self.parameters():
            p.requires_grad_(not frozen)

    def forward(self, base_probs: torch.Tensor, token_boosts: torch.Tensor) -> torch.Tensor:
        # base_probs, token_boosts: (K,) or (B,K)
        if base_probs.dim() == 1:
            base_probs = base_probs.unsqueeze(0)
        if token_boosts.dim() == 1:
            token_boosts = token_boosts.unsqueeze(0)

        x1 = torch.log(base_probs.clamp_min(1e-12))      # (B,K)
        x = torch.stack([x1, token_boosts], dim=-1)      # (B,K,2)
        h = self.act(self.fc1(x))                        # (B,K,H)
        return self.fc2(h).squeeze(-1)                   # (B,K)


# ----------------------------
# Cube-root scaling (logits domain)
# ----------------------------
def cube_root_weights(n: int, device, dtype):
    n = int(max(0, n))
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.arange(1, n + 1, device=device, dtype=dtype).pow(1.0 / 3.0)

def scale_topn_logits_by_cuberoots(logits: torch.Tensor, n: int) -> torch.Tensor:
    """
    logits: (B,V)
    Multiply top-n logits by [1^(1/3), 2^(1/3), ..., n^(1/3)] using scatter_.
    """
    n = int(max(0, n))
    if n <= 0:
        return logits
    vocab = logits.shape[-1]
    n = min(n, vocab)

    top_vals, top_idx = torch.topk(logits, k=n, dim=-1)         # (B,n)
    w = cube_root_weights(n, device=logits.device, dtype=logits.dtype)  # (n,)
    scaled = top_vals * w                                        # (B,n) broadcast

    out = logits.clone()
    out.scatter_(dim=-1, index=top_idx, src=scaled)
    return out


# ----------------------------
# Text utilities
# ----------------------------
def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext in [".txt", ".md"]:
        return p.read_text(encoding="utf-8", errors="replace")
    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        return "\n".join([para.text for para in d.paragraphs])
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file extension: {ext}")


# ----------------------------
# Generator (original logic preserved)
# ----------------------------
class NeuralTextGenerator(nn.Module):
    def __init__(
        self,
        model_name="gpt2",
        steer_strength=1.35,
        focus_strength=0.5,
        gelu_seed=1337,
        gelu_hidden=32,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        config = GPT2Config.from_pretrained(model_name)
        self.lm = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(self.device)
        self.lm.eval()

        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.gate_layer = ResonantGate(steer_strength=float(steer_strength))
        self.synthetic_bias = SyntheticGELUBias(
            vocab_size=config.vocab_size,
            hidden=gelu_hidden,
            approximate="tanh",
        ).to(self.device)

        self.synthetic_bias.reset_seed(int(gelu_seed))
        self.synthetic_bias.freeze_(True)

        self.eval()

    def build_token_boosts(self, text: str) -> Dict[int, float]:
        tokens = self.tokenizer.tokenize(text.lower())
        token_freq: Dict[str, int] = {}
        for t in tokens:
            token_freq[t] = token_freq.get(t, 0) + 1

        total = len(tokens)
        boosts: Dict[int, float] = {}
        for token, freq in token_freq.items():
            if freq > 1:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                if not ids:
                    continue
                boosts[int(ids[0])] = math.log(freq / max(total, 1) * 100 + 1e-12) + 2.0
        return boosts

    @torch.no_grad()
    def lm_next_logits_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
    ) -> Tuple[torch.Tensor, object]:
        """
        Returns next-token logits for the current prefix and updated cache.
        - If past_key_values is None: feed full input_ids
        - Else: feed only last token
        """
        input_ids = input_ids.to(self.device)
        if past_key_values is None:
            out = self.lm(input_ids, use_cache=True, return_dict=True)
        else:
            out = self.lm(input_ids[:, -1:], use_cache=True, past_key_values=past_key_values, return_dict=True)
        logits = out.logits[:, -1, :]  # (1,V) or (B,V)
        return logits, out.past_key_values

    @torch.no_grad()
    def get_lm_probs_from_logits(self, logits: torch.Tensor, cube_root_n: int = 0) -> torch.Tensor:
        # logits: (B,V)
        if int(cube_root_n) > 0:
            logits = scale_topn_logits_by_cuberoots(logits, int(cube_root_n))
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate_step_from_probs(
        self,
        probs_full: torch.Tensor,                  # (V,)
        token_boosts_dict: Dict[int, float],
        temperature: float,
        top_k: int,
    ) -> int:
        probs_full = self.focus_layer(probs_full)  # (V,) normalized

        top_indices = torch.topk(probs_full, int(top_k)).indices                 # (K,)
        base_probs = probs_full[top_indices]                                     # (K,)

        boosts = torch.tensor(
            [float(token_boosts_dict.get(int(t), 0.0)) for t in top_indices.tolist()],
            device=self.device,
            dtype=base_probs.dtype,
        )  # (K,)

        bias = self.synthetic_bias(base_probs, boosts).view(-1)                  # (K,)
        final_probs = self.gate_layer(base_probs, boosts + bias, temp=temperature).view(-1)  # (K,)

        k_idx = int(torch.multinomial(final_probs, num_samples=1).item())
        return int(top_indices[k_idx].item())

    @torch.no_grad()
    def generate(
        self,
        text: str,
        n_takeaways: int = 5,
        max_length: int = 200,
        seed: int = 42,
        text_seed: str = "",
        overlap_tokens: int = 20,
        temperature: float = 0.9,
        top_k: int = 50,
        cube_root_n: int = 0,
        use_kv_cache: bool = True,
    ) -> str:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        token_boosts_dict = self.build_token_boosts(text)

        prompt = text_seed or text[:200]
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) == 0:
            prompt_ids = [int(self.tokenizer.eos_token_id)]

        takeaways: List[str] = []
        eos_id = int(self.tokenizer.eos_token_id)

        # Each takeaway starts from last 20 tokens of previous as context (original)
        current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        for i in range(int(n_takeaways)):
            generated_ids = current_ids.clone()

            past = None
            if use_kv_cache:
                # Prime cache with current prefix
                logits, past = self.lm_next_logits_cached(generated_ids, past_key_values=None)
            else:
                logits, past = None, None

            # Hard cap on steps for stability
            for _ in range(int(max_length)):
                if use_kv_cache:
                    probs_full = self.get_lm_probs_from_logits(logits, cube_root_n=cube_root_n)[0]  # (V,)
                else:
                    logits_full, _ = self.lm_next_logits_cached(generated_ids, past_key_values=None)
                    probs_full = self.get_lm_probs_from_logits(logits_full, cube_root_n=cube_root_n)[0]

                next_tok = self.generate_step_from_probs(
                    probs_full=probs_full,
                    token_boosts_dict=token_boosts_dict,
                    temperature=float(temperature),
                    top_k=int(top_k),
                )

                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[next_tok]], device=self.device, dtype=torch.long)],
                    dim=1,
                )

                if next_tok == eos_id:
                    break

                if use_kv_cache:
                    logits, past = self.lm_next_logits_cached(generated_ids, past_key_values=past)

            takeaway = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

            if i > 0 and int(overlap_tokens) > 0:
                overlap_start = max(0, len(takeaways[-1]) - int(overlap_tokens))
                takeaway = takeaway[overlap_start:]

            takeaways.append(takeaway)

            keep = min(20, generated_ids.shape[1])
            current_ids = generated_ids[:, -keep:]

        return "\n\n".join(takeaways)


# ----------------------------
# Training (original logic: target-aware NLL on gated top-k)
# ----------------------------
class TextDataset(Dataset):
    """
    Fixed-length (4,) windows: 3 context + 1 target.
    """
    def __init__(self, texts: List[str], tokenizer, max_length=512, stride=128):
        self.tokenizer = tokenizer
        self.stride = int(stride)
        self.seq_len = 4

        tok = tokenizer(
            texts,
            truncation=True,
            max_length=int(max_length),
            padding=False,
            return_tensors="pt",
        )
        self.input_ids = tok["input_ids"]  # (N,T)

        # If the file is extremely short, still keep at least one sample
        if self.input_ids.numel() == 0:
            self.input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)

    def __len__(self):
        return max(1, len(self.input_ids) * self.stride)

    def __getitem__(self, idx):
        seq_idx = idx // self.stride
        pos = (idx % self.stride) + 1

        seq = self.input_ids[min(seq_idx, len(self.input_ids) - 1)]
        start = max(0, pos - 3)
        end = min(len(seq), start + self.seq_len)

        window = seq[start:end]
        if window.numel() < self.seq_len:
            pad = self.seq_len - window.numel()
            window = F.pad(window, (0, pad), value=self.tokenizer.pad_token_id)

        return window  # (4,)


def train_bias_net(
    infile,
    seed,
    steer,
    focus,
    gelu_seed,
    train_steps,
    lr,
    max_contexts,
    batch_size=16,
    train_top_k=64,
    progress=gr.Progress(),
):
    try:
        text = normalize(load_text(infile))
        if len(text) < 100:
            return None, "Text too short for training."

        gen = NeuralTextGenerator(
            steer_strength=float(steer),
            focus_strength=float(focus),
            gelu_seed=int(gelu_seed),
        )

        boosts_dict = gen.build_token_boosts(text)

        progress(0.05, desc="Tokenizing")
        dataset = TextDataset([text], gen.tokenizer, max_length=512, stride=128)
        if max_contexts and int(max_contexts) > 0:
            dataset = torch.utils.data.Subset(dataset, range(min(int(max_contexts), len(dataset))))

        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)

        gen.synthetic_bias.reset_seed(int(gelu_seed))
        gen.synthetic_bias.freeze_(False)
        gen.synthetic_bias.train()

        opt = optim.Adam(gen.synthetic_bias.parameters(), lr=float(lr))

        pad_id = int(gen.tokenizer.pad_token_id)
        top_k = int(train_top_k)
        steps = int(train_steps)

        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(dataloader):
            if step >= steps:
                break

            batch = batch.to(gen.device)     # (B,4)
            contexts = batch[:, :-1]         # (B,3)
            targets = batch[:, -1]           # (B,)

            # Remove padded targets
            keep = (targets != pad_id)
            if keep.sum().item() == 0:
                continue
            contexts = contexts[keep]
            targets = targets[keep]
            B = contexts.shape[0]

            opt.zero_grad()

            # LM full probs -> focus (original)
            logits, _ = gen.lm_next_logits_cached(contexts, past_key_values=None)  # contexts treated as (B,T)
            probs_full = gen.get_lm_probs_from_logits(logits, cube_root_n=0)       # (B,V)
            probs_full = gen.focus_layer(probs_full)                               # (B,V)

            # top-k candidates
            K = min(top_k, probs_full.shape[-1])
            top_vals, top_idx = torch.topk(probs_full, k=K, dim=-1)                # (B,K)

            # ensure target in candidate set
            for b in range(B):
                tgt = int(targets[b].item())
                if not (top_idx[b] == tgt).any():
                    top_idx[b, -1] = tgt
                    top_vals[b, -1] = probs_full[b, tgt]

            # renormalize base probs (keeps them as probabilities)
            top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-12)

            # build boosts per candidate (same as generation) + bump target
            token_boosts = torch.zeros_like(top_vals)
            tgt_local = torch.empty((B,), dtype=torch.long, device=gen.device)

            for b in range(B):
                idxs = top_idx[b].tolist()
                token_boosts[b] = torch.tensor(
                    [float(boosts_dict.get(int(t), 0.0)) for t in idxs],
                    device=gen.device,
                    dtype=top_vals.dtype,
                )
                tgt = int(targets[b].item())
                pos = int((top_idx[b] == tgt).nonzero(as_tuple=True)[0][0].item())
                tgt_local[b] = pos
                token_boosts[b, pos] = token_boosts[b, pos] + 1.0

            # original: bias then gate then NLL on target
            bias = gen.synthetic_bias(top_vals, token_boosts)                         # (B,K)
            gated = gen.gate_layer(top_vals, token_boosts + bias, temp=0.95)          # (B,K)

            log_probs = torch.log(gated.clamp_min(1e-12))
            loss = F.nll_loss(log_probs, tgt_local, reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.synthetic_bias.parameters(), 1.0)
            opt.step()

            running_loss += float(loss.item()) * B
            seen += B

            if (step + 1) % max(1, steps // 20) == 0:
                progress(min(1.0, (step + 1) / steps), desc=f"Training {step+1}/{steps}")

        avg_loss = (running_loss / max(1, seen)) if seen else float("nan")
        msg = f"Training complete. Avg loss: {avg_loss:.4f}"

        trained = {
            "gelu_state_dict": {k: v.detach().cpu() for k, v in gen.synthetic_bias.state_dict().items()},
            "gelu_seed": int(gelu_seed),
            "focus": float(focus),
            "steer": float(steer),
        }
        return trained, msg

    except Exception as e:
        import traceback
        return None, f"Training failed:\n{e}\n\n{traceback.format_exc()}"


def generate_with_training(
    infile,
    n_take,
    seed,
    t_seed,
    steer,
    focus,
    gelu_seed,
    trained_state,
    temperature,
    top_k,
    cube_root_n,
    use_kv_cache,
    progress=gr.Progress(),
):
    try:
        text = normalize(load_text(infile))

        gen = NeuralTextGenerator(
            steer_strength=float(steer),
            focus_strength=float(focus),
            gelu_seed=int(gelu_seed),
        )

        if isinstance(trained_state, dict) and "gelu_state_dict" in trained_state:
            gen.synthetic_bias.load_state_dict(trained_state["gelu_state_dict"], strict=True)
            gen.synthetic_bias.freeze_(True)
            gen.synthetic_bias.eval()

        progress(0.1, desc="Generating")
        out = gen.generate(
            text=text,
            n_takeaways=int(n_take),
            seed=int(seed),
            text_seed=t_seed or "",
            max_length=250,
            temperature=float(temperature),
            top_k=int(top_k),
            cube_root_n=int(cube_root_n),
            use_kv_cache=bool(use_kv_cache),
        )
        return out

    except Exception as e:
        import traceback
        return f"Generation failed:\n{e}\n\n{traceback.format_exc()}"


# ----------------------------
# Gradio UI
# ----------------------------
def build_app():
    with gr.Blocks(title="Neural Text Generator V4.4") as demo:
        gr.Markdown("# Pure Neural Text Generator V4.4\n*Original logic + stable training/generation*")

        trained_state = gr.State(None)

        with gr.Row():
            infile = gr.File(label="Input File", type="filepath")
            out_txt = gr.Textbox(label="Generated Text", lines=15)

        status = gr.Textbox(label="Status", lines=3)

        with gr.Row():
            n_take = gr.Slider(1, 10, value=5, step=1, label="# Takeaways")
            seed = gr.Number(value=42, label="Random Seed")
            temperature = gr.Slider(0.5, 1.5, value=0.9, label="Temperature")
            top_k = gr.Slider(5, 200, value=50, step=1, label="Top-k")

        with gr.Row():
            steer = gr.Slider(0, 3, value=1.35, label="Steer Strength")
            focus = gr.Slider(0, 1, value=0.5, label="Focus Strength")
            cube_root_n = gr.Slider(0, 200, value=0, step=1, label="Cube-root logit N (0=off)")
            use_kv_cache = gr.Checkbox(value=True, label="Use KV cache (faster)")

        with gr.Row():
            gelu_seed = gr.Number(value=1337, label="GELU Seed")
            t_seed = gr.Textbox(label="Text Seed (optional)", placeholder="Start with...")

        with gr.Row():
            train_steps = gr.Slider(50, 2000, value=500, step=50, label="Train Steps")
            lr = gr.Number(value=1e-3, label="Learning Rate")
            max_contexts = gr.Slider(0, 10000, value=2000, step=100, label="Max Contexts")
            batch_size = gr.Slider(2, 64, value=16, step=1, label="Batch Size")
            train_top_k = gr.Slider(8, 256, value=64, step=1, label="Train top-k")

        with gr.Row():
            train_btn = gr.Button("Train GELU Bias", variant="secondary")
            gen_btn = gr.Button("Generate", variant="primary")

        train_btn.click(
            train_bias_net,
            inputs=[infile, seed, steer, focus, gelu_seed, train_steps, lr, max_contexts, batch_size, train_top_k],
            outputs=[trained_state, status],
        )

        gen_btn.click(
            generate_with_training,
            inputs=[infile, n_take, seed, t_seed, steer, focus, gelu_seed, trained_state, temperature, top_k, cube_root_n, use_kv_cache],
            outputs=out_txt,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    # show_error=True prints exceptions to console; debug=True keeps main thread alive for logs.
    app.queue().launch(share=True, show_error=True, debug=True)

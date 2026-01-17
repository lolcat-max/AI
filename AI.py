#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurosymbolic Text Generator V3.3 (Seed-Context Enhanced)
Workflow: Load -> Train -> Generate Loop

Key upgrade:
- Build a longer seed_prefix (N tokens) and prime the rolling (w1,w2,w3) context from it.
"""

import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# ==========================================
# Core PyTorch Modules
# ==========================================

class LateralInhibition(nn.Module):
    def __init__(self, kernel_size=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.95, -0.9, -0.1, 0.3, -1.4, -1.2, -1.05])
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = kernel_size // 2

    def forward(self, x):
        if x.dim() == 1: x = x.view(1, 1, -1)
        elif x.dim() == 2: x = x.view(x.shape[0], 1, x.shape[1])
        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = F.relu(x + self.strength * modulation)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)

class ResonantGate(nn.Module):
    def __init__(self, steer_strength=1.35):
        super().__init__()
        self.steer_strength = float(steer_strength)
        self.noise_injector = nn.Dropout(p=0.05)

    def forward(self, lm_probs, token_boosts, temp=0.95):
        lm_probs = lm_probs.view(-1)
        token_boosts = token_boosts.view(-1)
        potentials = torch.log(lm_probs.clamp_min(1e-12))
        potentials = potentials + self.steer_strength * token_boosts
        potentials = potentials / max(float(temp), 1e-9)
        potentials = self.noise_injector(potentials)
        return F.softmax(potentials, dim=-1)

class SyntheticGELUBias(nn.Module):
    def __init__(self, hidden=32, approximate="tanh"):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.GELU(approximate=approximate)
        self.fc2 = nn.Linear(hidden, 1)

    def reset_seed(self, seed: int):
        g = torch.Generator()
        g.manual_seed(seed)
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.15, generator=g)
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.15, generator=g)
            nn.init.zeros_(self.fc2.bias)

    def freeze_(self, frozen=True):
        for p in self.parameters():
            p.requires_grad_(not frozen)

    def forward(self, base_probs, token_boosts):
        x1 = torch.log(base_probs.clamp_min(1e-12))
        x = torch.stack([x1, token_boosts], dim=-1)
        h = self.act(self.fc1(x))
        return self.fc2(h).squeeze(-1)

# ==========================================
# Text Utilities
# ==========================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "her", "hers", "him", "his", "i", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she",
    "so", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your", "yours"
}

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    # Lowercasing improves matching between TF-IDF vocab/token_boost and LM tokens.
    text = text.replace("\n", " ")
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-']*|[.,;:!?()]", text)
    if lowercase:
        toks = [t.lower() if re.match(r"^[A-Za-z]", t) else t for t in toks]
    return toks

def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in [".", ",", ";", ":", "!", "?", ")", "("]:
            if t in ["(", ")"]:
                out.append(t)
            else:
                if out: out[-1] += t
                else: out.append(t)
        else:
            out.append(t)
    s = " ".join(out)
    s = re.sub(r"\(\s+", "(", s).replace(") ", ")")
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")

# ==========================================
# Logic: QuadgramLM & Generator
# ==========================================

class QuadgramLM:
    def __init__(self, add_k=0.25):
        self.add_k = add_k
        self.uni, self.bi, self.tri, self.quad = {}, {}, {}, {}
        self.vocab, self.total = [], 0

    def ingest(self, tokens: List[str]):
        print("Building language model...")
        self.uni.clear(); self.bi.clear(); self.tri.clear(); self.quad.clear()
        self.total = 0

        for t in tqdm(tokens, desc="Indexing N-grams"):
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1
        for i in range(1, len(tokens)):
            self.bi[(tokens[i-1], tokens[i])] = self.bi.get((tokens[i-1], tokens[i]), 0) + 1
        for i in range(2, len(tokens)):
            self.tri[(tokens[i-2], tokens[i-1], tokens[i])] = self.tri.get((tokens[i-2], tokens[i-1], tokens[i]), 0) + 1
        for i in range(3, len(tokens)):
            self.quad[(tokens[i-3], tokens[i-2], tokens[i-1], tokens[i])] = self.quad.get((tokens[i-3], tokens[i-2], tokens[i-1], tokens[i]), 0) + 1
        self.vocab = list(self.uni.keys())

    def next_distribution(self, w1, w2, w3):
        cont = [d for (a, b, c, d), _ in self.quad.items() if a == w1 and b == w2 and c == w3]
        if not cont:
            cont = [c for (a, b, c), _ in self.tri.items() if a == w2 and b == w3]
        if not cont:
            cont = [b for (a, b), _ in self.bi.items() if a == w3]
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)]

        cand = list(dict.fromkeys(cont))
        V = len(self.vocab) + 1

        def get_prob(w4):
            c123 = self.tri.get((w1, w2, w3), 0)
            c1234 = self.quad.get((w1, w2, w3, w4), 0)
            if c123 > 0: return (c1234 + self.add_k) / (c123 + self.add_k * V)
            c12 = self.bi.get((w2, w3), 0)
            c123_tri = self.tri.get((w2, w3, w4), 0)
            if c12 > 0: return (c123_tri + self.add_k) / (c12 + self.add_k * V)
            c1 = self.uni.get(w3, 0)
            c12_bi = self.bi.get((w3, w4), 0)
            if c1 > 0: return (c12_bi + self.add_k) / (c1 + self.add_k * V)
            return (self.uni.get(w4, 0) + self.add_k) / (self.total + self.add_k * V)

        probs = torch.tensor([get_prob(w) for w in cand], dtype=torch.float32)
        return cand, probs / (probs.sum() + 1e-12)

@dataclass
class Nodelet:
    idx: int
    top_terms: List[Tuple[str, float]]
    energy: float
    narrative: str

@dataclass
class ModelState:
    nodelets: List[Nodelet]
    vocab100: List[str]
    binding_W: torch.Tensor
    bar_probs: torch.Tensor
    token_boost: Dict[str, float]
    pillar_weights: torch.Tensor
    geometric_bias: torch.Tensor

class NeuroSymbolicGraphGenerator:
    def __init__(self, nodelets_n=10, bars_n=100, svd_random_state=7, softmax_temp=0.85,
                 steer_strength=1.35, lm_add_k=0.25, focus_strength=0.5, gelu_seed=1337):
        self.nodelets_n, self.bars_n = int(nodelets_n), int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp, self.lm_add_k = float(softmax_temp), float(lm_add_k)
        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.gate_layer = ResonantGate(steer_strength=float(steer_strength))
        self.synthetic_bias = SyntheticGELUBias()
        self.synthetic_bias.reset_seed(int(gelu_seed))
        self.synthetic_bias.freeze_(True)

    def build_state(self, text: str) -> ModelState:
        print("Extracting neurosymbolic features...")
        text = normalize(text)
        paragraphs = re.split(r"\n\s*\n", text)

        vec = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=8000, ngram_range=(1, 2))
        X = vec.fit_transform(paragraphs)
        vocab = np.array(vec.get_feature_names_out())

        top_idx = np.argsort(-X.sum(axis=0).A1)[:self.bars_n]
        vocab100 = vocab[top_idx].tolist()
        X_svd = X[:, top_idx]

        svd = TruncatedSVD(n_components=min(self.nodelets_n, X_svd.shape[1], 10),
                           random_state=self.svd_random_state)
        svd.fit(X_svd)

        nodelets = []
        for i, comp in enumerate(svd.components_):
            terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))],
                           key=lambda x: -abs(x[1]))
            nodelets.append(Nodelet(i, terms, float(np.linalg.norm(comp)), f"N{i}"))

        W = torch.tensor(svd.components_, dtype=torch.float32)
        W = F.relu(W) / (W.max(dim=1, keepdim=True)[0] + 1e-12)
        energies = torch.tensor([n.energy for n in nodelets], dtype=torch.float32)
        energies = energies / (energies.max() + 1e-12)

        logits = (energies.view(-1, 1) * W).sum(dim=0)
        probs = F.softmax(logits / self.softmax_temp, dim=-1)
        probs = self.focus_layer(probs.view(1, 1, -1)).squeeze()

        token_boost: Dict[str, float] = {}
        for w, p in zip(vocab100, probs.detach().cpu().tolist()):
            for subw in w.split():
                if len(subw) > 2 and subw not in STOPWORDS:
                    token_boost[subw] = max(token_boost.get(subw, 0.0), math.log(p + 1e-12) + 5.0)

        return ModelState(
            nodelets=nodelets,
            vocab100=vocab100,
            binding_W=W,
            bar_probs=probs,
            token_boost=token_boost,
            pillar_weights=torch.zeros_like(probs),
            geometric_bias=torch.zeros_like(probs),
        )

    def _final_probs_for_context(self, lm, token_boost, w1, w2, w3):
        cand, base_probs = lm.next_distribution(w1, w2, w3)
        base_p = base_probs.clone()
        boosts = torch.tensor([token_boost.get(w, 0.0) for w in cand], dtype=torch.float32)
        bias = self.synthetic_bias(base_p, boosts)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze()
        return cand, self.gate_layer(base_p, boosts + bias)

    # ---------- Seed context expansion (NEW) ----------

    def _clean_seed_tokens(self, toks: List[str]) -> List[str]:
        out = []
        for t in toks:
            if re.match(r"^[a-z][a-z0-9_\-']*$", t) and len(t) > 1:
                out.append(t)
            elif t in [".", ",", ";", ":", "!", "?", "(", ")"]:
                out.append(t)
        return out

    def _top_lm_tokens(self, lm, k=50) -> List[str]:
        return [w for w, _ in sorted(lm.uni.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _expand_seed_prefix(self, seed_tokens: List[str], state: ModelState, lm: QuadgramLM,
                            target_len: int = 12, boost_take: int = 6, freq_take: int = 6) -> List[str]:
        """
        Build a longer prefix used to prime the rolling (w1,w2,w3) context.
        Order:
          1) cleaned seed tokens
          2) top boosted terms (prefer overlap with seed)
          3) frequent LM tokens as filler
        """
        seed_tokens = [t for t in seed_tokens if t not in ["(", ")"]]
        seed_core = [t for t in self._clean_seed_tokens(seed_tokens) if t not in STOPWORDS and t.isalpha()]

        # Boosted terms: prefer those mentioned in the seed
        seed_set = set(seed_core)
        boosted_sorted = sorted(state.token_boost.items(), key=lambda x: x[1], reverse=True)
        boosted_overlap = [w for (w, _) in boosted_sorted if w in seed_set]
        boosted_rest = [w for (w, _) in boosted_sorted if w not in seed_set]

        boosted_pick = boosted_overlap[:boost_take] + boosted_rest[:max(0, boost_take - len(boosted_overlap))]

        # Frequent tokens from LM (avoid stopwords and punctuation)
        freq = [w for w in self._top_lm_tokens(lm, k=200) if w.isalpha() and w not in STOPWORDS]
        freq_pick = freq[:freq_take]

        prefix = []
        for seq in (seed_core, boosted_pick, freq_pick):
            for t in seq:
                if t and t not in prefix:
                    prefix.append(t)
                if len(prefix) >= target_len:
                    return prefix[:target_len]

        # Ensure at least 3 tokens
        while len(prefix) < 3:
            prefix.append("the")

        return prefix

    def _pick_initial_context_from_prefix(self, prefix: List[str], lm: QuadgramLM) -> Tuple[str, str, str]:
        if prefix and len(prefix) >= 3:
            return (prefix[-3], prefix[-2], prefix[-1])
        if prefix and len(prefix) == 2:
            return (prefix[0], prefix[1], prefix[1])
        if prefix and len(prefix) == 1:
            return (prefix[0], prefix[0], prefix[0])
        # fallback
        top = self._top_lm_tokens(lm, k=3)
        if len(top) >= 3:
            return (top[0], top[1], top[2])
        return ("the", "system", "works")

    # ---------- Generation ----------

    def generate_report(self, state, lm, n_takeaways=7, seed=7, text_seed="",
                        overlap_tokens=3, max_steps_min=800, max_steps_max=900,
                        seed_ctx_len: int = 12, show_seed_prefix: bool = True) -> str:
        rng = np.random.default_rng(seed)

        raw_seed = (text_seed or "").strip()
        if not raw_seed:
            raw_seed = "quantum neural symbolic systems"

        seed_words = basic_tokenize(normalize(raw_seed), lowercase=True)
        seed_prefix = self._expand_seed_prefix(seed_words, state, lm, target_len=int(seed_ctx_len))

        # Prime the 3-token rolling context from the longer prefix
        w1, w2, w3 = self._pick_initial_context_from_prefix(seed_prefix, lm)
       
        takeaways = []
        for i in tqdm(range(n_takeaways), desc="Writing takeaways"):
            best_tokens = []
            best_tail = (w1, w2, w3)

            # retry limit (kept)
            for _ in range(10):
                if i == 0 and show_seed_prefix:
                    # First takeaway shows the full expanded prefix as part of the text
                    tokens_out = seed_prefix[:]  # visible priming
                    cw1, cw2, cw3 = w1, w2, w3
                else:
                    # Subsequent takeaways keep a short overlap from previous tail
                    tokens_out = [w1, w2, w3][:max(1, overlap_tokens)]
                    cw1, cw2, cw3 = w1, w2, w3

                for _step in range(rng.integers(max_steps_min, max_steps_max)):
                    cand, probs = self._final_probs_for_context(lm, state.token_boost, cw1, cw2, cw3)
                    p = probs.detach().cpu().numpy()
                    nxt = rng.choice(cand, p=p / (p.sum() + 1e-12))
                    tokens_out.append(nxt)
                    cw1, cw2, cw3 = cw2, cw3, nxt

                    if nxt in [".", "!", "?"] and sum(1 for t in tokens_out if t.isalpha()) > 200:
                        break

                best_tokens, best_tail = tokens_out, (cw1, cw2, cw3)
                break

            w1, w2, w3 = best_tail
            printable = best_tokens[overlap_tokens:] if i > 0 and len(best_tokens) > overlap_tokens else best_tokens
            takeaways.append(detokenize(printable))

        return "\n\n".join(takeaways)

# ==========================================
# Training & Main Loop
# ==========================================

def run_training(gen, state, lm, tokens, seed=1337, steps=500, lr=1e-3):
    print(f"\nTraining GELU bias ({steps} steps)...")
    gen.synthetic_bias.reset_seed(seed)
    gen.synthetic_bias.freeze_(False)
    opt = optim.Adam(gen.synthetic_bias.parameters(), lr=lr)

    positions = list(range(3, min(len(tokens), 5000)))
    rng = np.random.default_rng(seed)

    losses = []
    for step in tqdm(range(steps), desc="Optimizing"):
        opt.zero_grad()
        batch_pos = rng.choice(positions, size=min(24, len(positions)), replace=False)

        loss_acc, used = 0.0, 0
        for i in batch_pos:
            w1, w2, w3, true_next = tokens[i-3:i+1]
            cand, probs = gen._final_probs_for_context(lm, state.token_boost, w1, w2, w3)

            try:
                j = cand.index(true_next)
                loss_acc -= torch.log(probs[j].clamp_min(1e-12))
                used += 1
            except ValueError:
                continue

        if used > 0:
            loss = loss_acc / used
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.synthetic_bias.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))

    gen.synthetic_bias.freeze_(True)
    final_loss = np.mean(losses[-20:]) if losses else 0.0
    print(f"Training Complete. Final Loss: {final_loss:.4f}\n")

def main():
    print("=== Neurosymbolic Text Generator V3.3 (Seed-Context Enhanced) ===\n")

    infile = input("Input file (Enter for demo): ").strip()
    if infile and Path(infile).exists():
        text = load_text(infile)
        print(f"Loaded {len(text):,} chars")
    else:
        text = """Quantum computing leverages superposition and entanglement to solve complex
optimization problems that classical computers struggle with. Grover's algorithm
provides quadratic speedup for unstructured search while Shor's algorithm factors
large numbers exponentially faster. Neural networks excel at pattern recognition
through gradient descent optimization of multi-layer perceptrons."""
        print("Using demo text")

    seed_sys = 42
    gen = NeuroSymbolicGraphGenerator(gelu_seed=seed_sys)
    state = gen.build_state(text)

    tokens = basic_tokenize(text, lowercase=True)
    lm = QuadgramLM()
    lm.ingest(tokens)

    if input("\nTrain GELU bias now? (Y/n) [y]: ").lower().strip() != 'n':
        try:
            steps = int(input("Steps [500]: ") or "500")
        except ValueError:
            steps = 500
        run_training(gen, state, lm, tokens, seed=seed_sys, steps=steps)
    else:
        print("Skipping training (using random init)")

    print("Entering Generation Loop (Ctrl+C to quit)")

    while True:
        print("\n" + "-" * 40)
        text_seed = input("Text seed (or 'q' to quit): ").strip()
        if text_seed.lower() == 'q':
            break

        if not text_seed:
            text_seed = "the system"

        try:
            n_take = int(input("Num takeaways [5]: ") or "5")
            seed_ctx_len = int(input("Seed context tokens [12]: ") or "12")
            seed_run = int(input("Random seed [random]: ") or str(np.random.randint(10000)))
        except ValueError:
            n_take, seed_ctx_len, seed_run = 5, 12, np.random.randint(10000)

        output = gen.generate_report(
            state, lm,
            n_takeaways=n_take,
            seed=seed_run,
            text_seed=text_seed,
            seed_ctx_len=seed_ctx_len,
            show_seed_prefix=True
        )

        print("\n" + "=" * 80)
        print(output)
        print("=" * 80)

        if input("\nSave result? (y/N): ").lower() == 'y':
            fname = input("Filename [output.txt]: ").strip() or "output.txt"
            with open(fname, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Seed: {text_seed} ---\n{output}")
            print(f"Appended to {fname}")

    print("\nExiting.")

if __name__ == "__main__":
    main()

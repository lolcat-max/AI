# main_signal_no_attention.py
#
# Complete, self-contained runnable script combining:
# 1. Attention-free PyTorch signal processing module.
# 2. Main execution driver with stubs for caches and features.
# 3. Fix for the NameError by importing defaultdict.
# 4. Fix for the KeyError by safely handling out-of-vocabulary token IDs.
# 5. Added user input for seed text and expanded QA context.

import math
import re
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Signal Module: PyTorch NN Components (Attention-Free)
# --------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.embed.weight[pad_id].zero_()
    def forward(self, x_ids):
        return self.embed(x_ids)

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, groups=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class DSConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.dw = CausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation, groups=d_model, bias=False)
        self.pw = CausalConv1d(d_model, d_model, kernel_size=1, bias=True)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_bt_d):
        x_res = x_bt_d
        x = x_bt_d.transpose(1, 2)
        h = self.dw(x)
        h = F.gelu(h)
        h = self.pw(h)
        h = h.transpose(1, 2)
        x = x_res + self.dropout(h)
        x = self.ln(x)
        return x

class ContextEncoderCNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_blocks=41, kernel_size=5, pad_id=0, max_dilation=18):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, pad_id)
        dilations = [min(max_dilation, 2**i) for i in range(n_blocks)]
        self.blocks = nn.ModuleList(
            [DSConvBlock(d_model, kernel_size=kernel_size, dilation=d) for d in dilations]
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x_ids):
        x = self.embed(x_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        ctx = x[:, -1, :]
        return x, ctx

class FeatureNet(nn.Module):
    def __init__(self, in_features, d_hidden=128, out_features=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, out_features), nn.ReLU(), nn.Dropout(dropout)
        )
    def forward(self, feat):
        return self.net(feat)

class QuantumFeatMLP(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 32), nn.GELU())
        self.q_out = nn.Linear(32, 5)
        self.scale_out = nn.Linear(32, 1)
        self.temp_out = nn.Linear(32, 1)
    def forward(self, q_in):
        h = self.backbone(q_in)
        qvec = self.q_out(h)
        logit_scale = torch.sigmoid(self.scale_out(h)) * 2.0
        temperature = 0.5 + 1.5 * torch.sigmoid(self.temp_out(h))
        return qvec, logit_scale, temperature

class AnswerBankBias(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.Ea = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.Ea.weight, mean=0.0, std=0.02)
        self.bias_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, vocab_size))
    def forward(self, answer_token_ids):
        if answer_token_ids is None: return None
        Ea = self.Ea(answer_token_ids)
        kv = Ea.mean(dim=1)
        return self.bias_mlp(kv)

class Controller(nn.Module):
    def __init__(self, d_in, n_states=6, d_hidden=128):
        super().__init__()
        self.rnn = nn.GRUCell(d_in, d_hidden)
        self.state_head = nn.Linear(d_hidden, n_states)
        self.energy = nn.Parameter(torch.zeros(n_states))
        self.h = None
    def reset(self, B, device):
        self.h = torch.zeros(B, self.rnn.hidden_size, device=device)
    def forward(self, ctrl_in):
        self.h = self.rnn(ctrl_in, self.h)
        logits = self.state_head(self.h)
        probs = torch.softmax(logits, dim=-1)
        gates = F.softplus(self.energy).unsqueeze(0).expand_as(logits)
        return probs, gates

class MaskedLMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, ctx_vec, mask=None, bias=None, temperature=None, logit_scale=None):
        logits = self.head(self.ln(ctx_vec))
        if bias is not None: logits = logits + bias
        if logit_scale is not None: logits = logits * logit_scale
        if mask is not None: logits = logits.masked_fill(~mask.bool(), float("-inf"))
        temp = 1.0 if temperature is None else torch.clamp(temperature, 0.5, 2.0)
        return logits, torch.softmax(logits / temp, dim=-1)

class NeuralGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_blocks=4, kernel_size=5, feat_dim=16, quantum_dim=8, controller_states=6, pad_id=0):
        super().__init__()
        self.ctx_enc = ContextEncoderCNN(vocab_size, d_model, n_blocks, kernel_size, pad_id)
        self.feat_net = FeatureNet(feat_dim, out_features=64)
        self.quantum_net = QuantumFeatMLP(quantum_dim)
        self.answer_bias = AnswerBankBias(vocab_size, d_model)
        self.controller = Controller(d_in=d_model + 64 + 5, n_states=controller_states)
        self.lm = MaskedLMHead(d_model, vocab_size)
        self.vocab_size = vocab_size

    def reset(self, B, device):
        self.controller.reset(B, device)

    def forward(self, context_ids, feat_vec, quantum_vec, answer_token_ids=None, vocab_mask=None):
        _, ctx = self.ctx_enc(context_ids)
        f = self.feat_net(feat_vec)
        qhat, logit_scale, temperature = self.quantum_net(quantum_vec)
        bias = self.answer_bias(answer_token_ids)
        ctrl_in = torch.cat([ctx, f, qhat], dim=-1)
        _, energy_gates = self.controller(ctrl_in)
        gate_scalar = energy_gates.mean(dim=-1, keepdim=True)
        logits, probs = self.lm(
            ctx, mask=vocab_mask, bias=bias, temperature=temperature,
            logit_scale=logit_scale * (1.0 + 0.05 * gate_scalar)
        )
        return {"logits": logits, "probs": probs}

    @torch.no_grad()
    def step(self, context_ids, feat_vec, quantum_vec, allowed_token_ids, answer_token_ids=None):
        B, K = allowed_token_ids.shape
        device = context_ids.device
        vocab_mask = torch.zeros(B, self.vocab_size, dtype=torch.bool, device=device)
        vocab_mask.scatter_(1, allowed_token_ids, True)
        out = self.forward(context_ids, feat_vec, quantum_vec, answer_token_ids, vocab_mask)
        probs_allowed = out["probs"].gather(1, allowed_token_ids)
        samp_idx_in_K = torch.multinomial(probs_allowed, 1)
        chosen_token = allowed_token_ids.gather(1, samp_idx_in_K)
        return {**out, "allowed_probs": probs_allowed, "sampled_token": chosen_token}

# --------------------------
# Main Execution Driver
# --------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+|[.,!?;:<>/]", text.lower())

def build_ngram(tokens: List[str], n: int = 2) -> Dict[Tuple[str, ...], List[str]]:
    model = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        model[key].append(tokens[i + n])
    if tokens and len(tokens) >= n:
        model[tuple(tokens[-n:])].append("<eos>")
    return model

def build_vocab(tokens: List[str]):
    uniq = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(list(set(tokens)))
    stoi = {t: i for i, t in enumerate(uniq)}
    itos = {i: t for i, t in enumerate(uniq)}
    return stoi, itos

class PreprocessingCacheStub:
    def get_quantum_features(self, segment, **kwargs):
        length = max(1, len(segment))
        avg_len = sum(len(w) for w in segment) / length
        vocab_div = len(set(segment)) / length
        coherence = 0.4 + 0.6 * min(1.0, len(segment) / 64.0)
        return {
            "avg_energy": 0.3 + 0.7 * min(1.0, avg_len / 8.0),
            "energy_variance": 0.1 + 0.9 * abs(vocab_div - 0.5),
            "avg_probability": 0.5, "coherence": coherence,
            "uncertainty_product": max(0.05, 1.0 - coherence * 0.5)
        }

class FSMStub:
    def __init__(self, stoi, itos, tokens, qa_context, device):
        self.stoi, self.itos, self.device = stoi, itos, device
        self.word_freq, self.total_words = Counter(tokens), len(tokens)
        self.preprocessing_cache = PreprocessingCacheStub()
        self.qa_context, self.context_window = qa_context, []

def init_neural_generator(vocab_size, pad_id, device):
    ng = NeuralGenerator(vocab_size=vocab_size, d_model=192, n_blocks=4, pad_id=pad_id).to(device)
    ng.reset(B=1, device=device)
    return ng

def pack_step_tensors(fsm, candidates, T_ctx=64, feat_dim=16, quantum_dim=8):
    device = fsm.device
    ctx_tokens = fsm.context_window[-T_ctx:]
    ctx_ids = [fsm.stoi.get(t, fsm.stoi["<unk>"]) for t in ctx_tokens]
    ctx_ids = [fsm.stoi["<pad>"]] * (T_ctx - len(ctx_ids)) + ctx_ids
    context_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
    
    feat = [0.0] * feat_dim
    feat_vec = torch.tensor([feat], dtype=torch.float32, device=device)
    
    qraw = fsm.preprocessing_cache.get_quantum_features(fsm.context_window[-50:])
    qvec = list(qraw.values()) + [0.0] * (quantum_dim - len(qraw))
    quantum_vec = torch.tensor([qvec[:quantum_dim]], dtype=torch.float32, device=device)

    allowed_ids = torch.tensor([[fsm.stoi.get(t, fsm.stoi["<unk>"]) for t in candidates]], dtype=torch.long, device=device)
    
    answer_token_ids = None
    if fsm.qa_context:
        qa_toks = list(dict.fromkeys((fsm.qa_context.get('answer_patterns') or [])))[:64]
        if qa_toks:
            qa_ids = [fsm.stoi.get(t, fsm.stoi["<unk>"]) for t in qa_toks]
            answer_token_ids = torch.tensor([qa_ids], dtype=torch.long, device=device)
    
    return context_ids, feat_vec, quantum_vec, allowed_ids, answer_token_ids
# assoc_boost.py (put near your other helpers)
import torch
from collections import Counter

def assoc_distribution_from_ngram(ngram_model,
                                  context_window,
                                  candidates,
                                  unigram_counts=None,
                                  order=1,          # 1 uses last token; 2 can use last two tokens if present
                                  alpha=0.05,       # unigram smoothing strength
                                  gamma=1.0,        # sharpness of association distribution
                                  device=None):
    """
    Returns a torch.FloatTensor of shape (K,) aligned to candidates with P_assoc(cand).
    """
    device = device or torch.device("cpu")
    # Build context key
    if order >= 2 and len(context_window) >= 2:
        key = tuple(context_window[-2:])
    else:
        key = tuple(context_window[-1:])

    # Successor frequencies for the chosen key
    succ = ngram_model.get(key, [])
    succ_freqs = Counter(succ)
    total = sum(succ_freqs.values())

    # Unigram smoothing to avoid zeros
    if unigram_counts is None:
        # derive a minimal unigram from all successors of the last token if nothing else
        unigram_counts = Counter(succ) if succ else Counter()

    unigram_total = sum(unigram_counts.values()) or 1

    probs = []
    for cand in candidates:
        f = succ_freqs.get(cand, 0)
        u = unigram_counts.get(cand, 0)
        p = (f + alpha * (u / unigram_total)) / max(1e-12, (total + alpha))
        p = max(p, 1e-12)  # numerical floor
        probs.append(p)

    # Sharpen/flatten with gamma, then normalize
    t = torch.tensor(probs, dtype=torch.float32, device=device)
    if gamma != 1.0:
        t = torch.clamp(t, min=1e-12).pow(gamma)
    t = t / torch.clamp(t.sum(), min=1e-12)
    return t

def apply_assoc_boost(allowed_probs,
                      assoc_probs,
                      beta=1.3):   # boost strength (beta>1 boosts association)
    """
    Combine model probs with association distribution via geometric mixture:
        new ∝ allowed_probs * (assoc_probs ** beta)
    """
    allowed_probs = torch.clamp(allowed_probs, min=1e-12)
    assoc_probs = torch.clamp(assoc_probs, min=1e-12)
    mixed = allowed_probs * assoc_probs.pow(beta)
    mixed = mixed / torch.clamp(mixed.sum(), min=1e-12)
    return mixed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        filename = input("\nEnter corpus filename: ").strip()
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Corpus file not found at '{filename}'. Exiting.")
        return
        
    tokens = tokenize(text)
    ngram = build_ngram(tokens, n=2)
    stoi, itos = build_vocab(tokens)
    while True:
        # --- User Seed Input ---
        user_seed_str = input("Enter seed tokens (e.g., 'the quick') or leave empty for '<bos>': ").strip()
        if user_seed_str:
            seed_tokens = tokenize(user_seed_str)
        else:
            seed_tokens = ["<bos>"]
        print(f"Using seed: {seed_tokens}")
        
        # --- Expanded QA Context ---
        qa_context = {
            "answer_patterns": ["answer", "yes", ".", "ok", "got", "it", "true", "false"], 
            "answer_tokens": ["answer", "yes", ".", "ok", "got", "it", "true", "false", "thanks", "no", "problem", "correct", "incorrect"]
        }
        
        fsm = FSMStub(stoi, itos, tokens, qa_context, device)
        fsm.context_window = seed_tokens.copy()
        
        neural_gen = init_neural_generator(len(stoi), stoi["<pad>"], device)

        emitted = [] # Will not include the seed, only newly generated tokens
        max_len = 1000
        
        print("\n--- Starting Generation Loop ---\n")

        for step in range(max_len):
            key = tuple(fsm.context_window[-2:])
            #print(f"Step {step + 1}: Context key = {key}")

            candidates = ngram.get(key, ngram.get(tuple(fsm.context_window[-1:]), ["<eos>"]))
            unique_candidates = list(dict.fromkeys(candidates))
            #print(f"  -> Candidates: {unique_candidates}")

            if step == 0 and not any(t in ngram for t in [key, tuple(fsm.context_window[-1:])]):
                #print("  -> Initial seed has no successors in n-gram model. Using random token.")
                unique_candidates = [random.choice(tokens)]

            if not unique_candidates:
                 print("  -> No candidates found. Breaking loop.")
                 break

            tensors = pack_step_tensors(fsm, unique_candidates)
            context_ids, feat_vec, quantum_vec, allowed_ids, answer_token_ids = tensors
            out = neural_gen.step(context_ids, feat_vec, quantum_vec, allowed_ids, answer_token_ids)
              
            probs = out['allowed_probs'].squeeze(0).squeeze(-1) if out['allowed_probs'].ndim == 3 else out['allowed_probs'].squeeze()
            # Ensure probs is 1D (K,)
            if probs.ndim != 1:
                probs = probs.view(-1)

            # Build (or reuse) unigram counts once outside the loop for speed:
            # unigram_counts = Counter(tokens)  # do this once after tokenization

            assoc = assoc_distribution_from_ngram(
                ngram_model=ngram,                          # your bigram dict
                context_window=fsm.context_window,
                candidates=unique_candidates,
                unigram_counts=Counter(tokens),             # or reuse a prebuilt global unigram counter
                order=2,                                    # use last token; set 2 to use last two tokens when possible
                alpha=0.05,                                 # smoothing
                gamma=1.0,                                  # association sharpness; >1 sharpens, <1 flattens
                device=probs.device
            )

            # Mix the distributions: model probs and association
            probs_boosted = apply_assoc_boost(
                allowed_probs=probs,
                assoc_probs=assoc,
                beta=1.3                                    # boost strength; increase for stronger association
            )

            # Overwrite the probabilities used for sampling
            # Create a 2D tensor if your code expects (1,K)
            probs_for_sampling = probs_boosted.unsqueeze(0)

            # Now sample using the boosted distribution
            samp_idx_in_K = torch.multinomial(probs_for_sampling, 1)  # (1,1)
            sampled_id = allowed_ids.gather(1, samp_idx_in_K).item()
            tok = itos.get(sampled_id, "<unk>")

            #print(f"  -> Sampled Token: '{tok}'\n")

            if tok == "<eos>":
                print("--- <eos> token sampled. Ending generation. ---")
                break
            
            emitted.append(tok)
            fsm.context_window.append(tok)

        print("\n--- Generation Finished ---")
        print("Initial Seed:", " ".join(seed_tokens))
        print("Generated Text:", " ".join(emitted))

if __name__ == "__main__":
    main()

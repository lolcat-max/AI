#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-Theoretic Neurosymbolic Text Generator (Gradio GUI)
V3.2 + "SGLang-like" structured program layer (no sglang library)

Training button:
- Trains ONLY the SyntheticGELUBias (GELU MLP) on your input file.
- Stores learned weights in gr.State, then Generate loads them for sampling.

SGLang-like button:
- Implements minimal primitives (extend/+=, gen, select, fork, join) in pure Python
- Runs a structured program: fork takeaways -> join -> summarize

Dependencies:
  pip install gradio numpy scikit-learn networkx tqdm datasets pypdf python-docx torch
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import networkx as nx
from networkx.algorithms import isomorphism as iso
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


# ----------------------------
# PyTorch Isomorphic Neuromorphisms
# ----------------------------
class LateralInhibition(nn.Module):
    def __init__(self, kernel_size=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.95, -0.9, -0.1, 0.3, -1.4, -1.2, -1.05], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = int(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [L], [B,L], or [B,1,L]
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])

        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = x + self.strength * modulation
        out = F.relu(out)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)


class SynapticPruner(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(int(n_features)))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W * self.gain.view(1, -1)


class ResonantGate(nn.Module):
    def __init__(self, steer_strength=1.35):
        super().__init__()
        self.steer_strength = float(steer_strength)
        self.noise_injector = nn.Dropout(p=0.05)

    def forward(self, lm_probs: torch.Tensor, token_boosts: torch.Tensor, temp=0.95) -> torch.Tensor:
        lm_probs = lm_probs.view(-1)
        token_boosts = token_boosts.view(-1)

        potentials = torch.log(lm_probs.clamp_min(1e-12))
        potentials = potentials + self.steer_strength * token_boosts
        potentials = potentials / max(float(temp), 1e-9)
        potentials = self.noise_injector(potentials)
        return F.softmax(potentials, dim=-1)


class SyntheticGELUBias(nn.Module):
    """
    Trainable GELU MLP bias field:
    Input: [log(base_prob), token_boost] -> per-token bias
    """
    def __init__(self, hidden=32, approximate="tanh"):
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
        base_probs = base_probs.view(-1)
        token_boosts = token_boosts.view(-1)

        x1 = torch.log(base_probs.clamp_min(1e-12))
        x = torch.stack([x1, token_boosts], dim=-1)  # [V,2]
        h = self.act(self.fc1(x))
        return self.fc2(h).squeeze(-1)               # [V]


# ----------------------------
# Text utilities
# ----------------------------
STOPWORDS = set("""
a an and are as at be by for from has have he her hers him his i in is it its me my
of on or our ours she so that the their them they this to was we were what when where
which who will with you your yours
""".split())


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def basic_tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-']*|[.,;:!?()]", text)
    out = []
    for t in tokens:
        if re.match(r"[A-Za-z]", t):
            out.append(t.lower())
        else:
            out.append(t)
    return out


def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in [".", ",", ";", ":", "!", "?", ")", "("]:
            if t in ["(", ")"]:
                out.append(t)
            else:
                if out:
                    out[-1] = out[-1] + t
                else:
                    out.append(t)
        else:
            if out and out[-1].endswith("("):
                out[-1] = out[-1] + t
            else:
                out.append(t)
    s = " ".join(out)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s


# ----------------------------
# File loading
# ----------------------------
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
# Quadgram LM (symbolic)
# ----------------------------
class QuadgramLM:
    def __init__(self, add_k: float = 0.25):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def ingest(self, tokens: List[str]) -> None:
        self.uni.clear(); self.bi.clear(); self.tri.clear(); self.quad.clear()
        self.total = 0
        for t in tokens:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i + 1])
            self.bi[k] = self.bi.get(k, 0) + 1
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.tri[k] = self.tri.get(k, 0) + 1
        for i in range(len(tokens) - 3):
            k = (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
            self.quad[k] = self.quad.get(k, 0) + 1
        self.vocab = list(self.uni.keys())

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        cont = []
        for (a, b, c, d), count in self.quad.items():
            if a == w1 and b == w2 and c == w3:
                cont.append(d)
        if not cont:
            for (a, b, c), count in self.tri.items():
                if a == w2 and b == w3:
                    cont.append(c)
        if not cont:
            for (a, b), count in self.bi.items():
                if a == w3:
                    cont.append(b)
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)[:200]]

        seen = set()
        cand = []
        for w in cont:
            if w not in seen:
                seen.add(w)
                cand.append(w)
        cand = cand[:500]

        V = len(self.vocab) + 1
        add_k = self.add_k

        def get_prob(w4: str) -> float:
            c123 = self.tri.get((w1, w2, w3), 0)
            c1234 = self.quad.get((w1, w2, w3, w4), 0)
            if c123 > 0:
                return (c1234 + add_k) / (c123 + add_k * V)

            c12 = self.bi.get((w2, w3), 0)
            c123_tri = self.tri.get((w2, w3, w4), 0)
            if c12 > 0:
                return (c123_tri + add_k) / (c12 + add_k * V)

            c1 = self.uni.get(w3, 0)
            c12_bi = self.bi.get((w3, w4), 0)
            if c1 > 0:
                return (c12_bi + add_k) / (c1 + add_k * V)

            return (self.uni.get(w4, 0) + add_k) / (self.total + add_k * V)

        probs = torch.tensor([get_prob(w) for w in cand], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# ----------------------------
# Graph generator state
# ----------------------------
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
    semantic_graph: nx.Graph
    lm_graph: nx.DiGraph


class NeuroSymbolicGraphGenerator:
    def __init__(
        self,
        nodelets_n: int = 10,
        bars_n: int = 100,
        svd_random_state: int = 7,
        softmax_temp: float = 0.85,
        steer_strength: float = 1.35,
        lm_add_k: float = 0.25,
        pillar_strength: float = 0.85,
        geometric_strength: float = 0.3,
        rfe_enabled: bool = True,
        rfe_iterations: int = 3,
        rfe_removal_rate: float = 0.15,
        focus_strength: float = 0.5,
        gelu_seed: int = 1337,
        gelu_hidden: int = 32,
    ):
        self.nodelets_n = int(nodelets_n)
        self.bars_n = int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp = float(softmax_temp)
        self.lm_add_k = float(lm_add_k)
        self.pillar_strength = float(pillar_strength)
        self.geometric_strength = float(geometric_strength)
        self.rfe_enabled = bool(rfe_enabled)
        self.rfe_iterations = int(rfe_iterations)
        self.rfe_removal_rate = float(rfe_removal_rate)

        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.gate_layer = ResonantGate(steer_strength=float(steer_strength))

        self.synthetic_bias = SyntheticGELUBias(hidden=gelu_hidden, approximate="tanh")
        self.synthetic_bias.reset_seed(int(gelu_seed))
        self.synthetic_bias.freeze_(True)  # default frozen unless training button used

        self.pruner: Optional[SynapticPruner] = None

    def _token_class(self, tok: str) -> str:
        if tok in [".", ",", ";", ":", "!", "?", "(", ")"]:
            return "PUNC"
        if not re.match(r"[a-z]", tok):
            return "OTHER"
        L = len(tok)
        return "S" if L <= 3 else "M" if L <= 7 else "L"

    def _pick_initial_context(self, lm: QuadgramLM, rng: np.random.Generator, seed_words: List[str]) -> Tuple[str, str, str]:
        # Use last 3 word-like seed tokens if provided, else fall back to corpus vocab.
        sw = [t for t in seed_words if re.match(r"^[a-z][a-z0-9_\-']*$", t)]
        if len(sw) >= 3:
            return (sw[-3], sw[-2], sw[-1])
        if len(sw) == 2:
            return (sw[-2], sw[-1], sw[-1])
        if len(sw) == 1:
            return (sw[-1], sw[-1], sw[-1])
        seed_tok = lm.vocab[0] if lm.vocab else "the"
        return (seed_tok, seed_tok, seed_tok)

    def _build_token_structure_graph(self, tokens: List[str], max_nodes: int = 220) -> nx.DiGraph:
        toks = tokens[:max_nodes]
        G = nx.DiGraph()
        for i, t in enumerate(toks):
            G.add_node(i, cls=self._token_class(t))
        for i in range(len(toks) - 1):
            G.add_edge(i, i + 1, rel="adj")
        for i in range(len(toks) - 2):
            G.add_edge(i, i + 2, rel="skip")
        return G

    def _graph_signature(self, G: nx.Graph) -> Dict[str, object]:
        deg = [d for _, d in G.degree()]
        wl_mode = G.to_undirected() if G.is_directed() else G
        node_attr = "cls" if G.is_directed() else None
        wl = weisfeiler_lehman_graph_hash(wl_mode, node_attr=node_attr, iterations=3, digest_size=16)

        match_kwargs = (
            {"node_match": iso.categorical_node_match("cls", None),
             "edge_match": iso.categorical_edge_match("rel", None)}
            if G.is_directed()
            else {}
        )
        GM = iso.DiGraphMatcher(G, G, **match_kwargs) if G.is_directed() else iso.GraphMatcher(G, G, **match_kwargs)

        cnt = 0
        for _ in GM.isomorphisms_iter():
            cnt += 1
            if cnt >= 150:
                break

        return {"deg_hist": np.bincount(deg, minlength=16)[:16], "wl": wl, "aut_est": cnt}

    def _passes_automorphism_checks(self, ref_sig, out_sig) -> bool:
        strict = max(0.0, min(2.0, self.geometric_strength))

        ref = ref_sig["deg_hist"].astype(float); ref /= (ref.sum() + 1e-12)
        out = out_sig["deg_hist"].astype(float); out /= (out.sum() + 1e-12)
        if np.abs(ref - out).sum() > max(0.25, 1.10 - 0.35 * strict):
            return False

        ratio = max(1, out_sig["aut_est"]) / max(1, ref_sig["aut_est"])
        band = max(1.3, 3.5 - 1.2 * min(1.0, self.gate_layer.steer_strength / 2.0))
        if not (1.0 / band <= ratio <= band):
            return False

        if strict >= 1.6 and out_sig["wl"] != ref_sig["wl"]:
            return False

        return True

    def _synaptic_prune(self, W: torch.Tensor, energies: torch.Tensor, vocab100: List[str], progress=None):
        if not self.rfe_enabled or self.rfe_iterations <= 0:
            return W, vocab100

        k, bars_n = W.shape
        self.pruner = SynapticPruner(bars_n)

        W_curr = W.detach().clone().requires_grad_(True)
        kept_mask = torch.ones(bars_n, dtype=torch.bool)

        for iteration in range(self.rfe_iterations):
            if progress:
                progress(0.80 + 0.05 * (iteration / max(1, self.rfe_iterations)), desc=f"Synaptic Pruning {iteration+1}")

            W_modulated = self.pruner(W_curr)
            loss = -torch.sum(W_modulated * energies.view(-1, 1)) + 0.1 * torch.var(W_modulated, dim=0).sum()
            loss.backward()

            with torch.no_grad():
                grads = W_curr.grad.abs().sum(dim=0)
                weights = W_curr.abs().sum(dim=0)
                importance = 0.6 * weights + 0.4 * grads
                importance = importance / (importance.max() + 1e-12)

                n_keep = int(kept_mask.sum().item() * (1.0 - self.rfe_removal_rate))
                if n_keep < 10:
                    break

                active = torch.where(kept_mask)[0]
                local_importance = importance[active]
                _, top_local = torch.topk(local_importance, k=min(n_keep, local_importance.numel()))

                new_mask = torch.zeros_like(kept_mask)
                new_mask[active[top_local]] = True
                kept_mask = new_mask

                W_curr.grad.zero_()

        with torch.no_grad():
            final_idx = torch.where(kept_mask)[0]
            W_final = W[:, final_idx]
            vocab_final = [vocab100[i] for i in final_idx.tolist()]
        return W_final, vocab_final

    def build_state(self, text: str, progress=None) -> ModelState:
        if progress:
            progress(0, desc="Normalizing")
        text = normalize(text)

        vec = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
        X = vec.fit_transform(re.split(r"\n\s*\n", text)[:500])
        vocab = np.array(vec.get_feature_names_out())

        top_idx = np.argsort(-np.asarray(X.sum(axis=0)).ravel())[:self.bars_n]
        vocab100 = vocab[top_idx].tolist()

        X_svd = X[:, top_idx]
        n_rows, n_cols = X_svd.shape
        max_rank = min(n_rows, n_cols)
        k = 1 if max_rank <= 1 else min(self.nodelets_n, max_rank, 10)

        svd = TruncatedSVD(n_components=k, random_state=self.svd_random_state)
        svd.fit(X_svd)

        nodelets = []
        for i, comp in enumerate(svd.components_):
            terms = sorted([(vocab100[j], float(comp[j])) for j in range(len(comp))], key=lambda x: -abs(x[1]))[:10]
            eng = float(np.linalg.norm(comp))
            nodelets.append(Nodelet(i, terms, eng, f"Nodelet {i}"))

        W = torch.tensor(svd.components_, dtype=torch.float32)
        W = F.relu(W)
        W = W / (W.max(dim=1, keepdim=True)[0] + 1e-12)

        energies = torch.tensor([n.energy for n in nodelets], dtype=torch.float32)
        energies = energies / (energies.max() + 1e-12)

        W, vocab100 = self._synaptic_prune(W, energies, vocab100, progress)

        logits = (energies.view(-1, 1) * W).sum(dim=0)
        probs = F.softmax(logits / self.softmax_temp, dim=-1)
        probs = self.focus_layer(probs.view(1, 1, -1)).squeeze(0).squeeze(0)

        token_boost: Dict[str, float] = {}
        for w, p in zip(vocab100, probs.detach().cpu().tolist()):
            for subw in w.split():
                if len(subw) > 2 and subw not in STOPWORDS:
                    token_boost[subw] = max(token_boost.get(subw, 0.0), math.log(p + 1e-12) + 5.0)

        G_sem = nx.Graph()
        W_np = W.detach().cpu().numpy()
        for i in range(W_np.shape[0]):
            for j in range(W_np.shape[1]):
                if W_np[i, j] > 0.05:
                    G_sem.add_edge(f"N{i}", f"B{j}", weight=float(W_np[i, j]))

        return ModelState(
            nodelets=nodelets,
            vocab100=vocab100,
            binding_W=W,
            bar_probs=probs,
            token_boost=token_boost,
            pillar_weights=torch.zeros_like(probs),
            geometric_bias=torch.zeros_like(probs),
            semantic_graph=G_sem,
            lm_graph=nx.DiGraph(),
        )

    def _final_probs_for_context(
        self,
        lm: QuadgramLM,
        token_boost: Dict[str, float],
        w1: str, w2: str, w3: str
    ) -> Tuple[List[str], torch.Tensor]:
        cand, base_probs = lm.next_distribution(w1, w2, w3)

        if len(cand) == 0:
            cand = lm.vocab[:100] if lm.vocab else ["the", "is", "a"]
            base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))
        else:
            base_p = base_probs.detach().clone().to(dtype=torch.float32)

        if base_p.numel() != len(cand):
            base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))

        base_p = base_p.view(-1)
        base_p = base_p / (base_p.sum() + 1e-12)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

        boosts = torch.tensor([token_boost.get(w, 0.0) for w in cand], dtype=torch.float32).view(-1)
        bias = self.synthetic_bias(base_p, boosts).view(-1)

        final_probs = self.gate_layer(base_p, boosts + bias, temp=0.9)
        return cand, final_probs

    def generate_report(
        self,
        text: str,
        n_takeaways: int = 7,
        seed: int = 7,
        text_seed: str = "",
        progress=None,
        overlap_tokens: int = 3,
        max_steps_min: int = 800,
        max_steps_max: int = 900,
    ) -> str:
        rng = np.random.default_rng(int(seed))
        state = self.build_state(text, progress)

        tokens = basic_tokenize(text)
        lm = QuadgramLM(self.lm_add_k)
        lm.ingest(tokens)

        ref_sig = self._graph_signature(self._build_token_structure_graph(basic_tokenize(text)))
        seed_words = basic_tokenize(text_seed) if text_seed else []

        # persistent cross-fold context across takeaways
        w1, w2, w3 = self._pick_initial_context(lm, rng, seed_words)

        takeaways: List[str] = []
        overlap_tokens = int(max(0, min(16, overlap_tokens)))

        for i in range(int(n_takeaways)):
            if progress:
                progress(i / max(1, int(n_takeaways)), desc=f"Generating {i+1}/{n_takeaways}")

            best_sent_tokens: List[str] = []
            best_tail = (w1, w2, w3)

            # rejection loop; each attempt starts from current (w1,w2,w3)
            for _attempt in range(10):
                tokens_out = [w1, w2, w3] if overlap_tokens >= 3 else [w3]
                cw1, cw2, cw3 = w1, w2, w3

                # generate
                for _step in range(int(rng.integers(max_steps_min, max_steps_max))):
                    cand, probs = self._final_probs_for_context(lm, state.token_boost, cw1, cw2, cw3)

                    p = probs.detach().cpu().numpy()
                    p = p / (p.sum() + 1e-12)

                    nxt = rng.choice(cand, p=p)
                    tokens_out.append(nxt)
                    cw1, cw2, cw3 = cw2, cw3, nxt

                    if nxt in [".", "!", "?"] and len([t for t in tokens_out if t.isalpha()]) > 200:
                        break

                # evaluate graph gate
                sent = detokenize(tokens_out)
                out_sig = self._graph_signature(self._build_token_structure_graph(basic_tokenize(sent)))

                if self._passes_automorphism_checks(ref_sig, out_sig):
                    best_sent_tokens = tokens_out
                    best_tail = (cw1, cw2, cw3)
                    break

                # fallback keep last attempt
                best_sent_tokens = tokens_out
                best_tail = (cw1, cw2, cw3)

            # update cross-fold context for next takeaway
            w1, w2, w3 = best_tail

            # remove overlapped prefix to avoid visible duplication
            if i == 0 or overlap_tokens <= 0:
                printable = best_sent_tokens
            else:
                printable = best_sent_tokens[overlap_tokens:] if len(best_sent_tokens) > overlap_tokens else best_sent_tokens

            takeaways.append(detokenize(printable))

        return "\n\n".join(takeaways)


# ----------------------------
# Minimal "SGLang-like" DSL (no external libs)
# ----------------------------
class SGPrompt:
    """
    Minimal prompt-state object with += behavior.
    """
    def __init__(self, text: str = ""):
        self.text = str(text)

    def __iadd__(self, other: str):
        self.text += str(other)
        return self

    def __str__(self):
        return self.text


class SGContext:
    """
    Bridges SGPrompt <-> NeuroSymbolicGraphGenerator.
    """
    def __init__(
        self,
        corpus_text: str,
        generator: NeuroSymbolicGraphGenerator,
        seed: int = 7,
    ):
        self.corpus_text = normalize(corpus_text)
        self.generator = generator
        self.seed = int(seed)

    def clone(self, seed_offset: int) -> "SGContext":
        return SGContext(
            corpus_text=self.corpus_text,
            generator=self.generator,
            seed=self.seed + int(seed_offset),
        )


def sg_gen(
    ctx: SGContext,
    prompt: SGPrompt,
    max_tokens: int = 240,
    seed_offset: int = 0,
) -> str:
    """
    "gen": use prompt.text as text_seed and generate 1 takeaway.
    """
    steps_min = max(120, int(max_tokens) * 2)
    steps_max = max(180, int(max_tokens) * 3)

    out = ctx.generator.generate_report(
        ctx.corpus_text,
        n_takeaways=1,
        seed=ctx.seed + int(seed_offset),
        text_seed=prompt.text,
        progress=None,
        overlap_tokens=3,
        max_steps_min=steps_min,
        max_steps_max=steps_max,
    )
    return out.strip()


def sg_select(options: List[str], scores: Optional[List[float]] = None) -> str:
    """
    "select": deterministic argmax selection on scores; otherwise first option.
    """
    if not options:
        return ""
    if not scores or len(scores) != len(options):
        return options[0]
    idx = max(range(len(options)), key=lambda i: float(scores[i]))
    return options[idx]


def sg_fork(ctx: SGContext, prompt: SGPrompt, n: int) -> List[Tuple[SGContext, SGPrompt]]:
    """
    "fork": create N prompt-state clones (synchronous).
    """
    out: List[Tuple[SGContext, SGPrompt]] = []
    n = int(max(1, n))
    for i in range(n):
        out.append((ctx.clone(seed_offset=1000 + i), SGPrompt(prompt.text)))
    return out


def sg_join(prompts: List[SGPrompt], joiner: str = "\n\n") -> SGPrompt:
    """
    "join": merge prompt texts into one.
    """
    merged = SGPrompt("")
    merged.text = joiner.join(p.text for p in prompts)
    return merged


def run_sglang_style_program(
    infile: str,
    n_take: int,
    seed: int,
    steer: float,
    focus: float,
    gelu_seed: int,
    trained_state: Optional[dict] = None,
) -> str:
    """
    A structured program:
      1) fork N branches to generate takeaways
      2) join the results
      3) generate a short summary based on the joined text
    """
    corpus_text = load_text(infile)

    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        gelu_seed=int(gelu_seed),
    )

    # load trained GELU bias weights if present
    if isinstance(trained_state, dict) and "gelu_state_dict" in trained_state:
        try:
            gen.synthetic_bias.load_state_dict(trained_state["gelu_state_dict"], strict=True)
        except Exception:
            pass
        gen.synthetic_bias.freeze_(True)
        gen.synthetic_bias.eval()

    ctx = SGContext(corpus_text, gen, seed=int(seed))

    root = SGPrompt("Generate a technical takeaway based on the document.\n\n")
    branches = sg_fork(ctx, root, n=int(n_take))

    take_prompts: List[SGPrompt] = []
    for i, (bctx, bp) in enumerate(branches):
        bp += f"[Takeaway {i+1}] "
        bp += sg_gen(bctx, bp, max_tokens=220, seed_offset=i)
        take_prompts.append(bp)

    merged = sg_join(take_prompts, joiner="\n\n")

    summary = SGPrompt(
        "Summarize these takeaways in 1–2 compact paragraphs, preserving technical terms:\n\n"
        + merged.text
        + "\n\nSummary:"
    )
    return sg_gen(ctx, summary, max_tokens=260, seed_offset=999)


# ----------------------------
# Gradio functions: Train + Generate
# ----------------------------
def train_bias_net(
    infile,
    seed,
    steer,
    focus,
    gelu_seed,
    train_steps,
    lr,
    max_contexts,
    progress=gr.Progress()
):
    text = load_text(infile)

    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        gelu_seed=int(gelu_seed),
    )

    progress(0.0, desc="Building state")
    state = gen.build_state(text, progress)

    tokens = basic_tokenize(text)
    if len(tokens) < 10:
        return None, "Not enough tokens to train."

    lm = QuadgramLM(gen.lm_add_k)
    lm.ingest(tokens)

    # Make GELU net trainable
    gen.synthetic_bias.reset_seed(int(gelu_seed))
    gen.synthetic_bias.freeze_(False)
    gen.synthetic_bias.train()
    gen.gate_layer.eval()
    gen.focus_layer.eval()

    opt = optim.Adam(gen.synthetic_bias.parameters(), lr=float(lr))

    # contexts are tokens[i-3:i] -> predict tokens[i]
    positions = list(range(3, len(tokens)))
    if max_contexts and int(max_contexts) > 0:
        positions = positions[: min(len(positions), int(max_contexts))]

    if len(positions) == 0:
        return None, "No training contexts available."

    rng = np.random.default_rng(int(seed))
    batch_size = 24

    running_loss = 0.0
    running_used = 0

    steps = int(train_steps)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        used = 0

        batch_pos = rng.choice(positions, size=min(batch_size, len(positions)), replace=False)

        for i in batch_pos:
            w1, w2, w3 = tokens[i - 3], tokens[i - 2], tokens[i - 1]
            true_next = tokens[i]

            cand, probs = gen._final_probs_for_context(lm, state.token_boost, w1, w2, w3)

            # supervised only when true is in candidate set
            try:
                j = cand.index(true_next)
            except ValueError:
                continue

            # NLL
            nll = -torch.log(probs[j].clamp_min(1e-12))
            loss_acc = loss_acc + nll
            used += 1

        if used == 0:
            continue

        loss = loss_acc / used
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.synthetic_bias.parameters(), 1.0)
        opt.step()

        running_loss += float(loss.detach().cpu().item())
        running_used += used

        if (step + 1) % max(1, steps // 20) == 0:
            progress((step + 1) / steps, desc=f"Training GELU bias ({step+1}/{steps})")

    avg_loss = running_loss / max(1, (steps if steps > 0 else 1))
    msg = f"Trained SyntheticGELUBias. Avg batch loss={avg_loss:.4f}. Used={running_used} samples."

    trained = {
        "gelu_state_dict": {k: v.detach().cpu() for k, v in gen.synthetic_bias.state_dict().items()},
        "gelu_seed": int(gelu_seed),
        "focus": float(focus),
        "steer": float(steer),
    }
    return trained, msg


def generate_with_optional_training(
    infile,
    n_take,
    seed,
    t_seed,
    steer,
    focus,
    gelu_seed,
    trained_state,
    progress=gr.Progress()
):
    text = load_text(infile)

    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        gelu_seed=int(gelu_seed),
    )

    # If trained weights exist, load them (ignore if shapes mismatch)
    if isinstance(trained_state, dict) and "gelu_state_dict" in trained_state:
        try:
            gen.synthetic_bias.load_state_dict(trained_state["gelu_state_dict"], strict=True)
        except Exception:
            pass
        gen.synthetic_bias.freeze_(True)
        gen.synthetic_bias.eval()

    return gen.generate_report(text, int(n_take), int(seed), t_seed, progress)


def sglang_style_button(
    infile,
    n_take,
    seed,
    steer,
    focus,
    gelu_seed,
    trained_state,
):
    return run_sglang_style_program(
        infile=infile,
        n_take=int(n_take),
        seed=int(seed),
        steer=float(steer),
        focus=float(focus),
        gelu_seed=int(gelu_seed),
        trained_state=trained_state,
    )


# ----------------------------
# Gradio UI
# ----------------------------
def build_app():
    with gr.Blocks(title="Neurosymbolic V3.2 (Trainable GELU Bias + SGLang-like)") as demo:
        gr.Markdown("# Neurosymbolic Text Generator V3.2\n*Add Train button (GELU bias fine-tune) + SGLang-like program layer*")

        trained_state = gr.State(None)

        with gr.Row():
            infile = gr.File(label="Input File", type="filepath")
            out_txt = gr.Textbox(label="Output", lines=15)

        status = gr.Textbox(label="Status", lines=2)

        with gr.Row():
            n_take = gr.Slider(1, 20, value=5, label="Takeaways")
            seed = gr.Number(value=42, label="Global Seed")

        with gr.Row():
            steer = gr.Slider(0, 5, value=1.35, label="Steer Strength")
            focus = gr.Slider(0, 1, value=0.5, label="Focus Strength (Lateral Inhibition)")
            gelu_seed = gr.Number(value=1337, label="GELU Init Seed")

        with gr.Row():
            train_steps = gr.Slider(10, 2000, value=250, step=10, label="Train steps")
            lr = gr.Number(value=1e-3, label="LR")
            max_contexts = gr.Slider(0, 20000, value=4000, step=100, label="Max contexts (0=all)")

        t_seed = gr.Textbox(label="Text Seed")

        with gr.Row():
            train_btn = gr.Button("Train (GELU Bias)", variant="secondary")
            sg_btn = gr.Button("SGLang-style Program", variant="primary")

        train_btn.click(
            train_bias_net,
            inputs=[infile, seed, steer, focus, gelu_seed, train_steps, lr, max_contexts],
            outputs=[trained_state, status],
        )

        gen_btn.click(
            generate_with_optional_training,
            inputs=[infile, n_take, seed, t_seed, steer, focus, gelu_seed, trained_state],
            outputs=out_txt,
        )

        sg_btn.click(
            sglang_style_button,
            inputs=[infile, n_take, seed, steer, focus, gelu_seed, trained_state],
            outputs=out_txt,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()

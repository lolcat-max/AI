#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-Theoretic Neurosymbolic Text Generator (Gradio GUI)
V3.3 + SGLang-like Neural Runtime (Continuous Batching, Radix Cache, Speculative Decoding)

New in V3.3:
- Full "Continuous Batching" inference engine (no simple loops).
- "RadixLRUCache" reuses expensive neural distributions across shared prefixes.
- "Speculative Decoding" (propose n-gram, verify with gated GELU) inside the batch loop.
- SGLang-style primitives (fork, join, gen, select) backed by the batched runtime.

Dependencies:
  pip install gradio numpy scikit-learn networkx tqdm datasets pypdf python-docx torch
"""

from __future__ import annotations
import re
import math
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from collections import OrderedDict, deque

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
# SGLang-like Runtime Components
# ----------------------------
class RadixLRUCache:
    """
    Stores per-context decoding results (neural distributions) to emulate
    Prefix KV Cache reuse.
    Key: (cache_version, w1, w2, w3) -> Value: (candidates, probs_tensor)
    """
    def __init__(self, max_items: int = 25000):
        self.max_items = int(max(256, max_items))
        self._od: "OrderedDict[Tuple[int, str, str, str], Tuple[List[str], torch.Tensor]]" = OrderedDict()

    def get(self, key):
        v = self._od.get(key, None)
        if v is None:
            return None
        self._od.move_to_end(key)
        return v

    def put(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        while len(self._od) > self.max_items:
            self._od.popitem(last=False)

    def clear(self):
        self._od.clear()


@dataclass
class DecodeStream:
    """
    Represents a single active generation request (like a 'sequence' in SGLang).
    """
    stream_id: int
    tokens_out: List[str]
    w1: str
    w2: str
    w3: str
    done: bool = False
    alpha_count: int = 0
    max_steps: int = 1000
    stop_tokens: set = field(default_factory=lambda: {".", "!", "?"})
    min_alpha: int = 200  # min alpha tokens before stopping on punctuation


# ----------------------------
# PyTorch Neural Modules
# ----------------------------
class LateralInhibition(nn.Module):
    def __init__(self, kernel_size=7, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.95, -0.9, -0.1, 0.3, -1.4, -1.2, -1.05], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = int(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = torch.stack([x1, token_boosts], dim=-1)
        h = self.act(self.fc1(x))
        return self.fc2(h).squeeze(-1)


# ----------------------------
# Utils & Preprocessing
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
                if out: out[-1] += t
                else: out.append(t)
        else:
            if out and out[-1].endswith("("): out[-1] += t
            else: out.append(t)
    s = " ".join(out)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s

def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext in [".txt", ".md"]: return p.read_text(encoding="utf-8", errors="replace")
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
# Quadgram LM
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
            if a == w1 and b == w2 and c == w3: cont.append(d)
        if not cont:
            for (a, b, c), count in self.tri.items():
                if a == w2 and b == w3: cont.append(c)
        if not cont:
            for (a, b), count in self.bi.items():
                if a == w3: cont.append(b)
        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)[:200]]

        seen = set(); cand = []
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
            if c123 > 0: return (c1234 + add_k) / (c123 + add_k * V)
            c12 = self.bi.get((w2, w3), 0)
            c123_tri = self.tri.get((w2, w3, w4), 0)
            if c12 > 0: return (c123_tri + add_k) / (c12 + add_k * V)
            c1 = self.uni.get(w3, 0)
            c12_bi = self.bi.get((w3, w4), 0)
            if c1 > 0: return (c12_bi + add_k) / (c1 + add_k * V)
            return (self.uni.get(w4, 0) + add_k) / (self.total + add_k * V)

        probs = torch.tensor([get_prob(w) for w in cand], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# ----------------------------
# System State & Preparation
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

@dataclass
class PreparedCorpus:
    text: str
    tokens: List[str]
    lm: QuadgramLM
    state: ModelState
    ref_sig: Dict[str, object]

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
        radix_cache_items: int = 25000,
        speculative_accept_topk: int = 10,
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
        self.synthetic_bias.freeze_(True)
        self.pruner: Optional[SynapticPruner] = None

        # SGLang runtime state
        self.cache_version = 0
        self.radix_cache = RadixLRUCache(max_items=int(radix_cache_items))
        self.speculative_accept_topk = int(speculative_accept_topk)

    def bump_cache_version(self):
        self.cache_version += 1
        self.radix_cache.clear()

    def _token_class(self, tok: str) -> str:
        if tok in [".", ",", ";", ":", "!", "?", "(", ")"]: return "PUNC"
        if not re.match(r"[a-z]", tok): return "OTHER"
        L = len(tok)
        return "S" if L <= 3 else "M" if L <= 7 else "L"

    def _pick_initial_context(self, lm: QuadgramLM, rng: np.random.Generator, seed_words: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in seed_words if re.match(r"^[a-z][a-z0-9_\-']*$", t)]
        if len(sw) >= 3: return (sw[-3], sw[-2], sw[-1])
        if len(sw) == 2: return (sw[-2], sw[-1], sw[-1])
        if len(sw) == 1: return (sw[-1], sw[-1], sw[-1])
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
            if G.is_directed() else {}
        )
        GM = iso.DiGraphMatcher(G, G, **match_kwargs) if G.is_directed() else iso.GraphMatcher(G, G, **match_kwargs)
        cnt = 0
        for _ in GM.isomorphisms_iter():
            cnt += 1
            if cnt >= 150: break
        return {"deg_hist": np.bincount(deg, minlength=16)[:16], "wl": wl, "aut_est": cnt}

    def _passes_automorphism_checks(self, ref_sig, out_sig) -> bool:
        strict = max(0.0, min(2.0, self.geometric_strength))
        ref = ref_sig["deg_hist"].astype(float); ref /= (ref.sum() + 1e-12)
        out = out_sig["deg_hist"].astype(float); out /= (out.sum() + 1e-12)
        if np.abs(ref - out).sum() > max(0.25, 1.10 - 0.35 * strict): return False
        ratio = max(1, out_sig["aut_est"]) / max(1, ref_sig["aut_est"])
        band = max(1.3, 3.5 - 1.2 * min(1.0, self.gate_layer.steer_strength / 2.0))
        if not (1.0 / band <= ratio <= band): return False
        if strict >= 1.6 and out_sig["wl"] != ref_sig["wl"]: return False
        return True

    def _synaptic_prune(self, W: torch.Tensor, energies: torch.Tensor, vocab100: List[str], progress=None):
        if not self.rfe_enabled or self.rfe_iterations <= 0: return W, vocab100
        k, bars_n = W.shape
        self.pruner = SynapticPruner(bars_n)
        W_curr = W.detach().clone().requires_grad_(True)
        kept_mask = torch.ones(bars_n, dtype=torch.bool)

        for iteration in range(self.rfe_iterations):
            if progress: progress(0.80 + 0.05 * (iteration / max(1, self.rfe_iterations)), desc=f"Synaptic Pruning {iteration+1}")
            W_modulated = self.pruner(W_curr)
            loss = -torch.sum(W_modulated * energies.view(-1, 1)) + 0.1 * torch.var(W_modulated, dim=0).sum()
            loss.backward()
            with torch.no_grad():
                grads = W_curr.grad.abs().sum(dim=0)
                weights = W_curr.abs().sum(dim=0)
                importance = 0.6 * weights + 0.4 * grads
                importance = importance / (importance.max() + 1e-12)
                n_keep = int(kept_mask.sum().item() * (1.0 - self.rfe_removal_rate))
                if n_keep < 10: break
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
        if progress: progress(0, desc="Normalizing")
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
                if W_np[i, j] > 0.05: G_sem.add_edge(f"N{i}", f"B{j}", weight=float(W_np[i, j]))

        return ModelState(
            nodelets=nodelets, vocab100=vocab100, binding_W=W, bar_probs=probs,
            token_boost=token_boost, pillar_weights=torch.zeros_like(probs),
            geometric_bias=torch.zeros_like(probs), semantic_graph=G_sem, lm_graph=nx.DiGraph()
        )

    def prepare_corpus(self, text: str, progress=None) -> PreparedCorpus:
        text = normalize(text)
        state = self.build_state(text, progress)
        tokens = basic_tokenize(text)
        lm = QuadgramLM(self.lm_add_k)
        lm.ingest(tokens)
        ref_sig = self._graph_signature(self._build_token_structure_graph(tokens))
        return PreparedCorpus(text=text, tokens=tokens, lm=lm, state=state, ref_sig=ref_sig)

    # ----------------------------
    # Neural Runtime Methods
    # ----------------------------
    def _final_probs_for_context_cached(
        self,
        prep: PreparedCorpus,
        w1: str, w2: str, w3: str
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Computes neural gated probs. Uses RadixLRUCache to reuse results for same prefix.
        """
        key = (int(self.cache_version), str(w1), str(w2), str(w3))
        cached = self.radix_cache.get(key)
        if cached is not None:
            return cached

        cand, base_probs = prep.lm.next_distribution(w1, w2, w3)
        if len(cand) == 0:
            cand = prep.lm.vocab[:100] if prep.lm.vocab else ["the", "is", "a"]
            base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))
        else:
            base_p = base_probs.detach().clone().to(dtype=torch.float32)

        if base_p.numel() != len(cand):
            base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))

        base_p = base_p.view(-1)
        base_p = base_p / (base_p.sum() + 1e-12)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

        boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand], dtype=torch.float32).view(-1)
        bias = self.synthetic_bias(base_p, boosts).view(-1)
        final_probs = self.gate_layer(base_p, boosts + bias, temp=0.9).view(-1)

        self.radix_cache.put(key, (cand, final_probs.detach().clone()))
        return cand, final_probs


# ----------------------------
# Continuous Batching Decoder
# ----------------------------
class ContinuousBatchDecoder:
    """
    SGLang-like Inference Scheduler.
    - Manages multiple DecodeStreams.
    - Groups them by prefix to exploit RadixCache.
    - Runs batched Speculative Decoding (Propose -> Verify).
    """
    def __init__(
        self,
        gen: "NeuroSymbolicGraphGenerator",
        prep: PreparedCorpus,
        rng: np.random.Generator,
        token_budget_per_round: int = 64,
        speculative: bool = True,
    ):
        self.gen = gen
        self.prep = prep
        self.rng = rng
        self.token_budget_per_round = int(max(1, token_budget_per_round))
        self.speculative = bool(speculative)

    def _sample_from_probs(self, cand: List[str], probs: torch.Tensor) -> str:
        p = probs.detach().cpu().numpy()
        p = p / (p.sum() + 1e-12)
        return self.rng.choice(cand, p=p)

    def _propose_token_base(self, w1: str, w2: str, w3: str) -> str:
        cand, base_probs = self.prep.lm.next_distribution(w1, w2, w3)
        if not cand: return w3
        p = base_probs.detach().cpu().numpy()
        p = p / (p.sum() + 1e-12)
        return self.rng.choice(cand, p=p)

    def step_round(self, streams: List[DecodeStream]) -> None:
        active = [s for s in streams if not s.done]
        if not active: return

        # Sort by prefix to maximize cache hits (emulates RadixAttention tree traversal)
        active.sort(key=lambda s: (s.w1, s.w2, s.w3))
        active = active[: min(len(active), self.token_budget_per_round)]

        # Group by identical prefixes
        groups: Dict[Tuple[str, str, str], List[DecodeStream]] = {}
        for s in active:
            groups.setdefault((s.w1, s.w2, s.w3), []).append(s)

        for (w1, w2, w3), bucket in groups.items():
            # Neural step (expensive, possibly cached)
            cand, final_probs = self.gen._final_probs_for_context_cached(self.prep, w1, w2, w3)

            # Precompute acceptance set for speculative decoding
            if self.speculative:
                topk = min(self.gen.speculative_accept_topk, len(cand))
                _, idx = torch.topk(final_probs, k=topk)
                topk_set = set(idx.detach().cpu().tolist())

            for s in bucket:
                nxt = None
                if self.speculative:
                    # Propose cheaply
                    proposed = self._propose_token_base(s.w1, s.w2, s.w3)
                    try: j = cand.index(proposed)
                    except ValueError: j = -1

                    # Verify
                    if j >= 0 and j in topk_set:
                        nxt = proposed
                    else:
                        # Fallback to expensive sample
                        nxt = self._sample_from_probs(cand, final_probs)
                else:
                    nxt = self._sample_from_probs(cand, final_probs)

                s.tokens_out.append(nxt)
                if nxt.isalpha():
                    s.alpha_count += 1
                s.w1, s.w2, s.w3 = s.w2, s.w3, nxt

                # Stop conditions
                if s.alpha_count >= s.max_steps:
                    s.done = True
                elif nxt in s.stop_tokens and s.alpha_count > s.min_alpha:
                    s.done = True


# ----------------------------
# SGLang-like DSL & Primitives
# ----------------------------
class SGPrompt:
    def __init__(self, text: str = ""):
        self.text = str(text)
        self.fields: Dict[str, str] = {}

    def __iadd__(self, other: str):
        self.text += str(other)
        return self

    def __getitem__(self, key: str) -> str: return self.fields.get(key, "")
    def __setitem__(self, key: str, value: str): self.fields[str(key)] = str(value)
    def __str__(self): return self.text


class SGContext:
    def __init__(
        self,
        corpus_text: str,
        generator: NeuroSymbolicGraphGenerator,
        seed: int = 7,
        prepared: Optional[PreparedCorpus] = None,
    ):
        self.corpus_text = normalize(corpus_text)
        self.generator = generator
        self.seed = int(seed)
        self.prepared: Optional[PreparedCorpus] = prepared

    def ensure_prepared(self):
        if self.prepared is None:
            self.prepared = self.generator.prepare_corpus(self.corpus_text)

    def clone(self, seed_offset: int) -> "SGContext":
        return SGContext(
            corpus_text=self.corpus_text,
            generator=self.generator,
            seed=self.seed + int(seed_offset),
            prepared=self.prepared
        )


def sg_gen_batched(
    ctxs: List[SGContext],
    prompts: List[SGPrompt],
    max_tokens: int = 240,
    seed_offsets: Optional[List[int]] = None,
    stop_at_punc: bool = True
) -> List[str]:
    """
    Batched 'gen' primitive.
    """
    if not ctxs: return []
    gen = ctxs[0].generator
    prep = ctxs[0].prepared
    if prep is None:
        ctxs[0].ensure_prepared()
        prep = ctxs[0].prepared

    rng = np.random.default_rng(ctxs[0].seed)
    streams = []

    for i, (ctx, prompt) in enumerate(zip(ctxs, prompts)):
        off = seed_offsets[i] if seed_offsets else i
        local_rng = np.random.default_rng(ctx.seed + off)
        
        seed_words = basic_tokenize(prompt.text)
        w1, w2, w3 = gen._pick_initial_context(prep.lm, local_rng, seed_words)
        
        streams.append(DecodeStream(
            stream_id=i, tokens_out=[w1, w2, w3], w1=w1, w2=w2, w3=w3,
            max_steps=max_tokens, min_alpha=max_tokens // 2 if stop_at_punc else 99999
        ))

    decoder = ContinuousBatchDecoder(gen, prep, rng, token_budget_per_round=64, speculative=True)

    # Decode loop
    for _ in range(max_tokens * 2):
        if all(s.done for s in streams): break
        decoder.step_round(streams)

    results = []
    for s in streams:
        # trim initial context
        out_toks = s.tokens_out[3:] if len(s.tokens_out) > 3 else []
        results.append(detokenize(out_toks))
    return results


def sg_gen(ctx: SGContext, prompt: SGPrompt, max_tokens=240, seed_offset=0) -> str:
    res = sg_gen_batched([ctx], [prompt], max_tokens, [seed_offset])
    return res[0]


def sg_fork(ctx: SGContext, prompt: SGPrompt, n: int) -> List[Tuple[SGContext, SGPrompt]]:
    n = int(max(1, n))
    ctx.ensure_prepared()
    out = []
    for i in range(n):
        out.append((ctx.clone(seed_offset=1000 + i), SGPrompt(prompt.text)))
    return out


def sg_join(prompts: List[SGPrompt], joiner: str = "\n\n") -> SGPrompt:
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
    corpus_text = load_text(infile)

    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        gelu_seed=int(gelu_seed),
        radix_cache_items=30000,
        speculative_accept_topk=10,
    )

    if isinstance(trained_state, dict) and "gelu_state_dict" in trained_state:
        try: gen.synthetic_bias.load_state_dict(trained_state["gelu_state_dict"], strict=True)
        except Exception: pass
        gen.synthetic_bias.freeze_(True)
        gen.synthetic_bias.eval()
        gen.bump_cache_version()

    ctx = SGContext(corpus_text, gen, seed=int(seed))
    ctx.ensure_prepared()

    # 1. Fork prompt
    root = SGPrompt("Generate a technical takeaway based on the document.\n\n")
    branches = sg_fork(ctx, root, n=int(n_take))
    
    # 2. Batched generation of takeaways
    branch_ctxs = [b[0] for b in branches]
    branch_prompts = [b[1] for b in branches]
    
    # Add labels
    for i, bp in enumerate(branch_prompts):
        bp += f"[Takeaway {i+1}] "

    # Run inference in one batched engine
    take_texts = sg_gen_batched(branch_ctxs, branch_prompts, max_tokens=220, stop_at_punc=True)

    # 3. Join
    for i, txt in enumerate(take_texts):
        branch_prompts[i] += txt
        
    merged = sg_join(branch_prompts, joiner="\n\n")

    # 4. Summary (single stream)
    summary_prompt = SGPrompt(
        "explain these concepts in easy to read paragraphs:\n\n" + merged.text + "\n\nSummary:"
    )
    summary_text = sg_gen(ctx, summary_prompt, max_tokens=260)
    
    return summary_prompt.text + summary_text


# ----------------------------
# Gradio Training & App
# ----------------------------
def train_bias_net(
    infile, seed, steer, focus, gelu_seed, train_steps, lr, max_contexts, progress=gr.Progress()
):
    text = load_text(infile)
    gen = NeuroSymbolicGraphGenerator(steer_strength=float(steer), focus_strength=float(focus), gelu_seed=int(gelu_seed))
    
    progress(0.0, desc="Building state")
    prep = gen.prepare_corpus(text)
    tokens = prep.tokens
    
    if len(tokens) < 10: return None, "Not enough tokens."

    gen.synthetic_bias.reset_seed(int(gelu_seed))
    gen.synthetic_bias.freeze_(False)
    gen.synthetic_bias.train()
    
    opt = optim.Adam(gen.synthetic_bias.parameters(), lr=float(lr))
    positions = list(range(3, len(tokens)))
    if max_contexts and int(max_contexts) > 0:
        positions = positions[: min(len(positions), int(max_contexts))]

    rng = np.random.default_rng(int(seed))
    batch_size = 24
    running_loss = 0.0
    
    steps = int(train_steps)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0; used = 0
        batch_pos = rng.choice(positions, size=min(batch_size, len(positions)), replace=False)

        for i in batch_pos:
            w1, w2, w3 = tokens[i-3], tokens[i-2], tokens[i-1]
            true_next = tokens[i]
            
            # Direct forward (no cache)
            cand, base_probs = prep.lm.next_distribution(w1, w2, w3)
            if not cand: continue
            
            base_p = base_probs.detach().clone().to(dtype=torch.float32)
            base_p = base_p / (base_p.sum() + 1e-12)
            base_p = gen.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)
            boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand]).view(-1)
            bias = gen.synthetic_bias(base_p, boosts).view(-1)
            probs = gen.gate_layer(base_p, boosts + bias, temp=0.9)

            try: j = cand.index(true_next)
            except ValueError: continue
            
            loss_acc -= torch.log(probs[j].clamp_min(1e-12))
            used += 1

        if used > 0:
            loss = loss_acc / used
            loss.backward()
            opt.step()
            running_loss += float(loss.item())

        if (step+1) % max(1, steps//10) == 0:
            progress((step+1)/steps, desc=f"Training {step+1}/{steps}")

    return {
        "gelu_state_dict": {k: v.detach().cpu() for k,v in gen.synthetic_bias.state_dict().items()}
    }, f"Trained. Avg loss={running_loss/max(1,steps):.4f}"


def build_app():
    with gr.Blocks(title="Neurosymbolic V3.3 (SGLang Runtime)") as demo:
        gr.Markdown("# Neurosymbolic V3.3: SGLang-like Runtime\n*Continuous Batching, RadixCache, Speculative Decoding*")
        
        trained_state = gr.State(None)
        
        with gr.Row():
            infile = gr.File(label="Input File")
            out_txt = gr.Textbox(label="Structured Output", lines=20)
            
        with gr.Row():
            n_take = gr.Slider(1, 10, value=4, label="Parallel Forks (Batch Size)")
            seed = gr.Number(value=42, label="Seed")
            
        with gr.Row():
            steer = gr.Slider(0, 5, value=1.35, label="Steer")
            focus = gr.Slider(0, 1, value=0.5, label="Focus")
            gelu_seed = gr.Number(value=1337, label="GELU Seed")
            
        train_btn = gr.Button("Train GELU Bias (Optional)")
        run_btn = gr.Button("Run Structured Program (SGLang style)", variant="primary")
        status = gr.Textbox(label="Train Status")
        
        train_btn.click(
            train_bias_net,
            inputs=[infile, seed, steer, focus, gelu_seed, gr.Number(100, visible=False), gr.Number(0.001, visible=False), gr.Number(0, visible=False)],
            outputs=[trained_state, status]
        )
        
        run_btn.click(
            run_sglang_style_program,
            inputs=[infile, n_take, seed, steer, focus, gelu_seed, trained_state],
            outputs=out_txt
        )
        
    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurosymbolic Graph Text Generator (Gradio) — refactor/compress w/ "piecemeal unfold"

What you asked for:
- "compressing all ulterior probs": all probability shaping (softmax, pillars, geometric, concave focus)
  is composed into ONE pipeline function.
- "piecemeal unfold": the pipeline can be unfolded step-by-step (debug mode) to inspect intermediate
  distributions and biases.

Deps:
  pip install gradio numpy scikit-learn networkx tqdm datasets pypdf python-docx
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from networkx.algorithms import isomorphism as iso
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


# -----------------------------
# Basics
# -----------------------------
STOPWORDS = set(
    """
a an and are as at be by for from has have he her hers him his i in is it its me my
of on or our ours she so that the their them they this to was we were what when where
which who will with you your yours
""".split()
)

PUNC = {".", ",", ";", ":", "!", "?", "(", ")"}


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-12)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def concave_focus(p: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Concave focus: emphasize "central" probability mass.
    strength: 0..1
    """
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    if len(p) <= 1 or strength <= 0.0:
        return p

    strength = float(max(0.0, min(1.0, strength)))

    pos = np.arange(len(p), dtype=float)
    pos = pos / (len(p) - 1) if len(p) > 1 else pos
    centroid = float(np.dot(pos, p))

    dist = np.abs(pos - centroid)
    md = float(dist.max())
    if md > 1e-12:
        dist = dist / md

    cumsum = np.cumsum(p)
    centric = np.zeros_like(p)
    for i in range(len(p)):
        a, b = max(0, i - 3), min(len(p), i + 4)
        lp = p[a:b]
        lc = cumsum[a:b]
        centric[i] = float(np.dot(lp, lc))
    centric = centric / (float(centric.max()) + 1e-12)

    w = np.sqrt((1.0 - dist) * centric + 1e-12)
    out = p * (1.0 + strength * w)
    return out / (out.sum() + 1e-12)


def basic_tokenize(text: str) -> List[str]:
    text = text.replace("\n", " ")
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-']*|[.,;:!?()]", text)
    out: List[str] = []
    for t in toks:
        out.append(t.lower() if re.match(r"[A-Za-z]", t) else t)
    return out


def detokenize(tokens: List[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if t in PUNC:
            if t == "(":
                out.append(t)
            elif t == ")":
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


# -----------------------------
# File loading
# -----------------------------
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
        return "\n".join(para.text for para in d.paragraphs)
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# HF dataset loading (split mixing + context fields)
# -----------------------------
def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    splits: str = "train",
    text_columns: Optional[str] = None,
    context_columns: Optional[str] = None,
    max_samples: Optional[int] = None,
    mix_mode: str = "off",  # off|auto|manual
    manual_mix: str = "",  # train=0.7,validation=0.3
    prompt_text: str = "",
    seed: int = 7,
    progress: Optional[gr.Progress] = None,
) -> str:
    from datasets import load_dataset

    def csv(s: Optional[str]) -> List[str]:
        return [x.strip() for x in (s or "").split(",") if x.strip()]

    def pick_default(cols: List[str]) -> List[str]:
        cand = ["text", "article", "content", "document", "context", "sentence", "review", "body"]
        hit = [c for c in cand if c in cols]
        return hit[:1] if hit else [cols[0]]

    def join_fields(item: dict, main: List[str], ctx: List[str]) -> str:
        parts: List[str] = []

        def add(k: str, label: Optional[str]):
            if k not in item:
                return
            v = item[k]
            if v is None:
                return
            if isinstance(v, (list, tuple)):
                v = "\n".join(str(x) for x in v if str(x).strip())
            v = str(v).strip()
            if not v:
                return
            parts.append(f"{label}: {v}" if label else v)

        for k in main:
            add(k, None if len(main) == 1 else k)
        for k in ctx:
            if k in main:
                continue
            add(k, k)
        return "\n".join(parts).strip()

    def load_split(sp: str):
        return load_dataset(dataset_name, config_name, split=sp) if config_name else load_dataset(dataset_name, split=sp)

    split_list = csv(splits) or ["train"]
    main_cols = csv(text_columns)
    ctx_cols = csv(context_columns)

    if progress:
        progress(0.0, desc=f"Loading dataset: {dataset_name} ({', '.join(split_list)})")

    dsets: Dict[str, object] = {}
    for i, sp in enumerate(split_list):
        if progress:
            progress(0.05 + 0.20 * (i / max(1, len(split_list))), desc=f"Loading split: {sp}")
        dsets[sp] = load_split(sp)

    if not main_cols:
        cols = list(dsets[split_list[0]].column_names)
        main_cols = pick_default(cols)

    for sp, ds in dsets.items():
        cols = set(ds.column_names)
        for c in main_cols:
            if c not in cols:
                raise ValueError(f"Column '{c}' not found in split '{sp}'. Available: {ds.column_names}")
        for c in ctx_cols:
            if c and c not in cols:
                raise ValueError(f"Context column '{c}' not found in split '{sp}'. Available: {ds.column_names}")

    rng = np.random.default_rng(int(seed))
    total_cap = int(max_samples) if (max_samples and max_samples > 0) else None

    def manual_weights(spec: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for chunk in csv(spec):
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            k = k.strip()
            try:
                out[k] = float(v.strip())
            except ValueError:
                pass
        out = {k: out[k] for k in split_list if k in out}
        s = sum(out.values())
        return {k: out[k] / s for k in out} if s > 0 else {}

    def auto_weights(prompt: str) -> Dict[str, float]:
        prompt = (prompt or "").strip()
        if not prompt:
            return {sp: 1.0 / len(split_list) for sp in split_list}

        probe_docs: List[str] = []
        names: List[str] = []
        for sp in split_list:
            ds = dsets[sp]
            n = min(200, len(ds))
            if n <= 0:
                continue
            idx = rng.choice(len(ds), size=n, replace=False) if len(ds) >= n else np.arange(len(ds))
            txts: List[str] = []
            for j in idx:
                t = join_fields(ds[int(j)], main_cols, ctx_cols)
                if t:
                    txts.append(t)
            probe_docs.append("\n".join(txts)[:12000])
            names.append(sp)

        if not probe_docs:
            return {sp: 1.0 / len(split_list) for sp in split_list}

        vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=6000, ngram_range=(1, 2))
        X = vec.fit_transform([prompt] + probe_docs)
        sims = cosine_similarity(X[0], X[1:]).ravel()
        sims = np.clip(sims, 0.0, None)
        w = softmax(sims, temp=0.7)
        return {names[i]: float(w[i]) for i in range(len(names))}

    mm = (mix_mode or "off").lower().strip()
    if mm == "manual":
        weights = manual_weights(manual_mix) or {sp: 1.0 / len(split_list) for sp in split_list}
    elif mm == "auto":
        weights = auto_weights(prompt_text)
    else:
        weights = {split_list[0]: 1.0}

    if total_cap is None:
        alloc = {sp: len(dsets[sp]) for sp in weights}
    else:
        raw = {sp: weights.get(sp, 0.0) for sp in weights}
        s = sum(raw.values())
        raw = {sp: raw[sp] / s for sp in raw} if s > 0 else {sp: 1.0 / len(raw) for sp in raw}
        alloc = {sp: int(round(total_cap * raw[sp])) for sp in raw}
        drift = total_cap - sum(alloc.values())
        if drift:
            order = sorted(raw.keys(), key=lambda k: raw[k], reverse=True)
            k = 0
            while drift and order:
                sp = order[k % len(order)]
                if drift > 0:
                    alloc[sp] += 1
                    drift -= 1
                else:
                    if alloc[sp] > 0:
                        alloc[sp] -= 1
                        drift += 1
                k += 1

    if progress:
        progress(0.35, desc="Mixing splits: " + ", ".join([f"{sp}:{alloc.get(sp,0)}" for sp in alloc]))

    texts_out: List[str] = []
    for i, sp in enumerate(alloc.keys()):
        ds = dsets[sp]
        n_take = min(int(alloc[sp]), len(ds))
        if n_take <= 0:
            continue
        if progress:
            progress(0.35 + 0.55 * (i / max(1, len(alloc))), desc=f"Extracting from '{sp}'")
        idx = rng.choice(len(ds), size=n_take, replace=False) if n_take < len(ds) else np.arange(len(ds))
        for j in idx:
            t = join_fields(ds[int(j)], main_cols, ctx_cols)
            if t and t.strip():
                texts_out.append(t.strip())

    if progress:
        progress(1.0, desc="Dataset loaded successfully")

    return "\n\n".join(texts_out)


# -----------------------------
# LM (4-gram) + transition graph
# -----------------------------
class QuadgramLM:
    def __init__(self, add_k: float = 0.25):
        self.add_k = float(add_k)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0

    def fit(self, tokens: List[str]) -> None:
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
        self.quad.clear()
        self.total = 0

        def inc(d: dict, k, v=1):
            d[k] = d.get(k, 0) + v

        for t in tokens:
            inc(self.uni, t)
            self.total += 1
        for i in range(len(tokens) - 1):
            inc(self.bi, (tokens[i], tokens[i + 1]))
        for i in range(len(tokens) - 2):
            inc(self.tri, (tokens[i], tokens[i + 1], tokens[i + 2]))
        for i in range(len(tokens) - 3):
            inc(self.quad, (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]))

        self.vocab = list(self.uni.keys())

    def _p_uni(self, w: str) -> float:
        V = len(self.vocab) + 1
        return (self.uni.get(w, 0) + self.add_k) / (self.total + self.add_k * V)

    def _p_bi(self, w1: str, w2: str) -> float:
        V = len(self.vocab) + 1
        c1 = self.uni.get(w1, 0)
        c12 = self.bi.get((w1, w2), 0)
        return (c12 + self.add_k) / (c1 + self.add_k * V) if c1 > 0 else self._p_uni(w2)

    def _p_tri(self, w1: str, w2: str, w3: str) -> float:
        V = len(self.vocab) + 1
        c12 = self.bi.get((w1, w2), 0)
        c123 = self.tri.get((w1, w2, w3), 0)
        return (c123 + self.add_k) / (c12 + self.add_k * V) if c12 > 0 else self._p_bi(w2, w3)

    def _p_quad(self, w1: str, w2: str, w3: str, w4: str) -> float:
        V = len(self.vocab) + 1
        c123 = self.tri.get((w1, w2, w3), 0)
        c1234 = self.quad.get((w1, w2, w3, w4), 0)
        return (c1234 + self.add_k) / (c123 + self.add_k * V) if c123 > 0 else self._p_tri(w2, w3, w4)

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], np.ndarray]:
        cont: List[str] = []
        for (a, b, c, d), _ in self.quad.items():
            if a == w1 and b == w2 and c == w3:
                cont.append(d)
        if not cont:
            for (a, b, c), _ in self.tri.items():
                if a == w2 and b == w3:
                    cont.append(c)
        if not cont:
            for (a, b), _ in self.bi.items():
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

        probs = np.array([self._p_quad(w1, w2, w3, w) for w in cand], dtype=float)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs

    def build_transition_graph(self, max_edges: int = 20000) -> nx.DiGraph:
        G = nx.DiGraph()
        agg: Dict[Tuple[str, str], int] = {}
        for (w1, w2, w3, w4), c in self.quad.items():
            agg[(w3, w4)] = agg.get((w3, w4), 0) + int(c)

        for (a, b), c in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:max_edges]:
            if not G.has_node(a):
                G.add_node(a, kind="tok")
            if not G.has_node(b):
                G.add_node(b, kind="tok")
            G.add_edge(a, b, weight=float(c))
        return G


# -----------------------------
# Graph + model state
# -----------------------------
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
    binding_W: np.ndarray
    bar_probs: np.ndarray
    bar_logits: np.ndarray
    token_boost: Dict[str, float]
    pillar_weights: np.ndarray
    geometric_bias: np.ndarray
    semantic_graph: nx.Graph
    lm_graph: nx.DiGraph
    # optional pipeline trace
    prob_trace: Optional[List[Tuple[str, np.ndarray]]] = None


# -----------------------------
# Generator (compressed probability pipeline + unfold)
# -----------------------------
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
        pillar_floor: float = 0.25,
        geometric_strength: float = 0.3,
        rfe_enabled: bool = True,
        rfe_iterations: int = 3,
        rfe_removal_rate: float = 0.15,
        concave_focus_strength: float = 0.5,
        unfold_probs: bool = False,  # <-- "piecemeal unfold"
    ):
        self.nodelets_n = int(nodelets_n)
        self.bars_n = int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp = float(softmax_temp)
        self.steer_strength = float(steer_strength)
        self.lm_add_k = float(lm_add_k)
        self.pillar_strength = float(pillar_strength)
        self.pillar_floor = float(pillar_floor)
        self.geometric_strength = float(geometric_strength)
        self.rfe_enabled = bool(rfe_enabled)
        self.rfe_iterations = int(rfe_iterations)
        self.rfe_removal_rate = float(rfe_removal_rate)
        self.concave_focus_strength = float(concave_focus_strength)
        self.unfold_probs = bool(unfold_probs)

    # ---- token class + output structure graph gating
    def _token_class(self, tok: str) -> str:
        if tok in PUNC:
            return "PUNC"
        if not re.match(r"[a-z]", tok):
            return "OTHER"
        L = len(tok)
        return "S" if L <= 3 else "M" if L <= 7 else "L"

    def _token_structure_graph(self, tokens: List[str], max_nodes: int = 220) -> nx.DiGraph:
        toks = tokens[:max_nodes]
        G = nx.DiGraph()
        for i, t in enumerate(toks):
            G.add_node(i, cls=self._token_class(t))
        for i in range(len(toks) - 1):
            G.add_edge(i, i + 1, rel="adj")
        for i in range(len(toks) - 2):
            G.add_edge(i, i + 2, rel="skip")
        return G

    def _estimate_aut(self, G: nx.Graph, limit: int = 150) -> int:
        node_match = iso.categorical_node_match("cls", default=None) if G.is_directed() else None
        edge_match = iso.categorical_edge_match("rel", default=None) if G.is_directed() else None
        GM = iso.DiGraphMatcher(G, G, node_match=node_match, edge_match=edge_match) if G.is_directed() else iso.GraphMatcher(G, G)
        cnt = 0
        for _ in GM.isomorphisms_iter():
            cnt += 1
            if cnt >= limit:
                break
        return cnt

    def _graph_sig(self, G: nx.Graph) -> Dict[str, object]:
        deg = [d for _, d in G.degree()]
        if isinstance(G, nx.DiGraph):
            wl = weisfeiler_lehman_graph_hash(G.to_undirected(), node_attr="cls", iterations=3, digest_size=16)
        else:
            wl = weisfeiler_lehman_graph_hash(G, node_attr=None, iterations=3, digest_size=16)
        hist = np.bincount(np.array(deg, dtype=int), minlength=16)[:16]
        return {"n": G.number_of_nodes(), "m": G.number_of_edges(), "deg_hist": hist, "wl": wl, "aut_est": self._estimate_aut(G, 150)}

    def _passes(self, ref: Dict[str, object], out: Dict[str, object]) -> bool:
        strict = float(max(0.0, min(2.0, self.geometric_strength)))
        r = ref["deg_hist"].astype(float); o = out["deg_hist"].astype(float)
        r /= (r.sum() + 1e-12); o /= (o.sum() + 1e-12)
        l1 = float(np.abs(r - o).sum())
        max_l1 = 1.10 - 0.35 * strict
        if l1 > max(0.25, max_l1):
            return False

        a_ref = max(1, int(ref["aut_est"]))
        a_out = max(1, int(out["aut_est"]))
        ratio = a_out / a_ref
        band = 3.5 - 1.2 * min(2.0, float(self.steer_strength) / 2.0)
        band = max(1.3, band)
        if not (1.0 / band <= ratio <= band):
            return False

        if strict >= 1.6 and out["wl"] != ref["wl"]:
            return False
        return True

    # ---- chunking
    def _chunk_text(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 120]
        if len(paras) >= 6:
            return paras[:500]
        t2 = re.sub(r"\s+", " ", text).strip()
        if len(t2) < 400:
            return [t2]
        blocks: List[str] = []
        step = 320
        for i in range(0, min(len(t2), 320 * 600), step):
            blk = t2[i : i + step]
            if len(blk) >= 150:
                blocks.append(blk)
        return blocks or [t2[:800]]

    # ---- RFE
    def _rfe(self, W: np.ndarray, energies: np.ndarray, vocab: List[str], progress: Optional[gr.Progress] = None):
        if not self.rfe_enabled or self.rfe_iterations <= 0:
            return W, vocab
        Wc = W.copy()
        vc = list(vocab)
        for it in range(self.rfe_iterations):
            if progress:
                progress(0.8 + 0.05 * (it / self.rfe_iterations), desc=f"RFE {it+1}/{self.rfe_iterations}")
            imp = np.zeros(Wc.shape[1])
            for i in range(Wc.shape[0]):
                imp += energies[i] * np.abs(Wc[i, :])
            imp = 0.7 * imp + 0.3 * np.var(Wc, axis=0)
            imp = imp / (imp.max() + 1e-12)

            n = Wc.shape[1]
            rem = max(1, int(n * self.rfe_removal_rate))
            keep = n - rem
            if keep < max(10, self.bars_n // 3):
                break
            top = np.argsort(-imp)[:keep]
            top = np.sort(top)
            Wc = Wc[:, top]
            vc = [vc[i] for i in top]
        return Wc, vc

    # ---- semantic graph
    def _semantic_graph(self, nodelets: List[Nodelet], vocab: List[str], W: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        for n in nodelets:
            G.add_node(f"N{n.idx}", bipartite=0, kind="nodelet", energy=float(n.energy))
        for j, term in enumerate(vocab):
            G.add_node(f"B{j}", bipartite=1, kind="bar", term=term)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = float(W[i, j])
                if w > 0:
                    G.add_edge(f"N{i}", f"B{j}", weight=w)
        return G

    # ---- boosts
    def _make_boost(self, vocab: List[str], probs: np.ndarray, nodelets: List[Nodelet], pillar_bias: np.ndarray) -> Dict[str, float]:
        boost: Dict[str, float] = {}
        for term, p, pw in zip(vocab, probs, pillar_bias):
            for w in term.split():
                if len(w) <= 2 or w in STOPWORDS:
                    continue
                val = float(math.log(p + 1e-12) + 6.0) + 0.35 * float(np.log(float(pw) + 1e-12) + 1.0)
                boost[w] = max(boost.get(w, 0.0), val)

        E = np.array([n.energy for n in nodelets], dtype=float)
        E = E / (E.max() + 1e-12)
        for i, n in enumerate(nodelets):
            top = []
            for t, _ in n.top_terms[:12]:
                for tok in t.split():
                    if len(tok) > 2 and tok not in STOPWORDS:
                        top.append(tok)
            for w in top:
                boost[w] = boost.get(w, 0.0) + 0.35 * float(E[i])

        if boost:
            vals = np.array(list(boost.values()), dtype=float)
            lo, hi = float(np.percentile(vals, 10)), float(np.percentile(vals, 95))
            if hi - lo < 1e-9:
                hi = lo + 1.0
            for k in list(boost.keys()):
                boost[k] = 1.5 * clip01((boost[k] - lo) / (hi - lo))
        return boost

    def _seed_words(self, text_seed: str) -> List[str]:
        if not text_seed:
            return []
        toks = [t for t in basic_tokenize(text_seed) if re.match(r"[a-z]", t)]
        out, seen = [], set()
        for w in toks:
            if len(w) <= 2 or w in STOPWORDS:
                continue
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out[:50]

    # -----------------------------
    # THE compressed "ulterior probs" pipeline
    # -----------------------------
    def _prob_pipeline(
        self,
        base_logits: np.ndarray,
        base_probs: np.ndarray,
        *,
        pillar_strength: float,
        geometric_strength: float,
        concave_strength: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]:
        """
        Returns:
          logits_final, probs_final, (pillar_bias, geometric_bias), trace
        trace is the "piecemeal unfold" list of (name, vector).
        """
        trace: List[Tuple[str, np.ndarray]] = []
        trace.append(("base_logits", base_logits.copy()))
        trace.append(("base_probs", base_probs.copy()))

        # Pillars: additive bias based on normalized base_probs
        if pillar_strength > 0:
            pnorm = base_probs / (base_probs.max() + 1e-12)
            pillar_bias = pillar_strength * pnorm
        else:
            pillar_bias = np.zeros_like(base_probs)
        trace.append(("pillar_bias", pillar_bias.copy()))

        # Geometric modulation
        if len(base_probs) > 1 and geometric_strength > 0:
            pos = np.arange(len(base_probs), dtype=float)
            dist = np.abs(pos - pos[0])
            md = float(dist.max())
            nd = dist / md if md > 1e-12 else dist
            pv = base_probs - base_probs[0]
            angles = np.arctan2(pv, nd + 1e-12)
            an = (angles - angles.min()) / (angles.max() - angles.min() + 1e-12)
            geometric_bias = geometric_strength * (nd + an)
        else:
            geometric_bias = np.zeros_like(base_probs)
        trace.append(("geometric_bias", geometric_bias.copy()))

        # Combine
        logits = base_logits + pillar_bias + geometric_bias
        trace.append(("combined_logits", logits.copy()))

        probs = softmax(logits, temp=self.softmax_temp)
        trace.append(("softmax_probs", probs.copy()))

        probs = concave_focus(probs, strength=concave_strength)
        trace.append(("concave_probs", probs.copy()))

        return logits, probs, pillar_bias, geometric_bias, trace

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, text: str, progress: Optional[gr.Progress] = None) -> ModelState:
        if progress:
            progress(0.0, desc="Normalizing text")
        text = normalize(text)
        chunks = self._chunk_text(text)

        if progress:
            progress(0.2, desc="TF-IDF")
        vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=8000, ngram_range=(1, 2))
        X = vec.fit_transform(chunks)
        vocab = np.array(vec.get_feature_names_out())
        mass = np.asarray(X.sum(axis=0)).ravel()
        top_idx = np.argsort(-mass)[: self.bars_n]
        vocab100 = vocab[top_idx].tolist()

        if progress:
            progress(0.4, desc="SVD")
        k = min(self.nodelets_n, max(2, X.shape[0] - 1), max(2, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.svd_random_state)
        svd.fit(X)

        nodelets: List[Nodelet] = []
        for i, comp in enumerate(svd.components_):
            ti = np.argsort(-np.abs(comp))[:20]
            terms = [(vocab[j], float(comp[j])) for j in ti]
            energy = float(np.linalg.norm(comp))
            tops = [t for t, _ in terms[:8]]
            tops_clean = []
            for t in tops:
                if len(tops_clean) >= 6:
                    break
                if t not in tops_clean:
                    tops_clean.append(t)
            strength = "high" if energy > 2.2 else "moderate" if energy > 1.4 else "light"
            narrative = f"Nodelet {i} carries a {strength}-intensity semantic current, orbiting around {', '.join(tops_clean)}."
            nodelets.append(Nodelet(i, terms, energy, narrative))

        if progress:
            progress(0.8, desc="Bindings")
        W = np.zeros((k, self.bars_n), dtype=float)
        for i, n in enumerate(nodelets):
            weights = {t: abs(w) for t, w in n.top_terms}
            for j, term in enumerate(vocab100):
                W[i, j] = weights.get(term, 0.0)
            mx = float(W[i].max())
            if mx > 1e-12:
                W[i] /= mx

        energies = np.array([n.energy for n in nodelets], dtype=float)
        energies = energies / (energies.max() + 1e-12)

        W, vocab100 = self._rfe(W, energies, vocab100, progress=progress)

        rng = np.random.default_rng(self.svd_random_state)
        base_logits = (energies[:, None] * W).sum(axis=0) + 0.02 * rng.normal(size=W.shape[1])
        base_probs = softmax(base_logits, temp=self.softmax_temp)

        logits, probs, pillar_bias, geometric_bias, trace = self._prob_pipeline(
            base_logits,
            base_probs,
            pillar_strength=self.pillar_strength,
            geometric_strength=self.geometric_strength,
            concave_strength=self.concave_focus_strength,
        )

        token_boost = self._make_boost(vocab100, probs, nodelets, pillar_bias)
        semG = self._semantic_graph(nodelets, vocab100, W)

        return ModelState(
            nodelets=nodelets,
            vocab100=vocab100,
            binding_W=W,
            bar_probs=probs,
            bar_logits=logits,
            token_boost=token_boost,
            pillar_weights=pillar_bias,
            geometric_bias=geometric_bias,
            semantic_graph=semG,
            lm_graph=nx.DiGraph(),
            prob_trace=trace if self.unfold_probs else None,
        )

    # -----------------------------
    # Generation
    # -----------------------------
    def _choose_start(self, lm: QuadgramLM, boost: Dict[str, float], rng: np.random.Generator, seed_words: List[str]) -> str:
        usable = [w for w in seed_words if w in lm.uni and w not in STOPWORDS and len(w) > 2]
        if usable:
            usable.sort(key=lambda w: (boost.get(w, 0.0), lm.uni.get(w, 0)), reverse=True)
            return str(rng.choice(usable[:10]))
        boosted = [(w, b) for w, b in boost.items() if b > 0.9 and w in lm.uni]
        if boosted:
            return max(boosted, key=lambda x: x[1])[0]
        return max(lm.uni.items(), key=lambda x: x[1])[0]

    def _sample_next(
        self,
        lm: QuadgramLM,
        w1: str,
        w2: str,
        w3: str,
        boost: Dict[str, float],
        rng: np.random.Generator,
        temperature: float = 0.95,
    ) -> str:
        cand, base_p = lm.next_distribution(w1, w2, w3)
        base_p = concave_focus(base_p, strength=self.concave_focus_strength)

        scores = np.log(base_p + 1e-12)
        for i, w in enumerate(cand):
            if w in PUNC:
                continue
            scores[i] += self.steer_strength * boost.get(w, 0.0)

        scores = scores / max(temperature, 1e-9)
        scores = scores - np.max(scores)
        p = np.exp(scores)
        p = p / (p.sum() + 1e-12)

        p = concave_focus(p, strength=self.concave_focus_strength * 0.7)
        return str(rng.choice(cand, p=p))

    def _generate_sentence(
        self,
        lm: QuadgramLM,
        boost: Dict[str, float],
        rng: np.random.Generator,
        seed_words: List[str],
        min_len: int = 800,
        max_len: int = 900,
    ) -> str:
        seed = self._choose_start(lm, boost, rng, seed_words)
        tokens = ["("] if rng.random() < 0.12 else []
        tokens.append(seed)
        w1 = w2 = w3 = tokens[-1]

        target_len = int(rng.integers(min_len, max_len + 1))
        for _ in range(target_len):
            nxt = self._sample_next(lm, w1, w2, w3, boost, rng)
            tokens.append(nxt)
            w1, w2, w3 = w2, w3, nxt
            if nxt in {".", "!", "?"} and len([t for t in tokens if re.match(r"[a-z]", t)]) >= min_len:
                break

        if tokens and tokens[-1] not in {".", "!", "?"}:
            tokens.append(".")

        if tokens and tokens[0] == "(" and ")" not in tokens:
            if rng.random() < 0.6:
                tokens.insert(min(len(tokens), 6), ")")
            else:
                tokens = tokens[1:]

        return detokenize(tokens)

    def _generate_takeaways(
        self,
        text: str,
        state: ModelState,
        n_takeaways: int,
        rng: np.random.Generator,
        text_seed: str = "",
        progress: Optional[gr.Progress] = None,
    ) -> List[str]:
        if progress:
            progress(0.0, desc="Training LM (4-gram)")

        tokens = basic_tokenize(text)
        if len(tokens) < 600:
            tokens = tokens * 2

        lm = QuadgramLM(add_k=self.lm_add_k)
        lm.fit(tokens)
        state.lm_graph = lm.build_transition_graph(max_edges=20000)

        ref_tokens = basic_tokenize(text)
        refG = self._token_structure_graph(ref_tokens, 220)
        ref_sig = self._graph_sig(refG)

        seed_words = self._seed_words(text_seed)
        takeaways: List[str] = []

        energies = np.array([n.energy for n in state.nodelets], dtype=float)
        energies = energies / (energies.max() + 1e-12)
        energies = concave_focus(energies, strength=self.concave_focus_strength)

        max_attempts = int(6 + 6 * max(0.0, min(2.0, self.geometric_strength)))

        for i in range(int(n_takeaways)):
            if progress:
                progress(i / max(1, int(n_takeaways)), desc=f"Takeaway {i+1}/{int(n_takeaways)}")

            lead = int(rng.choice(len(state.nodelets), p=softmax(energies, temp=0.9)))
            row = state.binding_W[lead]

            local = dict(state.token_boost)
            for w in seed_words:
                local[w] = max(local.get(w, 0.0), 1.25)

            topbars = np.argsort(-row)[:10]
            for b in topbars:
                term = state.vocab100[b]
                strength = float(row[b])
                for w in term.split():
                    if len(w) > 2 and w not in STOPWORDS:
                        local[w] = max(local.get(w, 0.0), 1.0 + 0.6 * strength)

            for k in list(local.keys()):
                local[k] = float(min(2.0, max(0.0, local[k])))

            last = ""
            for _ in range(max_attempts):
                sent = self._generate_sentence(lm, local, rng, seed_words=seed_words)
                last = sent
                out_tokens = basic_tokenize(sent)
                outG = self._token_structure_graph(out_tokens, 220)
                out_sig = self._graph_sig(outG)
                if self._passes(ref_sig, out_sig):
                    takeaways.append(sent)
                    break
            else:
                takeaways.append(last)

        if progress:
            progress(1.0, desc="Done")
        return takeaways

    def generate_report(
        self,
        text: str,
        n_takeaways: int = 7,
        seed: int = 7,
        text_seed: str = "",
        progress: Optional[gr.Progress] = None,
    ) -> str:
        text = normalize(text)
        state = self.fit(text, progress=progress)
        rng = np.random.default_rng(int(seed))
        takeaways = self._generate_takeaways(text, state, int(n_takeaways), rng, text_seed=text_seed, progress=progress)
        return "\n\n".join(takeaways)


# -----------------------------
# Gradio actions
# -----------------------------
def generate_from_file(
    in_file: str,
    softmax_temp: float,
    steer_strength: float,
    geometric_strength: float,
    concave_focus_strength: float,
    rfe_enabled: bool,
    rfe_iterations: int,
    rfe_removal_rate: float,
    n_takeaways: int,
    seed: int,
    text_seed: str,
    output_name: str,
    unfold_probs: bool,
    progress: gr.Progress = gr.Progress(),
):
    if not in_file:
        raise gr.Error("Please upload an input file.")
    progress(0.0, desc="Loading file")
    raw = load_text(in_file)

    gen = NeuroSymbolicGraphGenerator(
        nodelets_n=10,
        bars_n=100,
        svd_random_state=7,
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        lm_add_k=0.25,
        pillar_strength=0.85,
        pillar_floor=0.25,
        geometric_strength=float(geometric_strength),
        rfe_enabled=bool(rfe_enabled),
        rfe_iterations=int(rfe_iterations),
        rfe_removal_rate=float(rfe_removal_rate),
        concave_focus_strength=float(concave_focus_strength),
        unfold_probs=bool(unfold_probs),
    )

    report = gen.generate_report(
        raw,
        n_takeaways=int(n_takeaways),
        seed=int(seed),
        text_seed=(text_seed or "").strip(),
        progress=progress,
    )

    # Optional: attach unfold trace at the bottom (when enabled)
    # (Kept minimal; if you want a prettier tab, tell me and I’ll add a UI tab.)
    if gen.unfold_probs:
        state = gen.fit(raw)  # light recompute without progress for trace (kept simple)
        if state.prob_trace:
            report += "\n\n" + "=" * 60 + "\nPROBABILITY PIPELINE TRACE (top-10)\n" + "=" * 60 + "\n"
            for name, vec in state.prob_trace:
                if vec.ndim == 1 and len(vec) > 0:
                    top = np.argsort(-vec)[:10]
                    vals = ", ".join([f"{int(i)}:{float(vec[i]):.4f}" for i in top])
                    report += f"{name}: {vals}\n"

    progress(0.0, desc="Saving output file")
    tmpdir = Path(tempfile.mkdtemp(prefix="neurosym_textgen_"))
    stem = Path(in_file).stem
    out_name = (output_name or "").strip()
    if not out_name:
        out_path = tmpdir / f"{stem}_generated_report.txt"
    else:
        out_name = out_name if out_name.lower().endswith(".txt") else out_name + ".txt"
        out_path = tmpdir / out_name
    out_path.write_text(report, encoding="utf-8")
    return report, str(out_path)


def generate_from_hf_dataset(
    dataset_name: str,
    config_name: str,
    splits: str,
    text_columns: str,
    context_columns: str,
    max_samples: int,
    mix_mode: str,
    manual_mix: str,
    softmax_temp: float,
    steer_strength: float,
    geometric_strength: float,
    concave_focus_strength: float,
    rfe_enabled: bool,
    rfe_iterations: int,
    rfe_removal_rate: float,
    n_takeaways: int,
    seed: int,
    text_seed: str,
    output_name: str,
    unfold_probs: bool,
    progress: gr.Progress = gr.Progress(),
):
    if not dataset_name or not dataset_name.strip():
        raise gr.Error("Please enter a dataset name.")

    raw = load_hf_dataset(
        dataset_name=dataset_name.strip(),
        config_name=config_name.strip() if config_name and config_name.strip() else None,
        splits=splits.strip() if splits and splits.strip() else "train",
        text_columns=text_columns.strip() if text_columns and text_columns.strip() else None,
        context_columns=context_columns.strip() if context_columns and context_columns.strip() else None,
        max_samples=int(max_samples) if max_samples and max_samples > 0 else None,
        mix_mode=(mix_mode or "off"),
        manual_mix=(manual_mix or ""),
        prompt_text=(text_seed or ""),
        seed=int(seed),
        progress=progress,
    )

    gen = NeuroSymbolicGraphGenerator(
        nodelets_n=10,
        bars_n=100,
        svd_random_state=7,
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        lm_add_k=0.25,
        pillar_strength=0.85,
        pillar_floor=0.25,
        geometric_strength=float(geometric_strength),
        rfe_enabled=bool(rfe_enabled),
        rfe_iterations=int(rfe_iterations),
        rfe_removal_rate=float(rfe_removal_rate),
        concave_focus_strength=float(concave_focus_strength),
        unfold_probs=bool(unfold_probs),
    )

    report = gen.generate_report(
        raw,
        n_takeaways=int(n_takeaways),
        seed=int(seed),
        text_seed=(text_seed or "").strip(),
        progress=progress,
    )

    if gen.unfold_probs:
        state = gen.fit(raw)
        if state.prob_trace:
            report += "\n\n" + "=" * 60 + "\nPROBABILITY PIPELINE TRACE (top-10)\n" + "=" * 60 + "\n"
            for name, vec in state.prob_trace:
                if vec.ndim == 1 and len(vec) > 0:
                    top = np.argsort(-vec)[:10]
                    vals = ", ".join([f"{int(i)}:{float(vec[i]):.4f}" for i in top])
                    report += f"{name}: {vals}\n"

    progress(0.0, desc="Saving output file")
    tmpdir = Path(tempfile.mkdtemp(prefix="neurosym_textgen_"))
    out_name = (output_name or "").strip()
    if not out_name:
        out_path = tmpdir / f"{dataset_name.replace('/', '_')}_generated_report.txt"
    else:
        out_name = out_name if out_name.lower().endswith(".txt") else out_name + ".txt"
        out_path = tmpdir / out_name
    out_path.write_text(report, encoding="utf-8")
    return report, str(out_path)


# -----------------------------
# Gradio UI
# -----------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Neurosymbolic Text Generator") as demo:
        gr.Markdown("# Neurosymbolic Text Generator (Graph Automorphism Gated)")
        gr.Markdown(
            "*TF-IDF/SVD nodelets + RFE + quad-gram LM + geometric modulation + concave focus + automorphism checks + HF split mixing + context fields*"
        )

        with gr.Tabs():
            with gr.Tab("Upload File"):
                with gr.Row():
                    in_file = gr.File(
                        label="Input file (.txt/.md/.docx/.pdf)",
                        file_types=[".txt", ".md", ".docx", ".pdf"],
                        type="filepath",
                    )
                    with gr.Column():
                        output_name_file = gr.Textbox(label="Output filename (optional)", placeholder="e.g., report.txt")
                        text_seed_file = gr.Textbox(label="Text seed (optional)", placeholder="e.g., 'quantum measurement'")
                        unfold_probs_file = gr.Checkbox(value=False, label="Unfold probability pipeline (trace)")

                with gr.Row():
                    softmax_temp_file = gr.Slider(0.2, 2.5, value=0.85, step=0.05, label="Softmax temp")
                    steer_strength_file = gr.Slider(0.0, 5.0, value=1.35, step=0.05, label="Steer strength")
                    geometric_strength_file = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Geometric strength")
                    concave_focus_strength_file = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Concave focus strength")

                with gr.Row():
                    rfe_enabled_file = gr.Checkbox(value=True, label="Enable RFE")
                    rfe_iterations_file = gr.Slider(1, 10, value=3, step=1, label="RFE iterations")
                    rfe_removal_rate_file = gr.Slider(0.05, 0.5, value=0.15, step=0.05, label="RFE removal rate")

                with gr.Row():
                    n_takeaways_file = gr.Slider(1, 30, value=7, step=1, label="# takeaways")
                    seed_file = gr.Number(value=7, precision=0, label="Numeric seed")

                run_btn_file = gr.Button("Generate from file", variant="primary", size="lg")

                with gr.Row():
                    preview_file = gr.Textbox(label="Generated report preview", lines=20)
                    download_file = gr.File(label="Download generated .txt")

            with gr.Tab("Hugging Face Dataset"):
                gr.Markdown(
                    """
### Load data from Hugging Face datasets
Examples:
- Dataset: `imdb` | splits: `train,test` | text_columns: `text`
- Dataset: `squad` | splits: `train,validation` | text_columns: `question,answers` | context_columns: `context,title`
- Dataset: `cnn_dailymail` | config: `3.0.0` | splits: `train,validation` | text_columns: `article,highlights`
"""
                )

                with gr.Row():
                    with gr.Column():
                        dataset_name = gr.Textbox(label="Dataset name", placeholder="e.g., imdb, ag_news, cnn_dailymail, squad", value="")
                        config_name = gr.Textbox(label="Config/Subset (if required)", placeholder="e.g., 3.0.0 for cnn_dailymail", value="")
                    with gr.Column():
                        splits = gr.Textbox(label="Splits (comma-separated)", placeholder="train,validation,test", value="train")
                        max_samples = gr.Number(label="Max samples (0 = all)", value=100, precision=0)

                with gr.Row():
                    text_columns = gr.Textbox(label="Main text columns (comma-separated, empty = auto)", placeholder="e.g., text OR article,highlights", value="", scale=2)
                    context_columns = gr.Textbox(label="Context/support columns (comma-separated, optional)", placeholder="e.g., context,title", value="", scale=2)

                with gr.Row():
                    mix_mode = gr.Dropdown(
                        choices=["off", "auto", "manual"],
                        value="off",
                        label="Split mixing mode",
                        info="off: use first split only • auto: infer weights from text seed • manual: provide weights",
                    )
                    manual_mix = gr.Textbox(label="Manual split weights (manual mode)", placeholder="e.g., train=0.7,validation=0.3,test=0.0", value="")

                with gr.Row():
                    output_name_hf = gr.Textbox(label="Output filename (optional)", placeholder="e.g., report.txt")
                    unfold_probs_hf = gr.Checkbox(value=False, label="Unfold probability pipeline (trace)")

                with gr.Row():
                    text_seed_hf = gr.Textbox(label="Text seed (optional; drives auto mixing)", placeholder="Words/phrase to bias generation", scale=2)

                with gr.Row():
                    softmax_temp_hf = gr.Slider(0.2, 2.5, value=0.85, step=0.05, label="Softmax temp")
                    steer_strength_hf = gr.Slider(0.0, 5.0, value=1.35, step=0.05, label="Steer strength")
                    geometric_strength_hf = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Geometric strength")
                    concave_focus_strength_hf = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Concave focus strength")

                with gr.Row():
                    rfe_enabled_hf = gr.Checkbox(value=True, label="Enable RFE")
                    rfe_iterations_hf = gr.Slider(1, 10, value=3, step=1, label="RFE iterations")
                    rfe_removal_rate_hf = gr.Slider(0.05, 0.5, value=0.15, step=0.05, label="RFE removal rate")

                with gr.Row():
                    n_takeaways_hf = gr.Slider(1, 30, value=7, step=1, label="# takeaways")
                    seed_hf = gr.Number(value=7, precision=0, label="Numeric seed")

                run_btn_hf = gr.Button("Generate from dataset", variant="primary", size="lg")

                with gr.Row():
                    preview_hf = gr.Textbox(label="Generated report preview", lines=20)
                    download_hf = gr.File(label="Download generated .txt")

        run_btn_file.click(
            fn=generate_from_file,
            inputs=[
                in_file,
                softmax_temp_file,
                steer_strength_file,
                geometric_strength_file,
                concave_focus_strength_file,
                rfe_enabled_file,
                rfe_iterations_file,
                rfe_removal_rate_file,
                n_takeaways_file,
                seed_file,
                text_seed_file,
                output_name_file,
                unfold_probs_file,
            ],
            outputs=[preview_file, download_file],
        )

        run_btn_hf.click(
            fn=generate_from_hf_dataset,
            inputs=[
                dataset_name,
                config_name,
                splits,
                text_columns,
                context_columns,
                max_samples,
                mix_mode,
                manual_mix,
                softmax_temp_hf,
                steer_strength_hf,
                geometric_strength_hf,
                concave_focus_strength_hf,
                rfe_enabled_hf,
                rfe_iterations_hf,
                rfe_removal_rate_hf,
                n_takeaways_hf,
                seed_hf,
                text_seed_hf,
                output_name_hf,
                unfold_probs_hf,
            ],
            outputs=[preview_hf, download_hf],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(share=False, max_file_size="1MB")

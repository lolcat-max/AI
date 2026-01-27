#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurosymbolic Text Generator (Gradio GUI) - Enhanced with Contextual Grid
Gaspare sparsification + Menger-curvature + Contextual Grid Associationality

Deps:
  pip install gradio numpy torch
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import OrderedDict, defaultdict

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ----------------------------
# Text utils
# ----------------------------

STOPWORDS = set(
    """
    a an and are as at be by for from has have he her hers him his i in is it its me my
    of on or our ours she so that the their them they this to was we were what when where
    which who will with you your yours
    """.split()
)

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
        out.append(t.lower() if re.match(r"[A-Za-z]", t) else t)
    return out

def detokenize(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in [".", ",", ";", ":", "!", "?", ")", "("]:
            if t in ["(", ")"]:
                out.append(t)
            else:
                if out:
                    out[-1] += t
                else:
                    out.append(t)
        else:
            if out and out[-1].endswith("("):
                out[-1] += t
            else:
                out.append(t)
    s = " ".join(out)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s

def _file_to_path(infile: Any) -> str:
    if infile is None:
        raise ValueError("No file provided.")
    if isinstance(infile, str):
        return infile
    if hasattr(infile, "name") and isinstance(infile.name, str):
        return infile.name
    if isinstance(infile, dict) and "path" in infile:
        return str(infile["path"])
    raise ValueError(f"Unsupported file input type: {type(infile)}")

def load_text(infile: Any) -> str:
    path = _file_to_path(infile)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext in [".txt", ".md"]:
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file extension: {ext} (txt/md only).")


# ----------------------------
# Pure NumPy TF-IDF + randomized SVD
# ----------------------------

def pure_tfidf(docs: List[str], max_features: int = 8000) -> Tuple[np.ndarray, List[str]]:
    # Build doc term counts
    doc_tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    tf_counts: List[Dict[str, int]] = []
    for doc in docs:
        toks = re.findall(r"\b\w+\b", doc.lower())
        doc_tokens.append(toks)
        counts: Dict[str, int] = {}
        seen = set()
        for w in toks:
            counts[w] = counts.get(w, 0) + 1
            if w not in seen:
                df[w] = df.get(w, 0) + 1
                seen.add(w)
        tf_counts.append(counts)

    # vocab by DF (simple + deterministic)
    vocab = sorted(df.keys(), key=lambda w: (-df[w], w))[:max_features]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    X = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    N = max(1, len(docs))

    for i, counts in enumerate(tf_counts):
        denom = max(1, len(counts))
        for w, c in counts.items():
            j = word_to_idx.get(w, None)
            if j is None:
                continue
            tf = c / denom
            idf = math.log(N / (1 + df.get(w, 0)))
            X[i, j] = tf * idf

    return X, vocab

def pure_truncated_svd(X: np.ndarray, n_components: int, random_state: int = 42, n_iter: int = 8):
    np.random.seed(int(random_state))
    m, n = X.shape
    k = max(1, min(int(n_components), m, n))

    Q = np.random.randn(n, k)
    Q, _ = np.linalg.qr(Q)

    XtX = X.T @ X
    for _ in range(int(n_iter)):
        Q, _ = np.linalg.qr(XtX @ Q)

    B = X @ Q
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return type("SVD", (), {"components_": Vt[:k]})()


# ----------------------------
# Contextual Grid Structure
# ----------------------------

@dataclass
class GridCell:
    """Represents a cell in the contextual grid with associational links."""
    word: str
    position: Tuple[int, int]  # (row, col) in grid
    frequency: float
    neighbors: Set[str] = field(default_factory=set)
    semantic_layer: int = 0  # depth in association hierarchy
    activation: float = 0.0

class ContextualGrid:
    """
    Multi-dimensional grid structure for candidate organization.
    Words are placed in a grid based on:
    - Frequency (vertical axis)
    - Semantic similarity (horizontal axis)
    - Contextual co-occurrence (depth/layer)
    """
    
    def __init__(self, grid_size: int = 20, semantic_layers: int = 3):
        self.grid_size = int(grid_size)
        self.semantic_layers = int(semantic_layers)
        self.cells: Dict[Tuple[int, int, int], GridCell] = {}  # (row, col, layer) -> cell
        self.word_to_pos: Dict[str, Tuple[int, int, int]] = {}
        self.association_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def add_word(self, word: str, freq: float, semantic_vec: Optional[np.ndarray] = None) -> None:
        """Add word to grid based on frequency and semantic properties."""
        # Frequency determines vertical position
        row = int(min(self.grid_size - 1, freq * self.grid_size))
        
        # Semantic vector determines horizontal position
        if semantic_vec is not None and len(semantic_vec) > 0:
            col = int(abs(hash(tuple(semantic_vec[:3]))) % self.grid_size)
        else:
            col = int(abs(hash(word)) % self.grid_size)
            
        # Find available layer
        layer = 0
        while (row, col, layer) in self.cells and layer < self.semantic_layers:
            layer += 1
            
        pos = (row, col, layer)
        cell = GridCell(
            word=word,
            position=pos,
            frequency=freq,
            semantic_layer=layer
        )
        
        self.cells[pos] = cell
        self.word_to_pos[word] = pos
        
    def add_association(self, word1: str, word2: str, strength: float) -> None:
        """Add associational link between words."""
        self.association_graph[word1][word2] = strength
        self.association_graph[word2][word1] = strength
        
        # Update neighbor sets
        if word1 in self.word_to_pos and word2 in self.word_to_pos:
            pos1 = self.word_to_pos[word1]
            pos2 = self.word_to_pos[word2]
            if pos1 in self.cells:
                self.cells[pos1].neighbors.add(word2)
            if pos2 in self.cells:
                self.cells[pos2].neighbors.add(word1)
    
    def get_neighborhood(self, word: str, radius: int = 2) -> List[str]:
        """Get words in spatial neighborhood of given word."""
        if word not in self.word_to_pos:
            return []
            
        row, col, layer = self.word_to_pos[word]
        neighbors = []
        
        for r in range(max(0, row - radius), min(self.grid_size, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.grid_size, col + radius + 1)):
                for l in range(self.semantic_layers):
                    if (r, c, l) in self.cells:
                        cell = self.cells[(r, c, l)]
                        if cell.word != word:
                            neighbors.append(cell.word)
        
        return neighbors
    
    def activate_region(self, center_words: List[str], decay: float = 0.7) -> Dict[str, float]:
        """
        Spread activation from center words through the grid.
        Returns activation levels for all words.
        """
        activations: Dict[str, float] = defaultdict(float)
        
        # Initialize center words
        for w in center_words:
            activations[w] = 1.0
            
        # Spread activation through association graph
        for _ in range(3):  # Multiple passes
            new_activations = activations.copy()
            
            for word, activation in activations.items():
                if activation > 0.01:  # Only spread from active nodes
                    for neighbor, strength in self.association_graph.get(word, {}).items():
                        new_activations[neighbor] = max(
                            new_activations[neighbor],
                            activation * decay * strength
                        )
            
            activations = new_activations
            
        return dict(activations)
    
    def get_candidates_by_activation(
        self, 
        context: List[str], 
        base_candidates: List[str],
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Select candidates using grid-based activation spreading.
        Returns list of (word, score) tuples.
        """
        # Activate regions around context words
        activations = self.activate_region(context)
        
        # Score candidates
        scored = []
        for cand in base_candidates:
            score = activations.get(cand, 0.0)
            
            # Boost by neighborhood density
            neighbors = self.get_neighborhood(cand, radius=1)
            neighborhood_score = sum(1 for n in neighbors if n in base_candidates) / max(1, len(neighbors))
            
            # Boost by association strength with context
            context_score = 0.0
            for ctx_word in context:
                context_score += self.association_graph.get(cand, {}).get(ctx_word, 0.0)
            context_score /= max(1, len(context))
            
            final_score = score + 0.3 * neighborhood_score + 0.4 * context_score
            scored.append((cand, final_score))
        
        # Sort and return top-k
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ----------------------------
# Menger-curvature reweighting for discrete probs
# ----------------------------

def menger_probs_transform(
    p: np.ndarray,
    n_triples: int = 500,
    seed: int = 42,
    strength: float = 1.5,
) -> np.ndarray:
    """
    Discrete "curvature" weighting on index triples.
    Uses 1/R (circumradius) from triangle side lengths; collinear => 0.
    """
    rng = np.random.default_rng(int(seed))
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + 1e-12)

    N = p.size
    if N < 3:
        return p

    idx = np.arange(N, dtype=np.float64)
    curv_sum = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)

    T = int(max(1, n_triples))
    for _ in range(T):
        i, j, k = rng.choice(N, 3, replace=False)
        x, y, z = idx[i], idx[j], idx[k]
        a = abs(y - x)
        b = abs(z - y)
        c = abs(z - x)

        if min(a, b, c) < 1e-12:
            curv = 0.0
        else:
            s = (a + b + c) / 2.0
            area2 = s * (s - a) * (s - b) * (s - c)
            if area2 <= 1e-18:
                curv = 0.0
            else:
                area = math.sqrt(area2)
                R = (a * b * c) / (4.0 * area + 1e-12)
                curv = 1.0 / (R + 1e-12)

        curv_sum[i] += curv; counts[i] += 1.0
        curv_sum[j] += curv; counts[j] += 1.0
        curv_sum[k] += curv; counts[k] += 1.0

    avg = curv_sum / np.maximum(counts, 1.0)
    avg = np.maximum(avg, 1e-6)
    avg = avg / (avg.max() + 1e-12)  # [0,1]

    p2 = p * (avg ** float(strength))
    return p2 / (p2.sum() + 1e-12)


# ----------------------------
# Tiny runtime cache
# ----------------------------

class RadixLRUCache:
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


# ----------------------------
# PyTorch neuro-ish modules
# ----------------------------

class LateralInhibition(nn.Module):
    def __init__(self, strength=0.5):
        super().__init__()
        self.strength = float(strength)
        k = torch.tensor([-0.95, -0.9, -0.1, 0.3, -1.4, -1.2, -1.05], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        modulation = F.conv1d(x, self.kernel, padding=self.pad)
        out = x + self.strength * modulation
        out = F.relu(out)
        return out / (out.sum(dim=-1, keepdim=True) + 1e-12)

class ResonantGate(nn.Module):
    def __init__(self, steer_strength=1.35):
        super().__init__()
        self.steer_strength = float(steer_strength)
        self.noise_injector = nn.Dropout(p=0.05)

    def forward(self, lm_probs: torch.Tensor, token_boosts: torch.Tensor, temp=0.7) -> torch.Tensor:
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
# Enhanced Quadgram LM with Grid
# ----------------------------

class QuadgramLM:
    def __init__(self, add_k: float = 0.25, use_grid: bool = True, grid_size: int = 20):
        self.add_k = float(add_k)
        self.use_grid = bool(use_grid)
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total = 0
        
        # Contextual grid for associational candidate selection
        self.grid: Optional[ContextualGrid] = ContextualGrid(grid_size=grid_size) if use_grid else None

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
        
        # Build contextual grid
        if self.grid is not None:
            self._build_grid(tokens)

    def _build_grid(self, tokens: List[str]) -> None:
        """Build contextual grid from token sequence."""
        if self.grid is None:
            return
            
        # Add words to grid based on frequency
        max_freq = max(self.uni.values()) if self.uni else 1
        for word, count in self.uni.items():
            if word not in STOPWORDS and len(word) > 2:
                norm_freq = count / max_freq
                self.grid.add_word(word, norm_freq)
        
        # Build association graph from co-occurrence
        window_size = 5
        for i in range(len(tokens)):
            center = tokens[i]
            if center in STOPWORDS or len(center) <= 2:
                continue
                
            # Look at surrounding context
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    neighbor = tokens[j]
                    if neighbor not in STOPWORDS and len(neighbor) > 2:
                        # Co-occurrence strength based on distance
                        distance = abs(i - j)
                        strength = 1.0 / (1.0 + distance)
                        self.grid.add_association(center, neighbor, strength)

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], torch.Tensor]:
        cont: List[str] = []

        # Gather candidates from n-gram contexts
        for (a, b, c, d), _count in self.quad.items():
            if a == w1 and b == w2 and c == w3:
                cont.append(d)

        if not cont:
            for (a, b, c), _count in self.tri.items():
                if a == w2 and b == w3:
                    cont.append(c)

        if not cont:
            for (a, b), _count in self.bi.items():
                if a == w3:
                    cont.append(b)

        if not cont:
            cont = [w for w, _ in sorted(self.uni.items(), key=lambda x: x[1], reverse=True)[:200]]

        # Deduplicate
        seen = set()
        cand = []
        for w in cont:
            if w not in seen:
                seen.add(w)
                cand.append(w)
        cand = cand[:500]
        
        # Use grid for associational reranking
        if self.grid is not None and len(cand) > 0:
            context_words = [w for w in [w1, w2, w3] if w in self.grid.word_to_pos]
            if context_words:
                grid_scored = self.grid.get_candidates_by_activation(
                    context_words, 
                    cand, 
                    top_k=min(300, len(cand))
                )
                
                # Blend grid scores with original order
                grid_dict = {word: score for word, score in grid_scored}
                cand_reranked = []
                for w in cand:
                    if w in grid_dict:
                        cand_reranked.append(w)
                
                # Add back any missing candidates
                for w in cand:
                    if w not in cand_reranked:
                        cand_reranked.append(w)
                
                cand = cand_reranked[:500]

        V = len(self.vocab) + 1
        add_k = self.add_k

        def get_prob(w4: str) -> float:
            c123 = self.tri.get((w1, w2, w3), 0)
            c1234 = self.quad.get((w1, w2, w3, w4), 0)
            if c123 > 0:
                return (c1234 + add_k) / (c123 + add_k * V)

            c23 = self.bi.get((w2, w3), 0)
            c234 = self.tri.get((w2, w3, w4), 0)
            if c23 > 0:
                return (c234 + add_k) / (c23 + add_k * V)

            c3 = self.uni.get(w3, 0)
            c34 = self.bi.get((w3, w4), 0)
            if c3 > 0:
                return (c34 + add_k) / (c3 + add_k * V)

            return (self.uni.get(w4, 0) + add_k) / (self.total + add_k * V)

        probs = torch.tensor([get_prob(w) for w in cand], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# ----------------------------
# Model state + generator
# ----------------------------

@dataclass
class Nodelet:
    idx: int
    top_terms: List[Tuple[str, float]]
    energy: float

@dataclass
class ModelState:
    vocab_top: List[str]
    bar_probs: torch.Tensor
    token_boost: Dict[str, float]

@dataclass
class PreparedCorpus:
    text: str
    tokens: List[str]
    lm: QuadgramLM
    state: ModelState

class NeuroSymbolicGraphGenerator:
    def __init__(
        self,
        nodelets_n: int = 10,
        bars_n: int = 100,
        svd_random_state: int = 7,
        softmax_temp: float = 0.85,
        steer_strength: float = 1.35,
        lm_add_k: float = 0.25,
        focus_strength: float = 0.5,
        gelu_seed: int = 1337,
        gelu_hidden: int = 32,
        radix_cache_items: int = 25000,
        speculative_accept_topk: int = 10,
        use_contextual_grid: bool = True,
        grid_size: int = 20,
    ):
        self.nodelets_n = int(nodelets_n)
        self.bars_n = int(bars_n)
        self.svd_random_state = int(svd_random_state)
        self.softmax_temp = float(softmax_temp)
        self.lm_add_k = float(lm_add_k)
        self.use_contextual_grid = bool(use_contextual_grid)
        self.grid_size = int(grid_size)

        self.focus_layer = LateralInhibition(strength=float(focus_strength))
        self.gate_layer = ResonantGate(steer_strength=float(steer_strength))
        self.synthetic_bias = SyntheticGELUBias(hidden=gelu_hidden, approximate="tanh")
        self.synthetic_bias.reset_seed(int(gelu_seed))
        self.synthetic_bias.freeze_(True)

        self.cache_version = 0
        self.radix_cache = RadixLRUCache(max_items=int(radix_cache_items))
        self.speculative_accept_topk = int(speculative_accept_topk)

    def bump_cache_version(self):
        self.cache_version += 1
        self.radix_cache.clear()

    def _pick_initial_context(self, lm: QuadgramLM, seed_words: List[str]) -> Tuple[str, str, str]:
        sw = [t for t in seed_words if re.match(r"^[a-z][a-z0-9_\-']*$", t)]
        if len(sw) >= 3:
            return (sw[-3], sw[-2], sw[-1])
        if len(sw) == 2:
            return (sw[-2], sw[-1], sw[-1])
        if len(sw) == 1:
            return (sw[-1], sw[-1], sw[-1])
        seed_tok = lm.vocab[0] if lm.vocab else "the"
        return (seed_tok, seed_tok, seed_tok)

    def build_state(self, text: str, progress=None) -> ModelState:
        if progress:
            progress(0.0, desc="Normalizing")
        text = normalize(text)

        docs = re.split(r"\n\s*\n", text)
        docs = [d.strip() for d in docs if d.strip()]
        docs = docs[:500] if docs else [text]

        X, vocab = pure_tfidf(docs, max_features=8000)
        if X.size == 0 or not vocab:
            return ModelState(vocab_top=[], bar_probs=torch.ones(1), token_boost={})

        col_sums = X.sum(axis=0)
        n_keep = min(self.bars_n, col_sums.shape[0])
        top_idx = np.argsort(-col_sums)[:n_keep]
        vocab_top = [vocab[i] for i in top_idx]
        X_svd = X[:, top_idx]

        n_rows, n_cols = X_svd.shape
        max_rank = min(n_rows, n_cols)
        k = max(1, min(self.nodelets_n, max_rank, 10))

        svd = pure_truncated_svd(X_svd, n_components=k, random_state=self.svd_random_state)

        # Nodelet energies from component norms
        nodelets: List[Nodelet] = []
        for i, comp in enumerate(svd.components_):
            terms = sorted(
                [(vocab_top[j], float(comp[j])) for j in range(len(comp))],
                key=lambda x: -abs(x[1]),
            )[:10]
            eng = float(np.linalg.norm(comp))
            nodelets.append(Nodelet(i, terms, eng))

        energies = np.array([n.energy for n in nodelets], dtype=np.float64)
        if energies.size == 0:
            energies = np.ones(1, dtype=np.float64)
        energies = energies / (energies.max() + 1e-12)

        W = torch.tensor(svd.components_, dtype=torch.float32)
        W = F.relu(W)
        W = W / (W.max(dim=1, keepdim=True)[0] + 1e-12)

        e_t = torch.tensor(energies, dtype=torch.float32).view(-1, 1)
        logits = (e_t * W).sum(dim=0)

        probs = F.softmax(logits / self.softmax_temp, dim=-1)
        probs = self.focus_layer(probs.view(1, 1, -1)).squeeze(0).squeeze(0)

        token_boost: Dict[str, float] = {}
        for w, p in zip(vocab_top, probs.detach().cpu().tolist()):
            for subw in w.split():
                if len(subw) > 2 and subw not in STOPWORDS:
                    token_boost[subw] = max(token_boost.get(subw, 0.0), math.log(p + 1e-12) + 5.0)

        return ModelState(vocab_top=vocab_top, bar_probs=probs, token_boost=token_boost)

    def prepare_corpus(self, text: str, progress=None) -> PreparedCorpus:
        text = normalize(text)
        state = self.build_state(text, progress)
        tokens = basic_tokenize(text)
        lm = QuadgramLM(
            self.lm_add_k, 
            use_grid=self.use_contextual_grid,
            grid_size=self.grid_size
        )
        lm.ingest(tokens)
        return PreparedCorpus(text=text, tokens=tokens, lm=lm, state=state)

    def _final_probs_for_context_cached(
        self,
        prep: PreparedCorpus,
        w1: str,
        w2: str,
        w3: str,
    ) -> Tuple[List[str], torch.Tensor]:
        key = (int(self.cache_version), str(w1), str(w2), str(w3))
        cached = self.radix_cache.get(key)
        if cached is not None:
            return cached

        cand, base_probs = prep.lm.next_distribution(w1, w2, w3)
        if not cand:
            cand = prep.lm.vocab[:100] if prep.lm.vocab else ["the", "is", "a"]
            base_p = torch.ones(len(cand), dtype=torch.float32) / max(1, len(cand))
        else:
            base_p = base_probs.detach().clone().to(dtype=torch.float32)

        base_p = base_p / (base_p.sum() + 1e-12)
        base_p = self.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

        boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand], dtype=torch.float32)
        bias = self.synthetic_bias(base_p, boosts).view(-1)

        final_probs = self.gate_layer(base_p, boosts + bias, temp=0.9).view(-1)
        self.radix_cache.put(key, (cand, final_probs.detach().clone()))
        return cand, final_probs


# ----------------------------
# Continuous batching decoder
# ----------------------------

@dataclass
class DecodeStream:
    stream_id: int
    tokens_out: List[str]
    w1: str
    w2: str
    w3: str
    done: bool = False
    alpha_count: int = 0
    max_steps: int = 240
    stop_tokens: set = field(default_factory=lambda: {".", "!", "?"})
    min_alpha: int = 120

class ContinuousBatchDecoder:
    def __init__(
        self,
        gen: NeuroSymbolicGraphGenerator,
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

    def _propose_token_base(self, w1: str, w2: str, w3: str) -> str:
        cand, base_probs = self.prep.lm.next_distribution(w1, w2, w3)
        if not cand:
            return w3
        p = base_probs.detach().cpu().numpy()
        p = p / (p.sum() + 1e-12)
        return self.rng.choice(cand, p=p)

    def _sample_from_probs(self, cand: List[str], probs: torch.Tensor) -> str:
        p = probs.detach().cpu().numpy()

        # Gaspare: dynamic sparse cutoff
        cutoff = np.mean(p) + (self.rng.random() * np.std(p))
        p_sparse = np.where(p > cutoff, p, 0.0)
        if p_sparse.sum() < 1e-12:
            p_sparse = p
        p_sparse = p_sparse / (p_sparse.sum() + 1e-12)

        # Menger: curvature weighting
        seed_menger = int(self.rng.integers(0, 2**32))
        p_menger = menger_probs_transform(p_sparse, n_triples=500, seed=seed_menger, strength=1.5)

        return self.rng.choice(cand, p=p_menger)

    def step_round(self, streams: List[DecodeStream]) -> None:
        active = [s for s in streams if not s.done]
        if not active:
            return

        active.sort(key=lambda s: (s.w1, s.w2, s.w3))
        active = active[: min(len(active), self.token_budget_per_round)]

        groups: Dict[Tuple[str, str, str], List[DecodeStream]] = {}
        for s in active:
            groups.setdefault((s.w1, s.w2, s.w3), []).append(s)

        for (w1, w2, w3), bucket in groups.items():
            cand, final_probs = self.gen._final_probs_for_context_cached(self.prep, w1, w2, w3)

            topk_set = None
            if self.speculative and len(cand) > 0:
                topk = min(self.gen.speculative_accept_topk, len(cand))
                _, idx = torch.topk(final_probs, k=topk)
                topk_set = set(idx.detach().cpu().tolist())

            for s in bucket:
                if not cand:
                    nxt = s.w3
                else:
                    if self.speculative and topk_set is not None:
                        proposed = self._propose_token_base(s.w1, s.w2, s.w3)
                        try:
                            j = cand.index(proposed)
                        except ValueError:
                            j = -1
                        nxt = proposed if (j >= 0 and j in topk_set) else self._sample_from_probs(cand, final_probs)
                    else:
                        nxt = self._sample_from_probs(cand, final_probs)

                s.tokens_out.append(nxt)
                if nxt.isalpha():
                    s.alpha_count += 1
                s.w1, s.w2, s.w3 = s.w2, s.w3, nxt

                if s.alpha_count >= s.max_steps:
                    s.done = True
                elif nxt in s.stop_tokens and s.alpha_count > s.min_alpha:
                    s.done = True


# ----------------------------
# SGLang-ish wrappers
# ----------------------------

class SGPrompt:
    def __init__(self, text: str = ""):
        self.text = str(text)

    def __iadd__(self, other: str):
        self.text += str(other)
        return self

    def __str__(self):
        return self.text

class SGContext:
    def __init__(self, corpus_text: str, generator: NeuroSymbolicGraphGenerator, seed: int = 7, prepared: Optional[PreparedCorpus] = None):
        self.corpus_text = normalize(corpus_text)
        self.generator = generator
        self.seed = int(seed)
        self.prepared: Optional[PreparedCorpus] = prepared

    def ensure_prepared(self):
        if self.prepared is None:
            self.prepared = self.generator.prepare_corpus(self.corpus_text)

    def clone(self, seed_offset: int) -> "SGContext":
        return SGContext(self.corpus_text, self.generator, seed=self.seed + int(seed_offset), prepared=self.prepared)

def sg_gen_batched(
    ctxs: List[SGContext],
    prompts: List[SGPrompt],
    max_tokens: int = 240,
    seed_offsets: Optional[List[int]] = None,
    stop_at_punc: bool = True,
) -> List[str]:
    if not ctxs:
        return []
    gen = ctxs[0].generator
    ctxs[0].ensure_prepared()
    prep = ctxs[0].prepared
    assert prep is not None

    rng = np.random.default_rng(ctxs[0].seed)
    streams: List[DecodeStream] = []

    for i, (ctx, prompt) in enumerate(zip(ctxs, prompts)):
        off = seed_offsets[i] if seed_offsets else i
        seed_words = basic_tokenize(prompt.text)
        w1, w2, w3 = gen._pick_initial_context(prep.lm, seed_words)
        streams.append(
            DecodeStream(
                stream_id=i,
                tokens_out=[w1, w2, w3],
                w1=w1, w2=w2, w3=w3,
                max_steps=int(max_tokens),
                min_alpha=int(max_tokens // 2) if stop_at_punc else 999999,
            )
        )

    decoder = ContinuousBatchDecoder(gen, prep, rng, token_budget_per_round=64, speculative=True)

    for _ in range(int(max_tokens) * 2):
        if all(s.done for s in streams):
            break
        decoder.step_round(streams)

    results = []
    for s in streams:
        out_toks = s.tokens_out[3:] if len(s.tokens_out) > 3 else []
        results.append(detokenize(out_toks))
    return results

def sg_gen(ctx: SGContext, prompt: SGPrompt, max_tokens=240, seed_offset=0) -> str:
    res = sg_gen_batched([ctx], [prompt], max_tokens=max_tokens, seed_offsets=[seed_offset])
    return res[0]

def sg_fork(ctx: SGContext, prompt: SGPrompt, n: int) -> List[Tuple[SGContext, SGPrompt]]:
    n = int(max(1, n))
    ctx.ensure_prepared()
    out = []
    for i in range(n):
        out.append((ctx.clone(seed_offset=1000 + i), SGPrompt(prompt.text)))
    return out

def sg_join(prompts: List[SGPrompt], joiner: str = "\n\n") -> SGPrompt:
    return SGPrompt(joiner.join(p.text for p in prompts))


# ----------------------------
# Program + training
# ----------------------------

def run_program(
    infile: Any,
    n_take: int,
    seed: int,
    steer: float,
    focus: float,
    gelu_seed: int,
    use_grid: bool,
    grid_size: int,
    takeaway_prefix: str,
    summary_prompt_tmpl: str,
    trained_state: Optional[dict] = None,
) -> str:
    corpus_text = load_text(infile)

    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        gelu_seed=int(gelu_seed),
        radix_cache_items=30000,
        speculative_accept_topk=10,
        use_contextual_grid=bool(use_grid),
        grid_size=int(grid_size),
    )

    if isinstance(trained_state, dict) and "gelu_state_dict" in trained_state:
        try:
            gen.synthetic_bias.load_state_dict(trained_state["gelu_state_dict"], strict=True)
        except Exception:
            pass
        gen.synthetic_bias.freeze_(True)
        gen.synthetic_bias.eval()
        gen.bump_cache_version()

    ctx = SGContext(corpus_text, gen, seed=int(seed))
    ctx.ensure_prepared()

    root = SGPrompt(str(takeaway_prefix).strip() + "\n\n")
    branches = sg_fork(ctx, root, n=int(n_take))
    branch_ctxs = [b[0] for b in branches]
    branch_prompts = [b[1] for b in branches]

    for i, bp in enumerate(branch_prompts):
        bp += f"[Takeaway {i+1}] "

    take_texts = sg_gen_batched(branch_ctxs, branch_prompts, max_tokens=220, stop_at_punc=True)
    for i, txt in enumerate(take_texts):
        branch_prompts[i] += txt

    merged = sg_join(branch_prompts, joiner="\n\n")
    final_sum_prompt = summary_prompt_tmpl.replace("{joined_takeaways}", merged.text)
    summary_prompt = SGPrompt(final_sum_prompt)
    summary_text = sg_gen(ctx, summary_prompt, max_tokens=260)
    return summary_prompt.text + summary_text

def train_bias_net(
    infile: Any,
    seed: int,
    steer: float,
    focus: float,
    gelu_seed: int,
    use_grid: bool,
    grid_size: int,
    train_steps: int,
    lr: float,
    max_contexts: int,
    progress=gr.Progress(),
):
    text = load_text(infile)
    gen = NeuroSymbolicGraphGenerator(
        steer_strength=float(steer),
        focus_strength=float(focus),
        gelu_seed=int(gelu_seed),
        use_contextual_grid=bool(use_grid),
        grid_size=int(grid_size),
    )

    progress(0.0, desc="Preparing")
    prep = gen.prepare_corpus(text)
    tokens = prep.tokens
    if len(tokens) < 12:
        return None, "Not enough tokens to train."

    gen.synthetic_bias.reset_seed(int(gelu_seed))
    gen.synthetic_bias.freeze_(False)
    gen.synthetic_bias.train()

    opt = optim.Adam(gen.synthetic_bias.parameters(), lr=float(lr))
    positions = list(range(3, len(tokens)))
    if max_contexts and int(max_contexts) > 0:
        positions = positions[: min(len(positions), int(max_contexts))]

    rng = np.random.default_rng(int(seed))
    batch_size = 24
    steps = int(train_steps)
    running_loss = 0.0

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        used = 0
        loss_acc = 0.0

        batch_pos = rng.choice(positions, size=min(batch_size, len(positions)), replace=False)
        for i in batch_pos:
            w1, w2, w3 = tokens[i - 3], tokens[i - 2], tokens[i - 1]
            true_next = tokens[i]

            cand, base_probs = prep.lm.next_distribution(w1, w2, w3)
            if not cand:
                continue

            base_p = base_probs.detach().clone().to(dtype=torch.float32)
            base_p = base_p / (base_p.sum() + 1e-12)
            base_p = gen.focus_layer(base_p.view(1, 1, -1)).squeeze(0).squeeze(0)

            boosts = torch.tensor([prep.state.token_boost.get(w, 0.0) for w in cand], dtype=torch.float32)
            bias = gen.synthetic_bias(base_p, boosts).view(-1)
            probs = gen.gate_layer(base_p, boosts + bias, temp=0.9)

            try:
                j = cand.index(true_next)
            except ValueError:
                continue

            loss_acc = loss_acc - torch.log(probs[j].clamp_min(1e-12))
            used += 1

        if used > 0:
            loss = loss_acc / used
            loss.backward()
            opt.step()
            running_loss += float(loss.item())

        if (step + 1) % max(1, steps // 10) == 0:
            progress((step + 1) / steps, desc=f"Training {step+1}/{steps}")

    state = {"gelu_state_dict": {k: v.detach().cpu() for k, v in gen.synthetic_bias.state_dict().items()}}
    return state, f"Trained. Avg loss={running_loss/max(1,steps):.4f}"


# ----------------------------
# Gradio app
# ----------------------------

def build_app():
    with gr.Blocks(title="Neurosymbolic Enhanced (Grid + Associations)") as demo:
        gr.Markdown(
            "# Neurosymbolic Enhanced\n"
            "*Gaspare + Menger + Contextual Grid with Associational Networks*\n\n"
            "**New Feature**: Contextual grid structures organize candidates spatially by frequency "
            "and semantic similarity, with associational links that spread activation through the vocabulary space."
        )

        trained_state = gr.State(None)

        with gr.Row():
            infile = gr.File(label="Input File (txt/md only)")
            out_txt = gr.Textbox(label="Structured Output", lines=20)

        with gr.Row():
            n_take = gr.Slider(1, 10, value=4, step=1, label="Parallel Forks (Batch Size)")
            seed = gr.Number(value=42, label="Seed", precision=0)

        with gr.Row():
            steer = gr.Slider(0, 5, value=1.35, label="Steer")
            focus = gr.Slider(0, 1, value=0.5, label="Focus")
            gelu_seed = gr.Number(value=1337, label="GELU Seed", precision=0)

        with gr.Accordion("Contextual Grid Settings", open=True):
            use_grid = gr.Checkbox(value=True, label="Enable Contextual Grid")
            grid_size = gr.Slider(10, 40, value=20, step=1, label="Grid Size (spatial resolution)")
            gr.Markdown(
                "*Grid organizes vocabulary by frequency and semantic clustering, "
                "enabling associational spreading activation for candidate selection.*"
            )

        with gr.Accordion("Editable Prompts", open=False):
            p_takeaway = gr.Textbox(label="Takeaway Prompt (Prefix)", value="", lines=2)
            p_summary = gr.Textbox(
                label="Prompt Template, {joined_takeaways} will be replaced",
                value="explain the nature of this?\n\n{joined_takeaways}\n\nplan:",
                lines=4,
            )

        train_btn = gr.Button("Train GELU Bias (Optional)")
        run_btn = gr.Button("Run Structured Program", variant="primary")
        status = gr.Textbox(label="Train Status")

        train_btn.click(
            train_bias_net,
            inputs=[
                infile,
                seed,
                steer,
                focus,
                gelu_seed,
                use_grid,
                grid_size,
                gr.Number(100, visible=False),
                gr.Number(0.001, visible=False),
                gr.Number(0, visible=False),
            ],
            outputs=[trained_state, status],
        )

        run_btn.click(
            run_program,
            inputs=[
                infile,
                n_take,
                seed,
                steer,
                focus,
                gelu_seed,
                use_grid,
                grid_size,
                p_takeaway,
                p_summary,
                trained_state,
            ],
            outputs=out_txt,
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(debug=True, show_error=True)

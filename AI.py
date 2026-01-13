#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurosymbolic Text Generator (Gradio GUI) with Hugging Face Dataset Support
+ Vertical Pillars (ADDITIVE logit-level bias)
+ Geometric Distance & Angle Modulation
+ Quad-gram Language Model (4-gram)
+ Cosine similarity edit->retrieve context->regenerate workflow
+ Regen uses edited text as prompt (prefix continuation) AND mixes edit into LM corpus
"""

from __future__ import annotations

import re
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


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


def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    text_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: Optional[gr.Progress] = None,
) -> str:
    """Load text from a Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    if progress:
        progress(0, desc=f"Loading dataset: {dataset_name}")

    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if text_column is None:
        text_candidates = ["text", "article", "content", "document", "context", "sentence"]
        for candidate in text_candidates:
            if candidate in ds.column_names:
                text_column = candidate
                break
        if text_column is None:
            text_column = ds.column_names[0]

    if text_column not in ds.column_names:
        raise ValueError(f"Column '{text_column}' not found. Available: {ds.column_names}")

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    if progress:
        progress(0.5, desc="Extracting text from dataset")

    texts = []
    for item in ds:
        text = item[text_column]
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    combined = "\n\n".join(texts)

    if progress:
        progress(1.0, desc="Dataset loaded successfully")

    return combined


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


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x.astype(float) / max(temp, 1e-12)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ----------------------------
# Tokenization + detokenization
# ----------------------------
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


# ----------------------------
# Neurosymbolic model pieces
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
    binding_W: np.ndarray
    bar_probs: np.ndarray
    bar_logits: np.ndarray
    token_boost: Dict[str, float]
    pillar_weights: np.ndarray
    geometric_bias: np.ndarray

    # Retrieval index (kept per session in a global dict; includes non-deepcopyable objects)
    chunks: List[str]
    tfidf_vec: TfidfVectorizer
    chunk_tfidf: Any  # sparse matrix


class NeuroSymbolicTextGenerator:
    """
    Interpretable neurosymbolic generator:
    - Nodelets: latent semantic factors (TF-IDF->SVD)
    - Bars: top N vocab items
    - W: bindings nodelets->bars
    - Vertical pillars: ADDITIVE logit-level bias
    - Geometric modulation: distance & angle from first word
    - Takeaways: quad-gram LM steered by token boosts
    """

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
    ):
        self.nodelets_n = nodelets_n
        self.bars_n = bars_n
        self.svd_random_state = svd_random_state
        self.softmax_temp = softmax_temp
        self.steer_strength = steer_strength
        self.lm_add_k = lm_add_k
        self.pillar_strength = float(pillar_strength)
        self.pillar_floor = float(pillar_floor)
        self.geometric_strength = float(geometric_strength)

    def fit(self, text: str, progress: Optional[gr.Progress] = None) -> ModelState:
        if progress:
            progress(0, desc="Normalizing text")
        text = normalize(text)
        chunks = self._chunk_text(text)

        if progress:
            progress(0.2, desc="Computing TF-IDF vectors")
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=8000,
            ngram_range=(1, 2),
        )
        X = vec.fit_transform(chunks)
        vocab = np.array(vec.get_feature_names_out())
        global_mass = np.asarray(X.sum(axis=0)).ravel()
        top_idx = np.argsort(-global_mass)[: self.bars_n]
        vocab100 = vocab[top_idx].tolist()

        if progress:
            progress(0.4, desc="Performing SVD decomposition")
        k = min(self.nodelets_n, max(2, X.shape[0] - 1), max(2, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.svd_random_state)
        svd.fit(X)

        nodelets = []
        for i, comp in enumerate(svd.components_):
            term_idx = np.argsort(-np.abs(comp))[:20]
            terms = [(vocab[j], float(comp[j])) for j in term_idx]
            energy = float(np.linalg.norm(comp))
            narrative = self._nodelet_narrative(i, terms, energy)
            nodelets.append(Nodelet(i, terms, energy, narrative))

        if progress:
            progress(0.8, desc="Building bindings matrix")
        W = np.zeros((k, self.bars_n))
        for i, n in enumerate(nodelets):
            weights = {t: abs(w) for t, w in n.top_terms}
            for j, term in enumerate(vocab100):
                W[i, j] = weights.get(term, 0.0)
            mx = W[i].max()
            if mx > 1e-12:
                W[i] /= mx

        energies = np.array([n.energy for n in nodelets])
        energies /= energies.max() + 1e-12

        rng = np.random.default_rng(self.svd_random_state)
        base_logits = (energies[:, None] * W).sum(axis=0)
        base_logits += 0.02 * rng.normal(size=base_logits.shape)
        base_probs = softmax(base_logits, temp=self.softmax_temp)

        # ---- ADDITIVE PILLARS (external field)
        if self.pillar_strength > 0.0:
            pnorm = base_probs / (base_probs.max() + 1e-12)
            pillar_bias = self.pillar_strength * pnorm
        else:
            pillar_bias = np.zeros_like(base_probs)

        # ---- GEOMETRIC DISTANCE & ANGLE MODULATION
        if len(base_probs) > 1 and self.geometric_strength > 0.0:
            positions = np.arange(len(base_probs)).astype(float)
            distances = np.abs(positions - positions[0])
            max_dist = distances.max()
            if max_dist > 1e-12:
                norm_distances = distances / max_dist
            else:
                norm_distances = distances

            prob_vectors = base_probs - base_probs[0]
            angles = np.arctan2(prob_vectors, norm_distances + 1e-12)
            angle_norm = (angles - angles.min()) / (angles.max() - angles.min() + 1e-12)

            geometric_bias = self.geometric_strength * (norm_distances + angle_norm)
        else:
            geometric_bias = np.zeros_like(base_probs)

        logits = base_logits + pillar_bias + geometric_bias
        probs = softmax(logits, temp=self.softmax_temp)

        token_boost = self._make_token_boost(vocab100, probs, nodelets, pillar_bias)

        return ModelState(
            nodelets=nodelets,
            vocab100=vocab100,
            binding_W=W,
            bar_probs=probs,
            bar_logits=logits,
            token_boost=token_boost,
            pillar_weights=pillar_bias,
            geometric_bias=geometric_bias,
            chunks=chunks,
            tfidf_vec=vec,
            chunk_tfidf=X,
        )

    def generate_report_from_state(
        self,
        lm_text: str,
        state: ModelState,
        n_takeaways: int = 7,
        seed: int = 7,
        text_seed: str = "",
        prompt_text: str = "",
        progress: Optional[gr.Progress] = None,
    ) -> str:
        lm_text = normalize(lm_text)
        rng = np.random.default_rng(seed)
        takeaways = self._generate_takeaways(
            lm_text,
            state,
            n_takeaways=n_takeaways,
            rng=rng,
            text_seed=text_seed,
            prompt_text=prompt_text,
            progress=progress,
        )
        return "\n\n".join(takeaways)

    def _chunk_text(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 120]
        if len(paras) >= 6:
            return paras[:500]
        text2 = re.sub(r"\s+", " ", text).strip()
        if len(text2) < 400:
            return [text2]
        blocks = []
        step = 320
        for i in range(0, min(len(text2), 320 * 600), step):
            blk = text2[i:i + step]
            if len(blk) >= 150:
                blocks.append(blk)
        return blocks or [text2[:800]]

    def _make_token_boost(
        self,
        vocab100: List[str],
        bar_probs: np.ndarray,
        nodelets: List[Nodelet],
        pillar_weights: np.ndarray,
    ) -> Dict[str, float]:
        boost: Dict[str, float] = {}
        for idx, (term, p) in enumerate(zip(vocab100, bar_probs)):
            pw = float(pillar_weights[idx]) if idx < len(pillar_weights) else 1.0
            for w in term.split():
                if len(w) <= 2 or w in STOPWORDS:
                    continue
                val = float(math.log(p + 1e-12) + 6.0) + 0.35 * float(np.log(pw + 1e-12) + 1.0)
                boost[w] = max(boost.get(w, 0.0), val)

        energies = np.array([n.energy for n in nodelets], dtype=float)
        energies = energies / (energies.max() + 1e-12)
        for i, n in enumerate(nodelets):
            top_words = []
            for t, _w in n.top_terms[:12]:
                for tok in t.split():
                    if len(tok) > 2 and tok not in STOPWORDS:
                        top_words.append(tok)
            node_strength = energies[i]
            for w in top_words:
                boost[w] = boost.get(w, 0.0) + 0.35 * node_strength

        if boost:
            vals = np.array(list(boost.values()), dtype=float)
            lo, hi = float(np.percentile(vals, 10)), float(np.percentile(vals, 95))
            if hi - lo < 1e-9:
                hi = lo + 1.0
            for k in list(boost.keys()):
                boost[k] = 1.5 * clip01((boost[k] - lo) / (hi - lo))
        return boost

    def _nodelet_narrative(self, i: int, terms: List[Tuple[str, float]], energy: float) -> str:
        tops = [t for t, _ in terms[:8]]
        tops_clean = []
        for t in tops:
            if len(tops_clean) >= 6:
                break
            if t not in tops_clean:
                tops_clean.append(t)
        strength = "high" if energy > 2.2 else "moderate" if energy > 1.4 else "light"
        return (
            f"Nodelet {i} carries a {strength}-intensity semantic current, orbiting around "
            f"{', '.join(tops_clean)}."
        )

    def _extract_seed_words(self, text_seed: str) -> List[str]:
        if not text_seed:
            return []
        toks = basic_tokenize(text_seed)
        words = [t for t in toks if re.match(r"[a-z]", t)]
        out = []
        seen = set()
        for w in words:
            if len(w) <= 2 or w in STOPWORDS:
                continue
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out[:50]

    def _sample_next(
        self,
        lm: 'QuadgramLM',
        w1: str,
        w2: str,
        w3: str,
        token_boost: Dict[str, float],
        rng: np.random.Generator,
        temperature: float = 0.95,
    ) -> str:
        cand, base_p = lm.next_distribution(w1, w2, w3)
        scores = np.log(base_p + 1e-12)
        for i, w in enumerate(cand):
            if w in [".", ",", ";", ":", "!", "?", "(", ")"]:
                continue
            scores[i] = scores[i] + self.steer_strength * token_boost.get(w, 0.0)
        scores = scores / max(temperature, 1e-9)
        scores = scores - np.max(scores)
        p = np.exp(scores)
        p = p / (p.sum() + 1e-12)
        return str(rng.choice(cand, p=p))

    def _choose_start_word(
        self,
        lm: 'QuadgramLM',
        token_boost: Dict[str, float],
        rng: np.random.Generator,
        seed_words: List[str],
    ) -> str:
        usable = [w for w in seed_words if w in lm.uni and w not in STOPWORDS and len(w) > 2]
        if usable:
            usable.sort(key=lambda w: (token_boost.get(w, 0.0), lm.uni.get(w, 0)), reverse=True)
            return str(rng.choice(usable[:10]))
        boosted = [(w, b) for w, b in token_boost.items() if b > 0.9 and w in lm.uni]
        if boosted:
            return max(boosted, key=lambda x: x[1])[0]
        return max(lm.uni.items(), key=lambda x: x[1])[0]

    def _generate_sentence(
        self,
        lm: 'QuadgramLM',
        token_boost: Dict[str, float],
        rng: np.random.Generator,
        seed_words: Optional[List[str]] = None,
        prompt_text: str = "",
        min_len: int = 800,
        max_len: int = 900,
    ) -> str:
        seed_words = seed_words or []

        prompt_tokens = basic_tokenize(prompt_text) if (prompt_text and prompt_text.strip()) else []
        if prompt_tokens and prompt_tokens[-1] in [".", "!", "?"]:
            prompt_tokens = prompt_tokens[:-1]

        tokens: List[str] = []

        if not prompt_tokens and rng.random() < 0.12:
            tokens.append("(")

        if prompt_tokens:
            tokens.extend(prompt_tokens)
        else:
            seed = self._choose_start_word(lm, token_boost, rng, seed_words)
            tokens.append(seed)

        if len(tokens) >= 3:
            w1, w2, w3 = tokens[-3], tokens[-2], tokens[-1]
        else:
            last = tokens[-1]
            w1 = w2 = w3 = last

        target_len = int(rng.integers(min_len, max_len + 1))
        remaining = max(0, target_len - len(tokens))

        for _ in range(remaining):
            nxt = self._sample_next(lm, w1, w2, w3, token_boost, rng)
            tokens.append(nxt)
            w1, w2, w3 = w2, w3, nxt
            if nxt in [".", "!", "?"] and len([t for t in tokens if re.match(r"[a-z]", t)]) >= min_len:
                break

        if tokens and tokens[-1] not in [".", "!", "?"]:
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
        prompt_text: str = "",
        progress: Optional[gr.Progress] = None,
    ) -> List[str]:
        if progress:
            progress(0, desc="Training language model")
        tokens = basic_tokenize(text)
        if len(tokens) < 600:
            tokens = tokens * 2
        lm = QuadgramLM(add_k=self.lm_add_k)
        lm.fit(tokens)

        seed_words = self._extract_seed_words(text_seed)
        takeaways = []
        energies = np.array([n.energy for n in state.nodelets], dtype=float)
        energies = energies / (energies.max() + 1e-12)

        for i in range(n_takeaways):
            if progress:
                progress(i / n_takeaways, desc=f"Generating takeaway {i+1}/{n_takeaways}")
            lead = int(rng.choice(len(state.nodelets), p=softmax(energies, temp=0.9)))
            row = state.binding_W[lead]

            local_boost = dict(state.token_boost)
            for w in seed_words:
                local_boost[w] = max(local_boost.get(w, 0.0), 1.25)

            topbars = np.argsort(-row)[:10]
            for b in topbars:
                term = state.vocab100[b]
                strength = float(row[b])
                for w in term.split():
                    if len(w) > 2 and w not in STOPWORDS:
                        local_boost[w] = max(local_boost.get(w, 0.0), 1.0 + 0.6 * strength)

            for k in list(local_boost.keys()):
                local_boost[k] = float(min(2.0, max(0.0, local_boost[k])))

            sent = self._generate_sentence(
                lm,
                local_boost,
                rng,
                seed_words=seed_words,
                prompt_text=(prompt_text if i == 0 else ""),
            )
            takeaways.append(sent)

        if progress:
            progress(1.0, desc="Generation complete")
        return takeaways


class QuadgramLM:
    """4-gram language model with add-k smoothing"""
    def __init__(self, add_k: float = 0.25):
        self.add_k = add_k
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.quad: Dict[Tuple[str, str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total: int = 0

    def fit(self, tokens: List[str]) -> None:
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
        self.quad.clear()
        self.total = 0

        def inc(d, key, val=1):
            d[key] = d.get(key, 0) + val

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

    def _prob_unigram(self, w: str) -> float:
        V = len(self.vocab) + 1
        return (self.uni.get(w, 0) + self.add_k) / (self.total + self.add_k * V)

    def _prob_bigram(self, w1: str, w2: str) -> float:
        V = len(self.vocab) + 1
        c1 = self.uni.get(w1, 0)
        c12 = self.bi.get((w1, w2), 0)
        return (c12 + self.add_k) / (c1 + self.add_k * V) if c1 > 0 else self._prob_unigram(w2)

    def _prob_trigram(self, w1: str, w2: str, w3: str) -> float:
        V = len(self.vocab) + 1
        c12 = self.bi.get((w1, w2), 0)
        c123 = self.tri.get((w1, w2, w3), 0)
        return (c123 + self.add_k) / (c12 + self.add_k * V) if c12 > 0 else self._prob_bigram(w2, w3)

    def _prob_quadgram(self, w1: str, w2: str, w3: str, w4: str) -> float:
        V = len(self.vocab) + 1
        c123 = self.tri.get((w1, w2, w3), 0)
        c1234 = self.quad.get((w1, w2, w3, w4), 0)
        return (c1234 + self.add_k) / (c123 + self.add_k * V) if c123 > 0 else self._prob_trigram(w2, w3, w4)

    def next_distribution(self, w1: str, w2: str, w3: str) -> Tuple[List[str], np.ndarray]:
        cont: List[str] = []

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

        seen = set()
        cand = []
        for w in cont:
            if w not in seen:
                seen.add(w)
                cand.append(w)
        cand = cand[:500]

        probs = np.array([self._prob_quadgram(w1, w2, w3, w) for w in cand], dtype=float)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# ----------------------------
# Cosine-similarity retrieval
# ----------------------------
def retrieve_topk_chunks(
    state: ModelState,
    query_text: str,
    k: int = 12,
) -> Tuple[str, List[List[Any]]]:
    q = normalize(query_text or "")
    if not q.strip():
        k2 = min(k, len(state.chunks))
        ctx = "\n\n".join(state.chunks[:k2])
        rows = [
            [i + 1, None, state.chunks[i][:220].replace("\n", " ") + ("..." if len(state.chunks[i]) > 220 else "")]
            for i in range(k2)
        ]
        return ctx, rows

    qv = state.tfidf_vec.transform([q])
    sims = cosine_similarity(qv, state.chunk_tfidf).ravel()
    idx = np.argsort(-sims)[: min(k, len(sims))]

    ctx = "\n\n".join(state.chunks[i] for i in idx)
    rows = []
    for rank, i in enumerate(idx, start=1):
        snippet = state.chunks[i][:220].replace("\n", " ")
        if len(state.chunks[i]) > 220:
            snippet += "..."
        rows.append([rank, float(sims[i]), snippet])
    return ctx, rows


def save_report_to_tmp(report: str, default_stem: str, output_name: str) -> str:
    tmpdir = Path(tempfile.mkdtemp(prefix="neurosym_textgen_"))
    out_name = (output_name or "").strip()
    if not out_name:
        out_path = tmpdir / f"{default_stem}_generated_report.txt"
    else:
        out_name = out_name if out_name.lower().endswith(".txt") else out_name + ".txt"
        out_path = tmpdir / out_name
    out_path.write_text(report, encoding="utf-8")
    return str(out_path)


# ----------------------------
# Per-session storage (for non-deepcopyable objects)
# ----------------------------
@dataclass
class SessionData:
    raw_text: str = ""
    state: Optional[ModelState] = None
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)
    last_context: str = ""


SESSIONS: Dict[str, SessionData] = {}


def _sid(request: Optional[gr.Request]) -> str:
    return request.session_hash if request and getattr(request, "session_hash", None) else "global"


def get_session(request: Optional[gr.Request]) -> SessionData:
    sid = _sid(request)
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionData()
    return SESSIONS[sid]


# ----------------------------
# Generation functions
# ----------------------------
def generate_from_file(
    in_file: str,
    softmax_temp: float,
    steer_strength: float,
    geometric_strength: float,
    n_takeaways: int,
    seed: int,
    text_seed: str,
    output_name: str,
    progress: gr.Progress = gr.Progress(),
    request: gr.Request | None = None,
):
    if not in_file:
        raise gr.Error("Please upload an input file.")
    progress(0, desc="Loading file")
    raw = load_text(in_file)

    gen = NeuroSymbolicTextGenerator(
        nodelets_n=10,
        bars_n=100,
        svd_random_state=7,
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        lm_add_k=0.25,
        pillar_strength=0.85,
        pillar_floor=0.25,
        geometric_strength=float(geometric_strength),
    )

    state = gen.fit(raw, progress=progress)
    report = gen.generate_report_from_state(
        lm_text=raw,
        state=state,
        n_takeaways=int(n_takeaways),
        seed=int(seed),
        text_seed=(text_seed or "").strip(),
        prompt_text="",  # initial generation doesn't need a hard prefix prompt
        progress=progress,
    )

    sess = get_session(request)
    sess.raw_text = raw
    sess.state = state
    sess.gen_kwargs = dict(
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        geometric_strength=float(geometric_strength),
    )
    sess.last_context = raw

    progress(0, desc="Saving output file")
    stem = Path(in_file).stem
    out_path = save_report_to_tmp(report, default_stem=stem, output_name=output_name)

    # also prime editor with the report
    return report, str(out_path), report


def generate_from_hf_dataset(
    dataset_name: str,
    config_name: str,
    split: str,
    text_column: str,
    max_samples: int,
    softmax_temp: float,
    steer_strength: float,
    geometric_strength: float,
    n_takeaways: int,
    seed: int,
    text_seed: str,
    output_name: str,
    progress: gr.Progress = gr.Progress(),
    request: gr.Request | None = None,
):
    if not dataset_name or not dataset_name.strip():
        raise gr.Error("Please enter a dataset name.")

    raw = load_hf_dataset(
        dataset_name=dataset_name.strip(),
        config_name=config_name.strip() if config_name and config_name.strip() else None,
        split=split,
        text_column=text_column.strip() if text_column and text_column.strip() else None,
        max_samples=int(max_samples) if max_samples and max_samples > 0 else None,
        progress=progress,
    )

    gen = NeuroSymbolicTextGenerator(
        nodelets_n=10,
        bars_n=100,
        svd_random_state=7,
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        lm_add_k=0.25,
        pillar_strength=0.85,
        pillar_floor=0.25,
        geometric_strength=float(geometric_strength),
    )

    state = gen.fit(raw, progress=progress)
    report = gen.generate_report_from_state(
        lm_text=raw,
        state=state,
        n_takeaways=int(n_takeaways),
        seed=int(seed),
        text_seed=(text_seed or "").strip(),
        prompt_text="",
        progress=progress,
    )

    sess = get_session(request)
    sess.raw_text = raw
    sess.state = state
    sess.gen_kwargs = dict(
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        geometric_strength=float(geometric_strength),
    )
    sess.last_context = raw

    progress(0, desc="Saving output file")
    default_stem = dataset_name.replace("/", "_")
    out_path = save_report_to_tmp(report, default_stem=default_stem, output_name=output_name)

    return report, str(out_path), report


def find_similar_context(
    edited_text: str,
    top_k: int,
    request: gr.Request | None = None,
):
    sess = get_session(request)
    if not sess.state:
        raise gr.Error("No fitted model in this session yet. Generate from a file or dataset first.")

    ctx, rows = retrieve_topk_chunks(sess.state, edited_text, k=int(top_k))
    sess.last_context = ctx
    return ctx, rows


def regenerate_from_edit(
    edited_text: str,
    context_text: str,
    use_context_only: bool,
    softmax_temp: float,
    steer_strength: float,
    geometric_strength: float,
    n_takeaways: int,
    seed: int,
    output_name: str,
    progress: gr.Progress = gr.Progress(),
    request: gr.Request | None = None,
):
    sess = get_session(request)
    if not sess.state:
        raise gr.Error("No fitted model in this session yet. Generate from a file or dataset first.")

    gen = NeuroSymbolicTextGenerator(
        nodelets_n=10,
        bars_n=100,
        svd_random_state=7,
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        lm_add_k=0.25,
        pillar_strength=0.85,
        pillar_floor=0.25,
        geometric_strength=float(geometric_strength),
    )

    edited = (edited_text or "").strip()

    # LM training corpus: base/context + edited text (so edit affects phrase stats)
    if use_context_only:
        base = (context_text or "").strip() or (sess.last_context or "").strip()
        if not base:
            base = sess.raw_text
    else:
        base = "\n\n".join([sess.raw_text, (context_text or "").strip()]).strip()

    lm_text = (base + ("\n\n" + edited if edited else "")).strip()

    report = gen.generate_report_from_state(
        lm_text=lm_text,
        state=sess.state,          # keep the fitted neurosymbolic steering
        n_takeaways=int(n_takeaways),
        seed=int(seed),
        text_seed=edited,          # soft steering
        prompt_text=edited,        # hard prefix continuation (first takeaway)
        progress=progress,
    )

    out_path = save_report_to_tmp(report, default_stem="edited_regen", output_name=output_name)
    return report, str(out_path)


# ----------------------------
# Gradio app
# ----------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Neurosymbolic Text Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Neurosymbolic Text Generator")
        gr.Markdown(
            "*TF‑IDF/SVD nodelets + quad-gram LM + geometric distance/angle modulation + HF dataset support + cosine-sim edit/regenerate (prompt + corpus mix)*"
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
                        output_name_file = gr.Textbox(
                            label="Output filename (optional)",
                            placeholder="e.g., report.txt (leave blank to auto-name)",
                        )
                        text_seed_file = gr.Textbox(
                            label="Text seed (optional)",
                            placeholder="Words/phrase to bias generation (e.g., 'quantum measurement')",
                        )

                with gr.Row():
                    softmax_temp_file = gr.Slider(0.2, 2.5, value=0.85, step=0.05, label="Softmax temp")
                    steer_strength_file = gr.Slider(0.0, 5.0, value=1.35, step=0.05, label="Steer strength")
                    geometric_strength_file = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Geometric strength")

                with gr.Row():
                    n_takeaways_file = gr.Slider(1, 30, value=7, step=1, label="# takeaways")
                    seed_file = gr.Number(value=7, precision=0, label="Numeric seed")

                run_btn_file = gr.Button("Generate sample from file", variant="primary", size="lg")

                with gr.Row():
                    preview_file = gr.Textbox(label="Generated report preview", lines=16)
                    download_file = gr.File(label="Download generated .txt")

                edited_text_file = gr.Textbox(
                    label="Edit text (optional) - used for cosine retrieval and regeneration prompt",
                    lines=8,
                    placeholder="Edit the generated report or paste a paragraph to steer regeneration.",
                )

            with gr.Tab("Hugging Face Dataset"):
                gr.Markdown("### Load data from Hugging Face datasets\nBrowse datasets at: https://huggingface.co/datasets")

                with gr.Row():
                    with gr.Column():
                        dataset_name = gr.Textbox(
                            label="Dataset name",
                            placeholder="e.g., imdb, ag_news, cnn_dailymail",
                            value="",
                        )
                        config_name = gr.Textbox(
                            label="Config/Subset (if required)",
                            placeholder="e.g., 3.0.0 for cnn_dailymail, wikitext-2-raw-v1 for wikitext",
                            value="",
                        )
                    with gr.Column():
                        split = gr.Textbox(
                            label="Split",
                            placeholder="train, test, validation, etc.",
                            value="train",
                        )
                        text_column = gr.Textbox(
                            label="Text column (leave empty for auto-detect)",
                            placeholder="e.g., text, article, context",
                            value="",
                        )

                with gr.Row():
                    max_samples = gr.Number(
                        label="Max samples (0 = all, recommended: 1000-5000)",
                        value=2000,
                        precision=0,
                    )
                    output_name_hf = gr.Textbox(
                        label="Output filename (optional)",
                        placeholder="e.g., report.txt",
                    )

                with gr.Row():
                    text_seed_hf = gr.Textbox(
                        label="Text seed (optional)",
                        placeholder="Words/phrase to bias generation",
                        scale=2,
                    )

                with gr.Row():
                    softmax_temp_hf = gr.Slider(0.2, 2.5, value=0.85, step=0.05, label="Softmax temp")
                    steer_strength_hf = gr.Slider(0.0, 5.0, value=1.35, step=0.05, label="Steer strength")
                    geometric_strength_hf = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Geometric strength")

                with gr.Row():
                    n_takeaways_hf = gr.Slider(1, 30, value=7, step=1, label="# takeaways")
                    seed_hf = gr.Number(value=7, precision=0, label="Numeric seed")

                run_btn_hf = gr.Button("Generate from dataset", variant="primary", size="lg")

                with gr.Row():
                    preview_hf = gr.Textbox(label="Generated report preview", lines=16)
                    download_hf = gr.File(label="Download generated .txt")

                edited_text_hf = gr.Textbox(
                    label="Edit text (optional) - used for cosine retrieval and regeneration prompt",
                    lines=8,
                    placeholder="Edit the generated report or paste a paragraph to steer regeneration.",
                )

            with gr.Tab("Edit & Generate"):
                gr.Markdown(
                    "Paste a prompt, retrieve similar chunks (cosine similarity), then generate.\n"
                    "Regeneration both (1) continues from your edit as a prompt and (2) includes the edit in the LM training corpus."
                )

                edited_text = gr.Textbox(
                    label="Edited text / query",
                    lines=10,
                    placeholder="Paste your edited paragraph here (or copy from the other tabs).",
                )

                with gr.Row():
                    top_k = gr.Slider(1, 50, value=12, step=1, label="Top-k chunks")
                    find_btn = gr.Button("Find similar context", variant="secondary")

                context_text = gr.Textbox(
                    label="Retrieved context (top-k chunks concatenated)",
                    lines=12,
                )

                matches = gr.Dataframe(
                    headers=["rank", "cosine_sim", "snippet"],
                    datatype=["number", "number", "str"],
                    row_count=12,
                    col_count=(3, "fixed"),
                    label="Top matches",
                )

                with gr.Row():
                    use_context_only = gr.Checkbox(value=True, label="Use retrieved context only (recommended)")
                    regen_output_name = gr.Textbox(
                        label="Output filename (optional)",
                        placeholder="e.g., regen.txt",
                    )

                with gr.Row():
                    softmax_temp_regen = gr.Slider(0.2, 2.5, value=0.85, step=0.05, label="Softmax temp")
                    steer_strength_regen = gr.Slider(0.0, 5.0, value=1.35, step=0.05, label="Steer strength")
                    geometric_strength_regen = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Geometric strength")

                with gr.Row():
                    n_takeaways_regen = gr.Slider(1, 30, value=7, step=1, label="# takeaways")
                    seed_regen = gr.Number(value=7, precision=0, label="Numeric seed")

                regen_btn = gr.Button("generate", variant="primary", size="lg")

                with gr.Row():
                    regen_preview = gr.Textbox(label="generated report preview", lines=16)
                    regen_download = gr.File(label="Download generated .txt")

        # Wiring
        run_btn_file.click(
            fn=generate_from_file,
            inputs=[
                in_file,
                softmax_temp_file,
                steer_strength_file,
                geometric_strength_file,
                n_takeaways_file,
                seed_file,
                text_seed_file,
                output_name_file,
            ],
            outputs=[preview_file, download_file, edited_text_file],
        )

        run_btn_hf.click(
            fn=generate_from_hf_dataset,
            inputs=[
                dataset_name,
                config_name,
                split,
                text_column,
                max_samples,
                softmax_temp_hf,
                steer_strength_hf,
                geometric_strength_hf,
                n_takeaways_hf,
                seed_hf,
                text_seed_hf,
                output_name_hf,
            ],
            outputs=[preview_hf, download_hf, edited_text_hf],
        )

        # Convenience: push edits into Edit & Regenerate tab
        edited_text_file.change(lambda x: x, edited_text_file, edited_text)
        edited_text_hf.change(lambda x: x, edited_text_hf, edited_text)

        find_btn.click(
            fn=find_similar_context,
            inputs=[edited_text, top_k],
            outputs=[context_text, matches],
        )

        regen_btn.click(
            fn=regenerate_from_edit,
            inputs=[
                edited_text,
                context_text,
                use_context_only,
                softmax_temp_regen,
                steer_strength_regen,
                geometric_strength_regen,
                n_takeaways_regen,
                seed_regen,
                regen_output_name,
            ],
            outputs=[regen_preview, regen_download],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()

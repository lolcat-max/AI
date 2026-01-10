#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neurosymbolic Text Generator (Gradio GUI)

Install:
  pip install numpy scikit-learn gradio

Optional (for file types):
  pip install python-docx
  pip install pypdf

Run:
  python neurosymbolic_textgen_gradio.py
"""

from __future__ import annotations

import re
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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
        try:
            import docx  # python-docx
        except Exception as e:
            raise RuntimeError("Reading .docx requires: pip install python-docx") from e
        d = docx.Document(str(p))
        return "\n".join([para.text for para in d.paragraphs])

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError("Reading .pdf requires: pip install pypdf") from e
        reader = PdfReader(str(p))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    raise ValueError(f"Unsupported file extension: {ext}")


# ----------------------------
# Text utilities
# ----------------------------
STOPWORDS = set("""
a an and are as at be by for from has have he her hers him his i in is it its
me my of on or our ours she so that the their them they this to was we were
what when where which who will with you your yours
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


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    return float(-np.sum(p * np.log(p + 1e-12)))


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ----------------------------
# Tokenization + simple trigram LM
# ----------------------------
def basic_tokenize(text: str) -> List[str]:
    """
    Tokenizer designed for unstructured text:
    - keeps words and a small set of punctuation tokens
    - lowercases words
    """
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


class TrigramLM:
    """
    Trigram LM with add-k smoothing and backoff to bigram/unigram.
    Stored as counters.
    """
    def __init__(self, add_k: float = 0.25):
        self.add_k = add_k
        self.uni: Dict[str, int] = {}
        self.bi: Dict[Tuple[str, str], int] = {}
        self.tri: Dict[Tuple[str, str, str], int] = {}
        self.vocab: List[str] = []
        self.total: int = 0

    def fit(self, tokens: List[str]) -> None:
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()
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

    def next_distribution(self, w1: str, w2: str) -> Tuple[List[str], np.ndarray]:
        cont: List[str] = []
        for (a, b, c), _count in self.tri.items():
            if a == w1 and b == w2:
                cont.append(c)

        if not cont:
            for (a, b), _count in self.bi.items():
                if a == w2:
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
        probs = np.array([self._prob_trigram(w1, w2, w) for w in cand], dtype=float)
        probs = probs / (probs.sum() + 1e-12)
        return cand, probs


# ----------------------------
# Neurosymbolic model pieces
# ----------------------------
@dataclass
class Nodelet:
    idx: int
    top_terms: List[Tuple[str, float]]   # (term, weight)
    energy: float                        # node strength
    narrative: str                       # human-readable description


@dataclass
class ModelState:
    nodelets: List[Nodelet]
    vocab100: List[str]
    binding_W: np.ndarray
    bar_probs: np.ndarray
    bar_logits: np.ndarray
    token_boost: Dict[str, float]


class NeuroSymbolicTextGenerator:
    """
    Interpretable neurosymbolic generator:
    - Nodelets: latent semantic factors (TF-IDF->SVD)
    - Bars: top 100 vocab items
    - W: bindings nodelets->bars
    - Takeaways generated by a trigram LM trained on the file, steered by bar_probs and nodelet bindings
    """

    def __init__(
        self,
        nodelets_n: int = 10,
        bars_n: int = 100,
        svd_random_state: int = 7,
        softmax_temp: float = 0.85,
        steer_strength: float = 1.35,
        lm_add_k: float = 0.25,
    ):
        self.nodelets_n = nodelets_n
        self.bars_n = bars_n
        self.svd_random_state = svd_random_state
        self.softmax_temp = softmax_temp
        self.steer_strength = steer_strength
        self.lm_add_k = lm_add_k

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

    def fit(self, text: str) -> ModelState:
        text = normalize(text)
        chunks = self._chunk_text(text)

        if not chunks or len(" ".join(chunks).strip()) < 250:
            raise ValueError("Not enough readable text. Provide a longer unstructured file.")

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

        k = min(self.nodelets_n, max(2, X.shape[0] - 1), max(2, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.svd_random_state)
        svd.fit(X)
        components = svd.components_

        nodelets: List[Nodelet] = []
        for i in range(k):
            comp = components[i]
            term_idx = np.argsort(-np.abs(comp))[:20]
            terms = [(vocab[j], float(comp[j])) for j in term_idx]
            energy = float(np.linalg.norm(comp))
            narrative = self._nodelet_narrative(i, terms, energy)
            nodelets.append(Nodelet(idx=i, top_terms=terms, energy=energy, narrative=narrative))

        W = np.zeros((k, self.bars_n), dtype=float)
        for i in range(k):
            weights = {t: abs(w) for t, w in nodelets[i].top_terms}
            for b, term in enumerate(vocab100):
                base = weights.get(term, 0.0)
                if base == 0.0:
                    toks = set(term.split())
                    near = 0.0
                    for tt, ww in weights.items():
                        if toks & set(tt.split()):
                            near = max(near, 0.35 * ww)
                    base = near
                W[i, b] = base

            mx = W[i].max()
            if mx > 1e-12:
                W[i] /= mx

        energies = np.array([n.energy for n in nodelets], dtype=float)
        energies = energies / (energies.max() + 1e-12)
        logits = (energies.reshape(-1, 1) * W).sum(axis=0)

        rng = np.random.default_rng(self.svd_random_state)
        logits = logits + 0.02 * rng.normal(size=logits.shape)
        probs = softmax(logits, temp=self.softmax_temp)

        token_boost = self._make_token_boost(vocab100, probs, nodelets)

        return ModelState(
            nodelets=nodelets,
            vocab100=vocab100,
            binding_W=W,
            bar_probs=probs,
            bar_logits=logits,
            token_boost=token_boost,
        )

    def _make_token_boost(
        self,
        vocab100: List[str],
        bar_probs: np.ndarray,
        nodelets: List[Nodelet],
    ) -> Dict[str, float]:
        boost: Dict[str, float] = {}

        for term, p in zip(vocab100, bar_probs):
            for w in term.split():
                if len(w) <= 2 or w in STOPWORDS:
                    continue
                boost[w] = max(boost.get(w, 0.0), float(math.log(p + 1e-12) + 6.0))

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
        lm: TrigramLM,
        w1: str,
        w2: str,
        token_boost: Dict[str, float],
        rng: np.random.Generator,
        temperature: float = 0.95,
    ) -> str:
        cand, base_p = lm.next_distribution(w1, w2)

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
        lm: TrigramLM,
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
        lm: TrigramLM,
        token_boost: Dict[str, float],
        rng: np.random.Generator,
        seed_words: Optional[List[str]] = None,
        min_len: int = 800,
        max_len: int = 900,
    ) -> str:
        seed_words = seed_words or []
        seed = self._choose_start_word(lm, token_boost, rng, seed_words)

        tokens = ["("] if rng.random() < 0.12 else []
        tokens.append(seed)

        w1 = tokens[-1]
        w2 = tokens[-1]

        target_len = int(rng.integers(min_len, max_len + 1))
        for _ in range(target_len):
            nxt = self._sample_next(lm, w1, w2, token_boost, rng)
            tokens.append(nxt)
            w1, w2 = w2, nxt

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
    ) -> List[str]:
        tokens = basic_tokenize(text)
        if len(tokens) < 600:
            tokens = tokens * 2

        lm = TrigramLM(add_k=self.lm_add_k)
        lm.fit(tokens)

        seed_words = self._extract_seed_words(text_seed)

        takeaways = []
        energies = np.array([n.energy for n in state.nodelets], dtype=float)
        energies = energies / (energies.max() + 1e-12)

        for _i in range(n_takeaways):
            lead = int(rng.choice(len(state.nodelets), p=softmax(energies, temp=0.9)))
            row = state.binding_W[lead]

            local_boost = dict(state.token_boost)

            # Persistent, light steering from user text seed
            for w in seed_words:
                local_boost[w] = max(local_boost.get(w, 0.0), 1.25)

            # Stronger per-takeaway steering from selected nodelet bindings
            topbars = np.argsort(-row)[:10]
            for b in topbars:
                term = state.vocab100[b]
                strength = float(row[b])  # 0..1
                for w in term.split():
                    if len(w) > 2 and w not in STOPWORDS:
                        local_boost[w] = max(local_boost.get(w, 0.0), 1.0 + 0.6 * strength)

            for k in list(local_boost.keys()):
                local_boost[k] = float(min(2.0, max(0.0, local_boost[k])))

            sent = self._generate_sentence(lm, local_boost, rng, seed_words=seed_words)
            takeaways.append(sent)

        return takeaways

    def generate_report(
        self,
        text: str,
        n_takeaways: int = 7,
        seed: int = 7,
        text_seed: str = "",
    ) -> str:
        text = normalize(text)
        state = self.fit(text)

        _H = entropy(state.bar_probs)
        _effk = math.exp(_H)

        rng = np.random.default_rng(seed)
        takeaways = self._generate_takeaways(text, state, n_takeaways=n_takeaways, rng=rng, text_seed=text_seed)
        return "\n\n".join(takeaways)


# ----------------------------
# Gradio app
# ----------------------------
def generate_from_file(
    in_file: str,
    softmax_temp: float,
    steer_strength: float,
    n_takeaways: int,
    seed: int,
    text_seed: str,
    output_name: str,
):
    if not in_file:
        raise gr.Error("Please upload an input file.")

    raw = load_text(in_file)

    gen = NeuroSymbolicTextGenerator(
        nodelets_n=10,
        bars_n=100,
        svd_random_state=7,
        softmax_temp=float(softmax_temp),
        steer_strength=float(steer_strength),
        lm_add_k=0.25,
    )

    report = gen.generate_report(
        raw,
        n_takeaways=int(n_takeaways),
        seed=int(seed),
        text_seed=(text_seed or "").strip(),
    )

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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Neurosymbolic Text Generator") as demo:
        gr.Markdown("## Neurosymbolic Text Generator (TF‑IDF/SVD nodelets + trigram LM)")

        with gr.Row():
            in_file = gr.File(
                label="Input file (.txt/.md/.docx/.pdf)",
                file_types=[".txt", ".md", ".docx", ".pdf"],
                type="filepath",
            )
            with gr.Column():
                output_name = gr.Textbox(
                    label="Output filename (optional)",
                    placeholder="e.g., report.txt (leave blank to auto-name)",
                )
                text_seed = gr.Textbox(
                    label="Text seed (optional)",
                    placeholder="Words/phrase to bias starts + token boosts (e.g., 'quantum measurement')",
                )

        with gr.Row():
            softmax_temp = gr.Slider(0.2, 2.5, value=0.85, step=0.05, label="Softmax temp")
            steer_strength = gr.Slider(0.0, 5.0, value=1.35, step=0.05, label="Steer strength")

        with gr.Row():
            n_takeaways = gr.Slider(1, 30, value=7, step=1, label="# takeaways")
            seed = gr.Number(value=7, precision=0, label="Numeric seed")

        run_btn = gr.Button("Generate report", variant="primary")

        with gr.Row():
            preview = gr.Textbox(label="Generated report preview", lines=20)
        download = gr.File(label="Download generated .txt")

        run_btn.click(
            fn=generate_from_file,
            inputs=[in_file, softmax_temp, steer_strength, n_takeaways, seed, text_seed, output_name],
            outputs=[preview, download],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()

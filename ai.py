import math
import random
import hashlib
from functools import singledispatch
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# GUI + persistence
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sqlite3
import os

# Optional: Hugging Face datasets
_HAS_HF = True
try:
    from datasets import load_dataset  # pip install datasets
except Exception:
    _HAS_HF = False


# ================================================================
# Deterministic dtype-based "sublimation" swaps
# ================================================================

def _rng_from_context(context: str) -> np.random.Generator:
    h = hashlib.sha256(context.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)

def _swap_index_pairs(n: int, rng: np.random.Generator, swap_frac: float = 0.3):
    if n <= 1:
        return []
    k = max(1, int(swap_frac * n) // 2)
    idx = rng.permutation(n)
    pairs = [(int(idx[2*i]), int(idx[2*i+1])) for i in range(min(k, n // 2))]
    return pairs

def _apply_pairs_permutation(seq, pairs):
    out = list(seq)
    for i, j in pairs:
        out[i], out[j] = out[j], out[i]
    return out

def _apply_pairs_permutation_array(arr: np.ndarray, pairs):
    out = arr.copy()
    for i, j in pairs:
        out[i], out[j] = out[j], out[i]
    return out

def safe_cast(x: np.ndarray, dtype, casting: str = "same_kind") -> np.ndarray:
    return x.astype(dtype, casting=casting, copy=False)

@singledispatch
def sublimate(x, *, context: str = "", swap_frac: float = 0.3, casting: str = "same_kind"):
    return x

@sublimate.register
def _(x: str, *, context: str = "", swap_frac: float = 0.3, casting: str = "same_kind"):
    if len(x) <= 1:
        return x
    rng = _rng_from_context(context + "|str")
    chars = list(x)
    pairs = _swap_index_pairs(len(chars), rng, swap_frac)
    out = _apply_pairs_permutation(chars, pairs)
    return "".join(out)

@sublimate.register
def _(x: list, *, context: str = "", swap_frac: float = 0.3, casting: str = "same_kind"):
    if len(x) <= 1:
        return x
    rng = _rng_from_context(context + "|list")
    pairs = _swap_index_pairs(len(x), rng, swap_frac)
    out = _apply_pairs_permutation(x, pairs)
    return out

@sublimate.register
def _(x: np.ndarray, *, context: str = "", swap_frac: float = 0.3, casting: str = "same_kind"):
    rng = _rng_from_context(context + "|nd")
    arr = x
    was_1d = (arr.ndim == 1)
    if not was_1d:
        arr = arr.reshape(-1)

    kind = arr.dtype.kind
    if kind in ("i", "u", "f", "c"):
        pairs = _swap_index_pairs(arr.size, rng, swap_frac)
        out = _apply_pairs_permutation_array(arr, pairs)
        out = safe_cast(out, arr.dtype, casting=casting)
    elif kind in ("S", "U", "O", "V"):
        pairs = _swap_index_pairs(arr.size, rng, swap_frac)
        out = _apply_pairs_permutation_array(arr, pairs)
    else:
        out = arr

    if was_1d:
        return out
    return out.reshape(x.shape)

def sublimate_candidates_and_probs(
    candidates: List[str],
    probs: np.ndarray,
    *,
    context: str,
    swap_frac: float = 0.3,
):
    if len(candidates) == 0:
        return candidates, probs
    rng = _rng_from_context(context + "|pair")
    pairs = _swap_index_pairs(len(candidates), rng, swap_frac)
    new_candidates = _apply_pairs_permutation(candidates, pairs)
    if probs is None or probs.ndim != 1 or probs.size != len(candidates):
        return new_candidates, probs
    new_probs = _apply_pairs_permutation_array(probs, pairs)
    s = float(new_probs.sum())
    if not np.isfinite(s) or s <= 0.0:
        new_probs = np.ones_like(new_probs, dtype=float) / len(new_probs)
    else:
        new_probs = new_probs / s
    return new_candidates, new_probs


# ================================================================
# Embedded Template Engine
# ================================================================

class EmbeddedTemplateMatcher:
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim
        self.templates: List[Tuple[str, np.ndarray]] = []
        self.word_vectors: Dict[str, np.ndarray] = {}

    def _rng_from_word(self, word: str) -> np.random.Generator:
        hash_bytes = hashlib.sha256(word.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:8], "little")
        return np.random.default_rng(seed)

    def _get_word_vector(self, word: str) -> np.ndarray:
        if word in self.word_vectors:
            return self.word_vectors[word]
        rng = self._rng_from_word(word)
        vec = rng.random(self.vector_dim) - 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.word_vectors[word] = vec
        return vec

    def _get_sentence_vector(self, sentence: str) -> np.ndarray:
        words = sentence.split()
        if not words:
            return np.zeros(self.vector_dim)
        vectors = [self._get_word_vector(w) for w in words]
        avg_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg_vec)
        return avg_vec / norm if norm > 0 else np.zeros(self.vector_dim)

    def build_templates_from_corpus(self, corpus: List[str], num_templates: int = 50):
        step = max(1, len(corpus) // max(1, num_templates))
        for i in range(0, len(corpus), step):
            sentence = corpus[i]
            if len(sentence.split()) > 5:
                template_vector = self._get_sentence_vector(sentence)
                self.templates.append((sentence, template_vector))

    def find_best_template(self, context: str) -> Optional[Tuple[str, np.ndarray]]:
        if not self.templates or not context:
            return None
        context_vector = self._get_sentence_vector(context)
        template_vectors = np.array([t[1] for t in self.templates])
        if template_vectors.size == 0:
            return None
        similarities = np.dot(template_vectors, context_vector)
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        if best_similarity > 0.4:
            return self.templates[best_idx]
        return None


# ================================================================
# RL memory with SQLite (incremental bandit updates)
# ================================================================

class RLMemory:
    def __init__(self, db_path: str = "rl_cache.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS bandit (
                context TEXT NOT NULL,
                action  TEXT NOT NULL,
                n       INTEGER NOT NULL DEFAULT 0,
                q       REAL    NOT NULL DEFAULT 0.0,
                PRIMARY KEY (context, action)
            )
        """)
        self.conn.commit()

    def get_qs(self, context: str, actions: List[str]) -> np.ndarray:
        qs = np.zeros(len(actions), dtype=float)
        if not actions:
            return qs
        for i, a in enumerate(actions):
            row = self.cur.execute(
                "SELECT n, q FROM bandit WHERE context=? AND action=?",
                (context, a)
            ).fetchone()
            if row is not None:
                qs[i] = float(row[1])
        return qs

    def update_batch(self, decisions: List[Tuple[str, str]], reward: float):
        for ctx, act in decisions:
            row = self.cur.execute(
                "SELECT n, q FROM bandit WHERE context=? AND action=?",
                (ctx, act)
            ).fetchone()
            if row is None:
                n, q = 0, 0.0
            else:
                n, q = int(row[0]), float(row[1])
            n_new = n + 1
            q_new = q + (reward - q) / max(1, n_new)
            self.cur.execute("""
                INSERT INTO bandit (context, action, n, q)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context, action)
                DO UPDATE SET n=excluded.n, q=excluded.q
            """, (ctx, act, n_new, q_new))
        self.conn.commit()

    def bias_probs(self, context: str, candidates: List[str], probs: np.ndarray, beta: float = 2.0) -> np.ndarray:
        if len(candidates) == 0:
            return probs
        qs = self.get_qs(context, candidates)
        weights = np.exp(beta * qs)
        out = probs * weights
        s = out.sum()
        if s > 0:
            out = out / s
        else:
            out = np.ones_like(out) / len(out)
        return out

    def close(self):
        self.conn.commit()
        self.conn.close()


# ================================================================
# OrigamiLLM (n-gram + templates + dtype sublimate) + RL integration
# ================================================================

class OrigamiLLM:
    def __init__(self, filename: Optional[str] = None, *, text: Optional[str] = None):
        if text is None:
            if filename is None:
                raise ValueError("filename or text must be provided")
            with open(filename, encoding="utf-8") as f:
                text = f.read()
        self.corpus = text.lower().split(".")
        self.bigrams: Dict[str, List[str]] = {}
        self.trigrams: Dict[Tuple[str, str], List[str]] = {}
        self.build_model()
        self.template_matcher = EmbeddedTemplateMatcher()
        self.template_matcher.build_templates_from_corpus(self.corpus)

    def build_model(self):
        for sentence in self.corpus:
            words = sentence.split()
            if len(words) < 3:
                continue
            for i in range(len(words) - 1):
                self.bigrams.setdefault(words[i], []).append(words[i + 1])
            for i in range(len(words) - 2):
                key = (words[i], words[i + 1])
                self.trigrams.setdefault(key, []).append(words[i + 2])

    def generate_text(
        self,
        start_word: Optional[str] = None,
        max_words: int = 200,
        stack_state: float = 1.0,
        chirality_strength: float = 1.0,
        swap_frac: float = 0.35,
    ) -> str:
        text, _ = self.generate_text_rl(
            start_word=start_word,
            max_words=max_words,
            stack_state=stack_state,
            chirality_strength=chirality_strength,
            swap_frac=swap_frac,
            rl_memory=None,
            rl_beta=0.0,
        )
        return text

    def generate_text_rl(
        self,
        start_word: Optional[str],
        max_words: int,
        stack_state: float,
        chirality_strength: float,
        swap_frac: float,
        rl_memory: Optional[RLMemory],
        rl_beta: float = 2.0,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        if not self.bigrams:
            return "", []
        rng = np.random.default_rng()
        if start_word is None or start_word not in self.bigrams:
            start_word = rng.choice(list(self.bigrams.keys()))
        result = [start_word]
        current_word = start_word
        decisions: List[Tuple[str, str]] = []

        for _ in range(max_words - 1):
            use_model = rng.random() < stack_state
            candidates: List[str] = []
            if use_model:
                trigram_key = tuple(result[-2:]) if len(result) >= 2 else None
                if trigram_key and trigram_key in self.trigrams:
                    candidates = self.trigrams[trigram_key]
                elif current_word in self.bigrams:
                    candidates = self.bigrams[current_word]
            if not candidates and self.bigrams:
                vocab = list(self.bigrams.keys())
                k = min(5, len(vocab))
                candidates = list(rng.choice(vocab, size=k, replace=False))

            probs = np.ones(len(candidates), dtype=float)
            s = float(probs.sum())
            probs = probs / s if s > 0 else probs

            context = " ".join(result[-5:])
            best_template_tuple = self.template_matcher.find_best_template(context)

            if chirality_strength > 0 and best_template_tuple is not None and candidates:
                template_text, _ = best_template_tuple
                template_words = template_text.split()
                try:
                    idx = template_words.index(result[-1])
                    if idx < len(template_words) - 1:
                        template_next_word = template_words[idx + 1]
                        if template_next_word in candidates:
                            boost_idx = candidates.index(template_next_word)
                            twist_factor = math.exp(chirality_strength)
                            probs[boost_idx] += 5.0 * chirality_strength * twist_factor
                except (ValueError, IndexError):
                    pass

            if len(candidates) > 1:
                candidates, probs = sublimate_candidates_and_probs(
                    candidates, probs, context=context, swap_frac=swap_frac
                )

            if rl_memory is not None and len(candidates) > 0:
                probs = rl_memory.bias_probs(context, candidates, probs, beta=rl_beta)

            total = float(probs.sum())
            probs = probs / total if total > 0 else np.ones_like(probs) / max(1, len(probs))

            next_word = rng.choice(candidates, p=probs) if candidates else rng.choice(list(self.bigrams.keys()))
            result.append(next_word)
            decisions.append((context, next_word))
            current_word = next_word

        generated = " ".join(result)
        return generated, decisions


# ================================================================
# Tkinter GUI with dataset loaders, corpus limit, output length, editable text reinforcement, beta slider, and mode dropdown
# ================================================================

class RLApp(tk.Tk):
    HF_DATASETS = [
        "facebook/natural_reasoning",
        "ag_news"

    ]

    def __init__(self):
        super().__init__()
        self.title("OrigamiLLM RL GUI")
        self.geometry("1800x860")

        # --- Source selectors ---
        source_bar = ttk.Frame(self)
        source_bar.pack(fill="x", padx=8, pady=8)

        ttk.Label(source_bar, text="Source:").pack(side="left")
        self.source_var = tk.StringVar(value="File")
        self.source_combo = ttk.Combobox(
            source_bar, textvariable=self.source_var, state="readonly",
            values=["File", "HF dataset", "Generic"], width=12
        )
        self.source_combo.pack(side="left", padx=6)
        self.source_combo.bind("<<ComboboxSelected>>", lambda e: self.on_source_change())

        # File loader
        self.file_btn = ttk.Button(source_bar, text="Open file...", command=self.on_open_file)
        self.file_btn.pack(side="left", padx=6)

        # HF dataset controls
        ttk.Label(source_bar, text="Dataset:").pack(side="left", padx=(12, 2))
        self.hf_ds_var = tk.StringVar(value=self.HF_DATASETS[0])
        self.hf_ds_combo = ttk.Combobox(
            source_bar, textvariable=self.hf_ds_var, state="readonly",
            values=self.HF_DATASETS, width=26
        )
        self.hf_ds_combo.pack(side="left", padx=6)

        ttk.Label(source_bar, text="Split:").pack(side="left", padx=(12, 2))
        self.hf_split_var = tk.StringVar(value="train")
        self.hf_split_combo = ttk.Combobox(
            source_bar, textvariable=self.hf_split_var, state="readonly",
            values=["train", "validation", "test"], width=12
        )
        self.hf_split_combo.pack(side="left", padx=6)

        self.hf_load_btn = ttk.Button(source_bar, text="Load HF", command=self.on_load_hf)
        self.hf_load_btn.pack(side="left", padx=6)

        # Generic builder controls
        ttk.Label(source_bar, text="Builder:").pack(side="left", padx=(12, 2))
        self.builder_var = tk.StringVar(value="text")  # csv|json|parquet|text|arrow|webdataset
        self.builder_combo = ttk.Combobox(
            source_bar, textvariable=self.builder_var, state="readonly",
            values=["text", "csv", "json", "parquet", "arrow", "webdataset"], width=12
        )
        self.builder_combo.pack(side="left", padx=6)

        ttk.Label(source_bar, text="data_files:").pack(side="left", padx=(12, 2))
        self.data_files_var = tk.StringVar()
        self.data_files_entry = ttk.Entry(source_bar, width=44, textvariable=self.data_files_var)
        self.data_files_entry.pack(side="left", padx=6)

        ttk.Label(source_bar, text="json field:").pack(side="left", padx=(12, 2))
        self.json_field_var = tk.StringVar()
        self.json_field_entry = ttk.Entry(source_bar, width=16, textvariable=self.json_field_var)
        self.json_field_entry.pack(side="left", padx=6)

        self.generic_load_btn = ttk.Button(source_bar, text="Load generic", command=self.on_load_generic)
        self.generic_load_btn.pack(side="left", padx=6)

        # Visibility by source
        self._set_controls_by_source("File")

        # --- Model + memory initialization ---
        self.llm: Optional[OrigamiLLM] = None
        self.memory = RLMemory("rl_cache.sqlite")
        self.current_text_source: str = ""

        # --- Top controls: start word, generate, RL, mode, beta, corpus KB limit, output length ---
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Start word:").pack(side="left")
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(top, width=24, textvariable=self.start_var)
        self.start_entry.state(["!disabled", "!readonly"])
        self.start_entry.configure(state="normal")
        self.start_entry.pack(side="left", padx=6)
        self.start_entry.focus_set()
        self.start_entry.bind("<Return>", lambda e: self.on_generate())

        self.gen_button = ttk.Button(top, text="Generate", command=self.on_generate, state="disabled")
        self.gen_button.pack(side="left", padx=6)

        self.reward_button = ttk.Button(top, text="Reward (+)", command=self.on_reward, state="disabled")
        self.reward_button.pack(side="left", padx=6)
        self.punish_button = ttk.Button(top, text="Punish (-)", command=self.on_punish, state="disabled")
        self.punish_button.pack(side="left", padx=6)

        self.use_edited_var = tk.BooleanVar(value=True)
        self.use_edited_chk = ttk.Checkbutton(
            top, text="Train on edited text", variable=self.use_edited_var
        )
        self.use_edited_chk.pack(side="left", padx=8)

        ttk.Label(top, text="Mode:").pack(side="left", padx=(12, 2))
        self.mode_var = tk.StringVar(value="Template+RL")
        self.mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            state="readonly",
            values=["Uniform only", "RL only", "Template only", "Template+RL", "Chaotic"],
            width=16,
        )
        self.mode_combo.pack(side="left", padx=6)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_mode_change())

        ttk.Label(top, text="RL β:").pack(side="left", padx=(12, 2))
        self.beta_var = tk.DoubleVar(value=2.0)
        self.beta_scale = ttk.Scale(
            top,
            from_=0.0,
            to=6.0,
            orient="horizontal",
            length=220,
            variable=self.beta_var,
            command=lambda v: self.on_beta_change(v),
        )
        self.beta_scale.pack(side="left", padx=6)
        self.beta_value_lbl = ttk.Label(top, text=f"{self.beta_var.get():.2f}")
        self.beta_value_lbl.pack(side="left", padx=4)

        ttk.Label(top, text="Corpus limit (KB):").pack(side="left", padx=(12, 2))
        self.corpus_kb_var = tk.IntVar(value=2048)  # default 2 MB
        self.corpus_kb_scale = ttk.Scale(
            top,
            from_=64,
            to=1024*256,   # up to 256 MB in KB units
            orient="horizontal",
            length=260,
            variable=self.corpus_kb_var,
            command=lambda v: self.on_kb_change(v),
        )
        self.corpus_kb_scale.pack(side="left", padx=6)
        self.corpus_kb_lbl = ttk.Label(top, text=f"{self.corpus_kb_var.get():d}")
        self.corpus_kb_lbl.pack(side="left", padx=4)

        ttk.Label(top, text="Output length:").pack(side="left", padx=(12, 2))
        self.out_len_var = tk.IntVar(value=300)
        self.out_len_scale = ttk.Scale(
            top,
            from_=10,
            to=3000,
            orient="horizontal",
            length=260,
            variable=self.out_len_var,
            command=lambda v: self.on_len_change(v),
        )
        self.out_len_scale.pack(side="left", padx=6)
        self.out_len_lbl = ttk.Label(top, text=f"{self.out_len_var.get():d}")
        self.out_len_lbl.pack(side="left", padx=4)

        # Text area
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.text = tk.Text(text_frame, wrap="word")
        self.text.pack(fill="both", expand=True)

        # Status bar
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=8)
        self.status = ttk.Label(bottom, text="Select a source: File, HF dataset, or Generic.")
        self.status.pack(side="left")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.last_decisions: List[Tuple[str, str]] = []

    # ---------- UI visibility ----------
    def _set_controls_by_source(self, source: str):
        is_hf = (source == "HF dataset")
        is_gen = (source == "Generic")
        self.file_btn.configure(state="normal" if source == "File" else "disabled")
        for w in (self.hf_ds_combo, self.hf_split_combo, self.hf_load_btn):
            w.configure(state="normal" if is_hf else "disabled")
        for w in (self.builder_combo, self.data_files_entry, self.json_field_entry, self.generic_load_btn):
            w.configure(state="normal" if is_gen else "disabled")

    def on_source_change(self):
        source = self.source_var.get()
        self._set_controls_by_source(source)
        self.status.config(text=f"Source = {source}.")
        self._set_generate_enabled(False)

    # ---------- Sliders and labels ----------
    def on_mode_change(self):
        self.status.config(text=f"Mode set to {self.mode_var.get()}.")

    def on_beta_change(self, value: str):
        try:
            v = float(value)
        except Exception:
            return
        step = 0.05
        snapped = round(v / step) * step
        if abs(snapped - self.beta_var.get()) > 1e-9:
            self.beta_var.set(snapped)
        self.beta_value_lbl.config(text=f"{self.beta_var.get():.2f}")

    def on_kb_change(self, value: str):
        try:
            v = float(value)
        except Exception:
            return
        step = 64  # KB
        snapped = int(round(v / step) * step)
        if snapped != self.corpus_kb_var.get():
            self.corpus_kb_var.set(snapped)
        self.corpus_kb_lbl.config(text=f"{self.corpus_kb_var.get():d}")

    def on_len_change(self, value: str):
        try:
            v = float(value)
        except Exception:
            return
        step = 10
        snapped = int(round(v / step) * step)
        if snapped != self.out_len_var.get():
            self.out_len_var.set(snapped)
        self.out_len_lbl.config(text=f"{self.out_len_var.get():d}")

    # ---------- Corpus ingestion ----------
    def on_open_file(self):
        path = filedialog.askopenfilename(
            title="Select corpus file (.txt)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            kb = max(1, int(self.corpus_kb_var.get()))
            size_bytes = kb * 1024
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read(size_bytes)
            self.llm = OrigamiLLM(text=text)
            self.current_text_source = f"File: {os.path.basename(path)} (<= {kb} KB)"
            self._set_generate_enabled(True)
            self.status.config(text=f"Loaded file (<= {kb} KB): {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def on_load_hf(self):
        if not _HAS_HF:
            messagebox.showerror("Missing dependency", "Install 'datasets' to use HF datasets.")
            return
        ds_id = self.hf_ds_var.get()
        split = self.hf_split_var.get()
        try:
            if ":" in ds_id:
                base, config = ds_id.split(":", 1)
                ds = load_dataset(base, config, split=split)
            else:
                ds = load_dataset(ds_id, split=split)
            corpus_text = self._flatten_dataset_to_text(ds)
            if not corpus_text.strip():
                raise RuntimeError("Loaded dataset produced empty text; try another split/dataset.")
            self.llm = OrigamiLLM(text=corpus_text)
            self.current_text_source = f"HF: {ds_id} [{split}]"
            self._set_generate_enabled(True)
            kb = int(self.corpus_kb_var.get())
            self.status.config(text=f"Loaded HF dataset: {ds_id} ({split}), size={len(ds)} | limit={kb} KB")
        except Exception as e:
            if "scripts" in str(e).lower() and "supported" in str(e).lower():
                messagebox.showinfo(
                    "Script unsupported",
                    "This dataset relies on a deprecated loader script.\nUse the Generic loader (csv/json/parquet/text) with data_files instead."
                )
            else:
                messagebox.showerror("HF load error", str(e))

    def on_load_generic(self):
        if not _HAS_HF:
            messagebox.showerror("Missing dependency", "Install 'datasets' to use generic loader.")
            return
        builder = self.builder_var.get()
        split = self.hf_split_var.get()
        raw = self.data_files_var.get().strip()
        if not raw:
            messagebox.showerror("Input required", "Provide one or more data_files paths or hf:// URLs.")
            return
        files = [p for p in raw.replace(",", " ").split() if p]
        kwargs = {"data_files": files}
        field = self.json_field_var.get().strip()
        if builder == "json" and field:
            kwargs["field"] = field
        try:
            ds = load_dataset(builder, **kwargs, split=split)
            corpus_text = self._flatten_dataset_to_text(ds)
            if not corpus_text.strip():
                raise RuntimeError("Generic load produced empty text; check builder/data_files/field.")
            self.llm = OrigamiLLM(text=corpus_text)
            self.current_text_source = f"{builder} data_files [{split}]"
            self._set_generate_enabled(True)
            kb = int(self.corpus_kb_var.get())
            self.status.config(text=f"Loaded generic dataset: builder={builder}, files={len(files)}, split={split}, size={len(ds)} | limit={kb} KB")
        except Exception as e:
            messagebox.showerror("Generic load error", str(e))

    def _flatten_dataset_to_text(self, ds) -> str:
        def collect_text(obj: Any, out: List[str]):
            if isinstance(obj, str):
                out.append(obj)
            elif isinstance(obj, list):
                for x in obj:
                    collect_text(x, out)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_text(v, out)

        chunks: List[str] = []
        kb = max(1, int(self.corpus_kb_var.get()))
        limit_bytes = kb * 1024
        used = 0
        max_items = min(len(ds), 50000)

        for i in range(max_items):
            if used >= limit_bytes:
                break
            ex = ds[i]
            pieces: List[str] = []
            collect_text(ex, pieces)
            if not pieces:
                continue
            seg = " ".join(pieces)
            seg_b = seg.encode("utf-8")
            if used + len(seg_b) > limit_bytes:
                remain = limit_bytes - used
                if remain <= 0:
                    break
                seg = seg_b[:remain].decode("utf-8", errors="ignore")
                seg_b = seg.encode("utf-8")
            chunks.append(seg)
            used += len(seg_b)

        return "\n".join(chunks)

    def _set_generate_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.gen_button.configure(state=state)
        self.reward_button.configure(state=state)
        self.punish_button.configure(state=state)

    # ---------- RL helpers ----------
    def build_decisions_from_text(self, txt: str) -> List[Tuple[str, str]]:
        words = txt.lower().split()
        decisions: List[Tuple[str, str]] = []
        for i in range(1, len(words)):
            ctx = " ".join(words[max(0, i-5):i])
            act = words[i]
            decisions.append((ctx, act))
        return decisions

    def get_mode_params(self):
        mode = self.mode_var.get()
        if mode == "Uniform only":
            return dict(stack_state=1.0, chirality_strength=0.0, swap_frac=0.0, use_rl=False)
        if mode == "RL only":
            return dict(stack_state=1.0, chirality_strength=0.0, swap_frac=0.2, use_rl=True)
        if mode == "Template only":
            return dict(stack_state=1.0, chirality_strength=1.0, swap_frac=0.35, use_rl=False)
        if mode == "Template+RL":
            return dict(stack_state=1.0, chirality_strength=1.0, swap_frac=0.35, use_rl=True)
        if mode == "Chaotic":
            return dict(stack_state=1.0, chirality_strength=0.7, swap_frac=0.95, use_rl=True)
        return dict(stack_state=1.0, chirality_strength=1.0, swap_frac=0.35, use_rl=True)

    # ---------- Actions ----------
    def on_generate(self):
        if self.llm is None:
            self.status.config(text="No corpus loaded; select File, HF dataset, or Generic.")
            return
        start = self.start_var.get().strip() or None
        beta = float(self.beta_var.get())
        params = self.get_mode_params()
        max_len = int(self.out_len_var.get())
        text, decisions = self.llm.generate_text_rl(
            start_word=start,
            max_words=max_len,
            stack_state=params["stack_state"],
            chirality_strength=params["chirality_strength"],
            swap_frac=params["swap_frac"],
            rl_memory=self.memory if params["use_rl"] else None,
            rl_beta=beta if params["use_rl"] else 0.0,
        )
        self.last_decisions = decisions
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.status.config(
            text=f"Generated {len(text.split())} tokens (target {max_len}) | β={beta:.2f} | Mode={self.mode_var.get()} | {self.current_text_source}"
        )

    def on_reward(self):
        if self.llm is None:
            self.status.config(text="No corpus loaded.")
            return
        if self.use_edited_var.get():
            text_now = self.text.get("1.0", "end-1c")
            decisions = self.build_decisions_from_text(text_now)
        else:
            decisions = self.last_decisions
        if not decisions:
            self.status.config(text="Nothing to reward yet.")
            return
        self.memory.update_batch(decisions, reward=+1.0)
        self.status.config(text="Rewarded current sample (+1).")

    def on_punish(self):
        if self.llm is None:
            self.status.config(text="No corpus loaded.")
            return
        if self.use_edited_var.get():
            text_now = self.text.get("1.0", "end-1c")
            decisions = self.build_decisions_from_text(text_now)
        else:
            decisions = self.last_decisions
        if not decisions:
            self.status.config(text="Nothing to punish yet.")
            return
        self.memory.update_batch(decisions, reward=-1.0)
        self.status.config(text="Punished current sample (-1).")

    def on_close(self):
        try:
            self.memory.close()
        finally:
            self.destroy()


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    app = RLApp()
    if app.winfo_exists():
        app.mainloop()

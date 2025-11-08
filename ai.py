import math
import random
import hashlib
from functools import singledispatch
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

# GUI + persistence
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sqlite3


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
    def __init__(self, filename: str):
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
# Chiral stacked 2D geometry with vsplit (optional visualizer)
# ================================================================

def wrap_angle_pi(theta: float) -> float:
    theta = np.fmod(theta + np.pi, 2 * np.pi)
    if theta <= 0:
        theta += 2 * np.pi
    return theta - np.pi

def create_2d_layer(num_points=100, size=1.0, layer_id=0):
    side = int(np.sqrt(num_points))
    x = np.linspace(-size/2, size/2, side)
    y = np.linspace(-size/2, size/2, side)
    X, Y = np.meshgrid(x, y)
    if side > 1:
        Y[::2] += (y[1] - y[0]) / 2
    points = np.column_stack((X.ravel()[:num_points], Y.ravel()[:num_points]))
    theta = np.deg2rad(layer_id * 30)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = points @ rot_matrix
    return points

def create_stacked_chiral_layers_combined(num_layers=8, twist_angle=15.0, z_spacing=0.1):
    all_layers = []
    points_per_layer = None
    for i in range(num_layers):
        layer_points = create_2d_layer(num_points=50, size=1.0, layer_id=i)
        if points_per_layer is None:
            points_per_layer = len(layer_points)
        chiral_twist_raw = i * np.deg2rad(twist_angle)
        chiral_twist = wrap_angle_pi(chiral_twist_raw)
        rot_matrix = np.array([
            [np.cos(chiral_twist), -np.sin(chiral_twist)],
            [np.sin(chiral_twist),  np.cos(chiral_twist)],
        ])
        twisted_points = layer_points @ rot_matrix
        z = np.full((len(twisted_points), 1), i * z_spacing)
        stacked_points = np.hstack((twisted_points, z))
        all_layers.append(stacked_points)
        for j in range(num_layers):
            layer_points = create_2d_layer(num_points=50, size=1.0, layer_id=i)
            chiral_twist_raw = j * np.deg2rad(twist_angle)
            chiral_twist = wrap_angle_pi(chiral_twist_raw)
            rot_matrix = np.array([
                [np.cos(chiral_twist), -np.sin(chiral_twist)],
                [np.sin(chiral_twist),  np.cos(chiral_twist)],
            ])
            twisted_points = layer_points @ rot_matrix
            z = np.full((len(twisted_points), 1), i * z_spacing)
            stacked_points = np.hstack((twisted_points, z))
            all_layers.append(stacked_points)
    combined = np.concatenate(all_layers, axis=0)
    return combined, points_per_layer

def split_layers_with_vsplit(combined: np.ndarray, points_per_layer: int) -> List[np.ndarray]:
    n_total = combined.shape[0]
    if n_total % points_per_layer != 0:
        raise ValueError("Total rows not divisible by points_per_layer for equal vsplit.")
    n_layers = n_total // points_per_layer
    return list(np.vsplit(combined, n_layers))

def interpolate_chirality(base_layers: List[np.ndarray], chiral_layers: List[np.ndarray], t: float):
    smooth_t = t * t * (3 - 2 * t)
    interpolated_layers = []
    for i in range(len(base_layers)):
        base_layer = base_layers[i]
        chiral_layer = chiral_layers[i]
        interp_xy = (1 - smooth_t) * base_layer[:, :2] + smooth_t * chiral_layer[:, :2]
        interp_layer = np.hstack((interp_xy, base_layer[:, 2:]))
        interpolated_layers.append(interp_layer)
        for j in range(len(base_layers)):
            base_layer = base_layers[i]
            chiral_layer = chiral_layers[j]
            interp_xy = (1 - smooth_t) * base_layer[:, :2] + smooth_t * chiral_layer[:, :2]
            interp_layer = np.hstack((interp_xy, base_layer[:, 2:]))
            interpolated_layers.append(interp_layer)
    return interpolated_layers * 8

def visualize_stacked_layers(layers: List[np.ndarray], ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for layer in layers:
        ax.scatter(layer[:, 0], layer[:, 1], layer[:, 2], s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# ================================================================
# Tkinter GUI with editable text reinforcement (persistent RL)
# ================================================================

class RLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OrigamiLLM RL GUI")
        self.geometry("1000x720")

        self.filename = filedialog.askopenfilename(
            title="Select corpus file (.txt)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not self.filename:
            messagebox.showerror("Error", "No corpus file selected.")
            self.destroy()
            return

        # Models and memory
        self.llm = OrigamiLLM(self.filename)
        self.memory = RLMemory("rl_cache.sqlite")

        # UI
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Start word:").pack(side="left")

        # Editable, focused Entry with variable and Return binding
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(top, width=24, textvariable=self.start_var)
        self.start_entry.state(["!disabled", "!readonly"])
        self.start_entry.configure(state="normal")
        self.start_entry.pack(side="left", padx=6)
        self.start_entry.focus_set()
        self.start_entry.bind("<Return>", lambda e: self.on_generate())

        self.gen_button = ttk.Button(top, text="Generate", command=self.on_generate)
        self.gen_button.pack(side="left", padx=6)

        self.reward_button = ttk.Button(top, text="Reward (+)", command=self.on_reward)
        self.reward_button.pack(side="left", padx=6)
        self.punish_button = ttk.Button(top, text="Punish (-)", command=self.on_punish)
        self.punish_button.pack(side="left", padx=6)

        # Toggle: learn from edited text vs last generated decisions
        self.use_edited_var = tk.BooleanVar(value=True)
        self.use_edited_chk = ttk.Checkbutton(
            top, text="Train on edited text", variable=self.use_edited_var
        )
        self.use_edited_chk.pack(side="left", padx=8)

        # Slider with DoubleVar, snapping step, and live label
        ttk.Label(top, text="RL β:").pack(side="left", padx=(12, 2))
        self.beta_var = tk.DoubleVar(value=2.0)
        self.beta_scale = ttk.Scale(
            top,
            from_=0.0,
            to=6.0,
            orient="horizontal",
            length=240,
            variable=self.beta_var,
            command=lambda v: self.on_beta_change(v),
        )
        self.beta_scale.pack(side="left", padx=6)
        self.beta_value_lbl = ttk.Label(top, text=f"{self.beta_var.get():.2f}")
        self.beta_value_lbl.pack(side="left", padx=4)

        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.text = tk.Text(text_frame, wrap="word")
        self.text.pack(fill="both", expand=True)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=8)
        self.status = ttk.Label(bottom, text="Ready.")
        self.status.pack(side="left")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.last_decisions: List[Tuple[str, str]] = []

    def on_beta_change(self, value: str):
        # Snap ttk.Scale to 0.05 steps and update label
        try:
            v = float(value)
        except Exception:
            return
        step = 0.05
        snapped = round(v / step) * step
        if abs(snapped - self.beta_var.get()) > 1e-9:
            self.beta_var.set(snapped)
        self.beta_value_lbl.config(text=f"{self.beta_var.get():.2f}")

    def build_decisions_from_text(self, txt: str) -> List[Tuple[str, str]]:
        """
        Reconstruct (context, action) from edited text, using up to 5-token history
        as context and the current token as the action, matching generation-time context.
        """
        words = txt.lower().split()
        decisions: List[Tuple[str, str]] = []
        for i in range(1, len(words)):
            ctx = " ".join(words[max(0, i-5):i])
            act = words[i]
            decisions.append((ctx, act))
        return decisions

    def on_generate(self):
        start = self.start_var.get().strip() or None
        beta = float(self.beta_var.get())
        text, decisions = self.llm.generate_text_rl(
            start_word=start,
            max_words=300,
            stack_state=1.0,
            chirality_strength=1.0,
            swap_frac=0.5,
            rl_memory=self.memory,
            rl_beta=beta,
        )
        self.last_decisions = decisions
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.status.config(text=f"Generated {len(text.split())} tokens with β={beta:.2f}.")

    def on_reward(self):
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
# Demo / CLI
# ================================================================

if __name__ == "__main__":
    app = RLApp()
    if app.winfo_exists():
        app.mainloop()


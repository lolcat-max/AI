import math
import random
import hashlib
from functools import singledispatch
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


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
# N-grams and Modulus Avalanche Utilities
# ================================================================

def build_ngrams(corpus_sentences: List[str]) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], List[str]]]:
    bigrams: Dict[str, List[str]] = {}
    trigrams: Dict[Tuple[str, str], List[str]] = {}
    for sentence in corpus_sentences:
        words = sentence.split()
        if len(words) < 3:
            continue
        for i in range(len(words) - 1):
            w0, w1 = words[i], words[i + 1]
            bigrams.setdefault(w0, []).append(w1)
        for i in range(len(words) - 2):
            key = (words[i], words[i + 1])
            trigrams.setdefault(key, []).append(words[i + 2])
    return bigrams, trigrams

def avalanche_list_mod(lst: List[str], rounds: int, m: int) -> List[str]:
    # Duplicate entries by cycling bucket ((r) % m) each round to avalanche list mass in phases
    out = list(lst)
    n = len(out)
    for r in range(rounds):
        bucket = r % max(m, 1)
        extra = [out[i] for i in range(n) if (i % m) == bucket]
        out.extend(extra)
        n = len(out)
    return out

def avalanche_ngrams_mod(bigrams: Dict[str, List[str]],
                         trigrams: Dict[Tuple[str, str], List[str]],
                         rounds: int,
                         m: int):
    for w in list(bigrams.keys()):
        bigrams[w] = avalanche_list_mod(bigrams[w], rounds=rounds, m=m)
    for k in list(trigrams.keys()):
        trigrams[k] = avalanche_list_mod(trigrams[k], rounds=rounds, m=m)

def avalanche_probs_mod(probs: np.ndarray, m: int, rounds: int, eps: float = 1e-12) -> np.ndarray:
    # Repeatedly smooth probabilities by modulus buckets, then renormalize to keep a valid p
    p = probs.astype(float, copy=True)
    p = np.maximum(p, eps)
    total = p.sum()
    p = p / total if total > 0 else np.ones_like(p) / max(1, p.size)
    n = p.size
    for r in range(rounds):
        bucket_sums = np.zeros(m, dtype=float)
        bucket_counts = np.zeros(m, dtype=float)
        for i in range(n):
            b = i % m
            bucket_sums[b] += p[i]
            bucket_counts[b] += 1.0
        bucket_avg = bucket_sums / np.maximum(bucket_counts, 1.0)
        b_avg_vec = np.fromiter((bucket_avg[i % m] for i in range(n)), dtype=float, count=n)
        mix = 0.5 + 0.5 * (r + 1) / max(rounds, 1)  # ramp mixing across rounds
        p = (1.0 - mix) * p + mix * b_avg_vec
        p = np.maximum(p, eps)
        s = p.sum()
        p = p / s if s > 0 else np.ones_like(p) / max(1, p.size)
    return p


# ================================================================
# Single-function text generator using modulus avalanche
# ================================================================

def generate_text_once_mod5(
    filename: str,
    start_word: Optional[str] = None,
    max_words: int = 200,
    stack_state: float = 1.0,
    chirality_strength: float = 1.0,
    swap_frac: float = 0.35,
    avalanche_rounds: int = 3,
    mod: int = 5,
    use_sublimate: bool = False,
    rng_seed: Optional[int] = None,
) -> str:
    rng = np.random.default_rng(rng_seed)

    # Read and prepare corpus
    with open(filename, encoding="utf-8") as f:
        text = f.read().lower()
    corpus = [s.strip() for s in text.split(".") if s.strip()]

    # Build n-grams
    bigrams, trigrams = build_ngrams(corpus)

    # Avalanche dataset (lists) by modulus buckets, repeatedly
    avalanche_ngrams_mod(bigrams, trigrams, rounds=avalanche_rounds, m=mod)

    # Templates
    tmpl = EmbeddedTemplateMatcher()
    tmpl.build_templates_from_corpus(corpus, num_templates=50)

    if not bigrams:
        return ""

    # Seed
    if start_word is None or start_word not in bigrams:
        start_word = rng.choice(list(bigrams.keys()))

    result = [start_word]
    current = start_word

    for _ in range(max_words - 1):
        use_model = rng.random() < stack_state
        candidates: List[str] = []

        if use_model:
            if len(result) >= 2:
                tri_key = (result[-2], result[-1])
                if tri_key in trigrams:
                    candidates = trigrams[tri_key]
            if not candidates and current in bigrams:
                candidates = bigrams[current]

        if not candidates and bigrams:
            vocab = list(bigrams.keys())
            k = min(5, len(vocab))
            candidates = list(rng.choice(vocab, size=k, replace=False))

        # Base uniform probs
        probs = np.ones(len(candidates), dtype=float)
        probs /= probs.sum() if probs.sum() > 0 else 1.0

        # Template-driven boost (chirality)
        context_text = " ".join(result[-5:])
        best_t = tmpl.find_best_template(context_text)
        if chirality_strength > 0 and best_t is not None and candidates:
            template_text, _ = best_t
            t_words = template_text.split()
            try:
                idx = t_words.index(result[-1])
                if idx < len(t_words) - 1:
                    tnext = t_words[idx + 1]
                    if tnext in candidates:
                        bidx = candidates.index(tnext)
                        twist = math.exp(chirality_strength)
                        probs[bidx] += 5.0 * chirality_strength * twist
            except ValueError:
                pass

        # Optional deterministic “sublimation” swap before modulus avalanche
        if use_sublimate and len(candidates) > 1:
            candidates, probs = sublimate_candidates_and_probs(
                candidates, probs, context=context_text, swap_frac=swap_frac
            )

        # Normalize then avalanche probabilities by modulus buckets repeatedly
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones_like(probs) / max(1, probs.size)
        probs = avalanche_probs_mod(probs, m=mod, rounds=avalanche_rounds)

        # Final choice with valid p
        next_word = rng.choice(candidates, p=probs)
        result.append(next_word)
        current = next_word

    return " ".join(result)


# ================================================================
# Chiral stacked 2D geometry with vsplit (optional visualization)
# ================================================================

def wrap_angle_pi(theta: float) -> float:
    # Wrap angle to (-pi, pi] using C-like remainder via np.fmod
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
# Demo / CLI
# ================================================================

if __name__ == "__main__":
    print("="*70)
    print("CHIRAL STACKED 2D LLM - SINGLE-FUNCTION TEXT GENERATION (modulus avalanche)")
    print("="*70)

    filename = input("Filename: ").strip()
    while True:
        start = input("Start word (blank for auto): ").strip() or None

        text = generate_text_once_mod5(
            filename=filename,
            start_word=start,
            max_words=800,
            stack_state=1.0,
            chirality_strength=1.0,
            swap_frac=0.95,
            avalanche_rounds=8,
            mod=5,
            use_sublimate=False,
            rng_seed=None,
        )
        print("\n--- GENERATED TEXT ---")
        print(text)
        print("-" * 22 + "\n")

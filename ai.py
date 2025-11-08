import math
import random
import hashlib
from functools import singledispatch
from typing import List, Tuple, Optional

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
        self.word_vectors: dict[str, np.ndarray] = {}

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
# OrigamiLLM (n-gram + templates + dtype sublimate)
# ================================================================

class OrigamiLLM:
    def __init__(self):
        filename = input("Filename: ").strip()
        with open(filename, encoding="utf-8") as f:
            text = f.read()
        self.corpus = text.lower().split(".")

        self.bigrams: dict[str, List[str]] = {}
        self.trigrams: dict[Tuple[str, str], List[str]] = {}
        self.build_model()

        self.template_matcher = EmbeddedTemplateMatcher()
        self.template_matcher.build_templates_from_corpus(self.corpus)

    def build_model(self):
        for sentence in self.corpus:
            words = sentence.split()
            if len(words) < 3:
                continue
            for i in range(len(words) - 1):
                if words[i] not in self.bigrams:
                    self.bigrams[words[i]] = []
                self.bigrams[words[i]].append(words[i + 1])
            for i in range(len(words) - 2):
                key = (words[i], words[i + 1])
                if key not in self.trigrams:
                    self.trigrams[key] = []
                self.trigrams[key].append(words[i + 2])

    def generate_text(
        self,
        start_word: Optional[str] = None,
        max_words: int = 200,
        stack_state: float = 1.0,
        chirality_strength: float = 1.0,
        swap_frac: float = 0.35,
    ) -> str:
        if not self.bigrams:
            return ""
        if start_word is None or start_word not in self.bigrams:
            start_word = random.choice(list(self.bigrams.keys()))

        result = [start_word]
        current_word = start_word

        for _ in range(max_words - 1):
            use_model = random.random() < stack_state
            candidates: List[str] = []

            if use_model:
                trigram_key = tuple(result[-2:]) if len(result) >= 2 else None
                if trigram_key and trigram_key in self.trigrams:
                    candidates = self.trigrams[trigram_key]
                elif current_word in self.bigrams:
                    candidates = self.bigrams[current_word]

            if not candidates and self.bigrams:
                candidates = [random.choice(list(self.bigrams.keys())) for _ in range(5)]

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

            total = float(probs.sum())
            probs = probs / total if total > 0 else np.ones_like(probs) / max(1, len(probs))

            candidates, probs = sublimate_candidates_and_probs(
                candidates, probs, context=context, swap_frac=swap_frac
            )

            if candidates:
                next_word = np.random.choice(candidates, p=probs)
            else:
                next_word = random.choice(list(self.bigrams.keys())) if self.bigrams else "end"

            result.append(next_word)
            current_word = next_word

        generated = " ".join(result)
        return generated


# ================================================================
# Chiral stacked 2D geometry with vsplit
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
    # Build one combined (N_total, 3) array; each layer contributes the same number of rows
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
    # Concatenate once; do not use vstack here, as splitting is requested via vsplit later
    combined = np.concatenate(all_layers, axis=0)
    return combined, points_per_layer

def split_layers_with_vsplit(combined: np.ndarray, points_per_layer: int) -> List[np.ndarray]:
    # Recover equal-sized layers row-wise using np.vsplit
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
    return interpolated_layers

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
    print("CHIRAL STACKED 2D LLM - TEXT GENERATION DEMO (vsplit edition)")
    print("="*70)

    llm = OrigamiLLM()

    base_twist = 0.0
    chiral_twist = 15.0

    # Get combined arrays
    base_combined, ppl = create_stacked_chiral_layers_combined(num_layers=4, twist_angle=base_twist)
    chiral_combined, _ = create_stacked_chiral_layers_combined(num_layers=4, twist_angle=chiral_twist)

    # Recover per-layer arrays using vsplit
    base_layers = split_layers_with_vsplit(base_combined, ppl)
    chiral_layers = split_layers_with_vsplit(chiral_combined, ppl)

    print("\n" + "="*70)
    print("INTERACTIVE TEXT GENERATION")
    print("="*70)

    while True:
        try:
            start_word = input("USER (start word): ").strip()
        except EOFError:
            break
        if not start_word:
            continue

        print("\n--- GENERATED TEXT ---")
        text = llm.generate_text(
            start_word=start_word,
            max_words=800,
            stack_state=1.0,
            chirality_strength=1.0,
            swap_frac=0.35,
        )
        print(text)
        print("-" * 22 + "\n")
        # To visualize, uncomment:
        # visualize_stacked_layers(interpolate_chirality(base_layers, chiral_layers, 1.0))

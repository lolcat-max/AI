import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import threading
import os
import re
from dataclasses import dataclass

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            default = args[0]
            super().__init__(default)
        else:
            super().__init__(int)

# =====================================================================
# DYADIC OPERATIONS FOUNDATION (Computer Science Primitives)
# =====================================================================
class DyadicOperations:
    """
    Fundamental two-element operations from computer science, 
    adapted for Vector Symbolic Architectures [web:59][web:65].
    """
    
    @staticmethod
    def AND(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Logical AND: element-wise minimum (for normalized vectors)."""
        return np.minimum(a, b)
    
    @staticmethod
    def OR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Logical OR: element-wise maximum (for normalized vectors)."""
        return np.maximum(a, b)
    
    @staticmethod
    def XOR(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Logical XOR: element-wise difference modulated."""
        return np.abs(a - b)
    
    @staticmethod
    def BIND(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        VSA Binding: reversible operation that distributes over bundling.
        Uses circular convolution in Fourier space (FHRR VSA) [web:63][web:67].
        """
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        return np.real(np.fft.ifft(fft_a * fft_b))
    
    @staticmethod
    def UNBIND(h: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Inverse of BIND: retrieve b from bind(a,b) given a."""
        fft_h = np.fft.fft(h)
        fft_a = np.fft.fft(a)
        return np.real(np.fft.ifft(fft_h / (fft_a + 1e-9)))
    
    @staticmethod
    def BUNDLE(vectors: List[np.ndarray]) -> np.ndarray:
        """
        VSA Bundling: superposition operation (like OR/ADD).
        Creates sum that preserves similarity to inputs [web:60][web:66].
        """
        return np.sum(vectors, axis=0)

# =====================================================================
# 2D POLARIZATION VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_polarized_vector(self, normalize: bool = True) -> np.ndarray:
        """Generate vector from 2D polar coordinates (polarization states)."""
        dim_2d = self.dimensions // 2
        theta = np.random.uniform(0, 2 * np.pi, dim_2d)
        r = np.ones(dim_2d)
        x_channel = r * np.cos(theta)
        y_channel = r * np.sin(theta)
        vec = np.stack([x_channel, y_channel], axis=0).reshape(-1)
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """VSA binding using dyadic BIND operation."""
        return DyadicOperations.BIND(vec_a, vec_b)
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """VSA bundling using dyadic BUNDLE operation."""
        return DyadicOperations.BUNDLE(vectors)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9)

    def add_to_codebook(self, symbol: str) -> np.ndarray:
        with self.lock:
            if symbol not in self.codebook:
                self.codebook[symbol] = self.create_polarized_vector()
            return self.codebook[symbol]

    def save_codebook(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
        print(f"✓ Polarized codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        print(f"✓ Polarized codebook loaded from {filepath}")

# =====================================================================
# POLARIZATION TRANSITION ENCODER (N-GRAM COUNTS + DYADIC OPERATIONS)
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        self.unigram_counts = IntDefaultDict()
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.bigram_transitions = defaultdict(IntDefaultDict)
        self.trigram_transitions = defaultdict(IntDefaultDict)
        self.lock = threading.Lock()

    def encode_unigram(self, token: str):
        with self.lock:
            self.unigram_counts[token] += 1

    def encode_bigram(self, token1: str, token2: str):
        with self.lock:
            self.bigram_transitions[token1][token2] += 1
            key = (token1, token2)
            if key not in self.bigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                # Use dyadic BIND operation for bigram encoding
                self.bigram_vectors[key] = self.vsa.bind(vec1, vec2)

    def encode_trigram(self, token1: str, token2: str, token3: str):
        with self.lock:
            self.trigram_transitions[(token1, token2)][token3] += 1
            key = (token1, token2, token3)
            if key not in self.trigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                vec3 = self.vsa.add_to_codebook(token3)
                # Use dyadic BIND operations for trigram encoding
                bound12 = self.vsa.bind(vec1, vec2)
                self.trigram_vectors[key] = self.vsa.bind(bound12, vec3)

    def _process_sequence_batch(self, sequences: List[List[str]]):
        for sequence in sequences:
            for token in sequence:
                self.encode_unigram(token)
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning polarized transitions from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self._process_sequence_batch, batches), 
                     total=len(batches), desc="Polarized batches", ncols=80))
        print(f"  ✓ Learned {len(self.unigram_counts)} unigram counts")
        print(f"  ✓ Learned {sum(len(v) for v in self.bigram_transitions.values())} bigram transitions")
        print(f"  ✓ Learned {sum(len(v) for v in self.trigram_transitions.values())} trigram transitions")

    def get_unigram_probabilities(self) -> Dict[str, float]:
        total = sum(self.unigram_counts.values())
        if total == 0:
            return {}
        return {token: count / total for token, count in self.unigram_counts.items()}

    def get_bigram_probabilities(self, last_token: str) -> Optional[Dict[str, float]]:
        if last_token not in self.bigram_transitions: 
            return None
        candidates = self.bigram_transitions[last_token]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def get_trigram_probabilities(self, last_two_tokens: Tuple[str, str]) -> Optional[Dict[str, float]]:
        if last_two_tokens not in self.trigram_transitions: 
            return None
        candidates = self.trigram_transitions[last_two_tokens]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for name, data in [("unigram_counts", self.unigram_counts),
                          ("bigram_transitions", self.bigram_transitions),
                          ("trigram_transitions", self.trigram_transitions),
                          ("bigram_vectors", self.bigram_vectors),
                          ("trigram_vectors", self.trigram_vectors)]:
            with open(os.path.join(directory, f"{name}.pkl"), 'wb') as f:
                pickle.dump(data, f)
        print(f"✓ Polarized transition model saved to {directory}")

    def load_model(self, directory: str):
        for name in ["unigram_counts", "bigram_transitions", "trigram_transitions", 
                     "bigram_vectors", "trigram_vectors"]:
            with open(os.path.join(directory, f"{name}.pkl"), 'rb') as f:
                setattr(self, name, pickle.load(f))
        print(f"✓ Polarized transition model loaded from {directory}")

# =====================================================================
# ONLINE REINFORCEMENT LEARNING FEEDBACK BUFFER
# =====================================================================
class FeedbackBuffer:
    """Real-time human feedback for online RL [web:51][web:55][web:59]."""
    def __init__(self, vsa: VectorSymbolicArchitecture, buffer_size: int = 50):
        self.vsa = vsa
        self.buffer_size = buffer_size
        self.positive_tokens = deque(maxlen=buffer_size)  # "I like that"
        self.negative_tokens = deque(maxlen=buffer_size)  # "bad", "no"
        self.token_rewards = defaultdict(float)  # Running reward per token
        self.lock = threading.Lock()
    
    def add_positive_feedback(self, tokens: List[str], reward: float = 1.0):
        """Add positive feedback for recently generated tokens [web:55][web:59]."""
        with self.lock:
            for token in tokens:
                self.positive_tokens.append(token)
                self.token_rewards[token] += reward
                print(f"  [✓] Rewarded: {token} (+{reward:.2f})")
    
    def add_negative_feedback(self, tokens: List[str], penalty: float = -0.5):
        """Add negative feedback for recently generated tokens [web:55][web:59]."""
        with self.lock:
            for token in tokens:
                self.negative_tokens.append(token)
                self.token_rewards[token] += penalty
                print(f"  [✗] Penalized: {token} ({penalty:.2f})")
    
    def get_token_reward(self, token: str) -> float:
        """Get accumulated reward for a token [web:51][web:55]."""
        return self.token_rewards.get(token, 0.0)
    
    def get_similar_token_reward(self, token: str, top_k: int = 5) -> float:
        """Get reward from semantically similar tokens [web:55][web:59]."""
        if token not in self.vsa.codebook:
            return 0.0
        
        token_vec = self.vsa.codebook[token]
        similar_rewards = []
        
        # Find similar rewarded/penalized tokens
        for rewarded_token, reward in self.token_rewards.items():
            if rewarded_token in self.vsa.codebook:
                sim = self.vsa.similarity(token_vec, self.vsa.codebook[rewarded_token])
                if sim > 0.5:  # Similarity threshold
                    similar_rewards.append(reward * sim)
        
        if similar_rewards:
            return np.mean(sorted(similar_rewards, reverse=True)[:top_k])
        return 0.0

# =====================================================================
# SYMPLECTIC ISOHEDRAL TILER
# =====================================================================
@dataclass
class TileInstance:
    """One tile in the isohedral tiling."""
    token: str                     # Which symbol this tile corresponds to
    grid_pos: Tuple[int, int]      # (i,j) grid coordinates
    vertices: np.ndarray           # (m, 2) array of xy vertices in R^2
    angle: float  # symplectic rotation angle in phase space

class SymplecticIsohedralTiler:
    """
    Build an isohedral tiling from VSA polarization indices, using
    2D symplectic (area-preserving) transforms of a single prototile.

    - Underlying phase space: R^2 with ω = dq ∧ dp.
    - Each category / token chooses a symplectic matrix A(θ) (a rotation).
    - Tiling is a regular Z^2 translation grid, so all tiles are
      translates/rotates of one base polygon (isohedral).
    """
    def __init__(
        self,
        vsa: VectorSymbolicArchitecture,
        base_tile: Optional[np.ndarray] = None,
        grid_size: Tuple[int, int] = (16, 16),
        cell_size: float = 1.0,
    ):
        """
        vsa       : your VectorSymbolicArchitecture instance (already populated)
        base_tile : (m,2) polygon in local coords; if None, use unit square
        grid_size : tiling grid (rows, cols)
        cell_size : spacing between tile origins
        """
        self.vsa = vsa
        self.grid_size = grid_size
        self.cell_size = cell_size

        if base_tile is None:
            # Unit square centered at origin (area 1, simple symplectic cell)
            self.base_tile = np.array([
                [-0.5, -0.5],
                [ 0.5, -0.5],
                [ 0.5,  0.5],
                [-0.5,  0.5],
            ], dtype=float)
        else:
            self.base_tile = np.asarray(base_tile, dtype=float)

        # Cache angles per token so generation is deterministic once built
        self._token_angles: Dict[str, float] = {}

    @staticmethod
    def symplectic_rotation(theta: float) -> np.ndarray:
        """
        2D rotation matrix with det=1. In 2D, any orientation-preserving
        isometry is symplectic w.r.t. ω = dq ∧ dp [web:25][web:31].
        """
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]], dtype=float)
        # det(R) = 1, so area is preserved [web:34]
        return R

    def _token_angle_from_index(self, token_vec: np.ndarray) -> float:
        """
        Use the FIRST (q,p) pair (index 0,1) of the polarized vector
        to define an angle in phase space. You can generalize this
        to averaging over all pairs if you want.
        """
        if token_vec.shape[0] < 2:
            return 0.0
        q = token_vec[0]
        p = token_vec[1]
        theta = float(np.arctan2(p, q))  # in [-π, π]
        return theta

    def _get_token_angle(self, token: str) -> float:
        if token not in self._token_angles:
            vec = self.vsa.add_to_codebook(token)
            theta = self._token_angle_from_index(vec)
            self._token_angles[token] = theta
        return self._token_angles[token]

    def _tile_origin(self, i: int, j: int) -> np.ndarray:
        """
        Origin (translation) for tile at grid cell (i,j).
        """
        return np.array([j * self.cell_size, i * self.cell_size], dtype=float)

    def build_tiling_from_tokens(
        self,
        tokens: List[str],
        wrap: bool = True
    ) -> List[TileInstance]:
        """
        Build an isohedral tiling grid where each cell is one copy of the
        same base polygon, transformed by a symplectic rotation derived
        from the token's polarization angle [web:20][web:22].
        """
        if not tokens:
            raise ValueError("Need at least one token to build a tiling.")

        rows, cols = self.grid_size
        n_tokens = len(tokens)
        tiles: List[TileInstance] = []

        for i in range(rows):
            for j in range(cols):
                if wrap:
                    idx = (i * cols + j) % n_tokens
                else:
                    idx = min(i * cols + j, n_tokens - 1)

                token = tokens[idx]
                theta = self._get_token_angle(token)
                R = self.symplectic_rotation(theta)      # area-preserving
                origin = self._tile_origin(i, j)         # translation

                # Apply symplectic transform + translation
                verts = (self.base_tile @ R.T) + origin  # (m,2)

                tiles.append(TileInstance(
                    token=token,
                    grid_pos=(i, j),
                    vertices=verts,
                    angle=theta
                ))

        return tiles

    def render_svg(self, tiles: List[TileInstance], filepath: str, scale: float = 30.0):
        """Render tiling as SVG with token labels."""
        rows, cols = self.grid_size
        max_x = cols * self.cell_size
        max_y = rows * self.cell_size
        
        width = int(max_x * scale) + 100
        height = int(max_y * scale) + 100
        
        svg_lines = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<style>text { font-size: 10px; text-anchor: middle; }</style>',
            '<rect width="100%" height="100%" fill="white"/>'
        ]
        
        offset_x = 50
        offset_y = 50
        
        for tile_idx, tile in enumerate(tiles):
            centroid = tile.vertices.mean(axis=0)
            cx = offset_x + centroid[0] * scale
            cy = offset_y + centroid[1] * scale
            
            # Color by rotation angle (symplectic phase)
            color = f"hsl({int(np.degrees(tile.angle)) % 360}, 70%, 60%)"
            
            # Draw tile polygon
            pts = [(offset_x + v[0]*scale, offset_y + v[1]*scale) for v in tile.vertices]
            pts_str = " ".join([f"{x},{y}" for x, y in pts])
            svg_lines.append(f'<polygon points="{pts_str}" fill="{color}" stroke="black" stroke-width="0.5"/>')
            
            # Add token label
            svg_lines.append(f'<text x="{cx}" y="{cy}" fill="white">{tile.token}</text>')
        
        svg_lines.append('</svg>')
        
        with open(filepath, 'w') as f:
            f.write("\n".join(svg_lines))
        print(f"✓ Tiling saved to {filepath}")

# =====================================================================
# COMBINED RL + TILING GENERATOR WITH DYADIC OPERATIONS
# =====================================================================
class CombinedRLTilingGenerator:
    """Generate text and simultaneously build/update isohedral tiling."""
    
    def __init__(self, vsa: VectorSymbolicArchitecture, transition_encoder: TransitionEncoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.feedback_buffer = FeedbackBuffer(vsa, buffer_size=100)
        self.generation_buffer = deque(maxlen=20)
        self.tiler = SymplecticIsohedralTiler(vsa, grid_size=(8, 8), cell_size=1.0)
        self.semantic_categories = self._build_semantic_categories()
        self.generated_sequence: List[str] = []
        self.tiling_tiles: List[TileInstance] = []
        # Demonstrate dyadic operations on example vectors
        self._demonstrate_dyadic_operations()
        
    def _demonstrate_dyadic_operations(self):
        """Show how dyadic operations work on polarized vectors."""
        print("\n[DYADIC OPERATIONS DEMONSTRATION]")
        vec1 = self.vsa.create_polarized_vector()
        vec2 = self.vsa.create_polarized_vector()
        
        # AND operation
        and_result = DyadicOperations.AND(vec1, vec2)
        print(f"  ✓ AND similarity: {self.vsa.similarity(vec1, and_result):.3f}")
        
        # OR operation
        or_result = DyadicOperations.OR(vec1, vec2)
        print(f"  ✓ OR similarity: {self.vsa.similarity(vec1, or_result):.3f}")
        
        # XOR operation
        xor_result = DyadicOperations.XOR(vec1, vec2)
        print(f"  ✓ XOR similarity: {self.vsa.similarity(vec1, xor_result):.3f}")
        
        # BIND/UNBIND demonstration
        bound = DyadicOperations.BIND(vec1, vec2)
        unbound = DyadicOperations.UNBIND(bound, vec1)
        print(f"  ✓ BIND/UNBIND recovery: {self.vsa.similarity(vec2, unbound):.3f}")
        
        # BUNDLE demonstration
        bundle = DyadicOperations.BUNDLE([vec1, vec2])
        print(f"  ✓ BUNDLE similarity to vec1: {self.vsa.similarity(vec1, bundle):.3f}")
        print("-"*80)
        
    def _build_semantic_categories(self) -> Dict[str, List[str]]:
        print("Building semantic category clusters...")
        categories = defaultdict(list)
        for token, vec in tqdm(self.vsa.codebook.items(), desc="Categorizing", ncols=80):
            if vec.shape[0] >= 2:
                x_channel = vec[0]
                y_channel = vec[1]
                angle = np.arctan2(y_channel, x_channel)
                category_id = int((angle + np.pi) / (np.pi / 4)) % 8
                categories[f"cat_{category_id}"].append(token)
        print(f"  ✓ Created {len(categories)} semantic categories")
        return dict(categories)
    
    def _get_incompatible_category(self, current_category: str) -> str:
        current_id = int(current_category.split("_")[1])
        opposite_id = (current_id + 4) % 8
        return f"cat_{opposite_id}"
    
    def _get_token_category(self, token: str) -> Optional[str]:
        for cat_id, tokens in self.semantic_categories.items():
            if token in tokens:
                return cat_id
        return None
    
    def _compute_semantic_plausibility(self, candidate: str, context: List[str], 
                                      context_window: int = 3) -> float:
        if candidate not in self.vsa.codebook:
            return 0.0
        
        candidate_vec = self.vsa.codebook[candidate]
        recent_context = context[-context_window:]
        similarities = []
        
        for ctx_token in recent_context:
            if ctx_token in self.vsa.codebook:
                ctx_vec = self.vsa.codebook[ctx_token]
                sim = self.vsa.similarity(candidate_vec, ctx_vec)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        return np.mean(similarities)
    
    def _get_ngram_plausibility(self, candidate: str, context: List[str]) -> float:
        plausibility = 0.1
        
        if len(context) >= 1:
            last_token = context[-1]
            if last_token in self.transition_encoder.bigram_transitions:
                bigram_probs = self.transition_encoder.get_bigram_probabilities(last_token)
                if bigram_probs and candidate in bigram_probs:
                    plausibility *= bigram_probs[candidate]
        
        if len(context) >= 2:
            last_two = tuple(context[-2:])
            if last_two in self.transition_encoder.trigram_transitions:
                trigram_probs = self.transition_encoder.get_trigram_probabilities(last_two)
                if trigram_probs and candidate in trigram_probs:
                    plausibility *= trigram_probs[candidate] * 2.0
        
        return plausibility
    
    def apply_feedback_to_probs(self, probs: Dict[str, float], rl_weight: float = 2.0) -> Dict[str, float]:
        adjusted_probs = {}
        
        for token, prob in probs.items():
            direct_reward = self.feedback_buffer.get_token_reward(token)
            similar_reward = self.feedback_buffer.get_similar_token_reward(token)
            total_reward = direct_reward + (similar_reward * 0.5)
            rl_boost = np.exp(total_reward * rl_weight)
            adjusted_probs[token] = prob * rl_boost
        
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {k: v/total for k, v in adjusted_probs.items()}
        
        return adjusted_probs
    
    def stream_generation_with_tiling(self, seed: List[str], max_tokens: int = 50, 
                                     temperature: float = 1.0, 
                                     error_rate: float = 0.0001,
                                     plausibility_weight: float = 0.9,
                                     rl_weight: float = 2.0):
        """Generate text and update tiling in real-time."""
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return
        
        self.generated_sequence = []
        
        for step in range(max_tokens):
            force_error = np.random.random() < error_rate
            
            if force_error and len(context) >= 1:
                last_token = context[-1]
                last_category = self._get_token_category(last_token)
                
                if last_category:
                    error_category = self._get_incompatible_category(last_category)
                    candidate_tokens = self.semantic_categories.get(error_category, [])
                    
                    if candidate_tokens:
                        plausibility_scores = {}
                        
                        for token in candidate_tokens:
                            vsa_plausibility = self._compute_semantic_plausibility(token, context)
                            ngram_plausibility = self._get_ngram_plausibility(token, context)
                            combined = (vsa_plausibility + ngram_plausibility) / 2.0
                            plausibility_scores[token] = combined
                        
                        probs = {}
                        total_counts = sum(self.transition_encoder.unigram_counts.values())
                        
                        for token in candidate_tokens:
                            base_prob = self.transition_encoder.unigram_counts.get(token, 1) / total_counts
                            plausibility = plausibility_scores.get(token, 0.0)
                            plausibility_boost = 1.0 + (plausibility * plausibility_weight * 10.0)
                            probs[token] = base_prob * plausibility_boost
                        
                        probs = self.apply_feedback_to_probs(probs, rl_weight=rl_weight)
                        
                        tokens = list(probs.keys())
                        prob_vals = np.array(list(probs.values()))
                        
                        if temperature > 0:
                            prob_vals = np.log(prob_vals + 1e-9) / temperature
                            prob_vals = np.exp(prob_vals)
                        prob_vals /= np.sum(prob_vals)
                        
                        next_token = np.random.choice(tokens, p=prob_vals)
                        self.generation_buffer.append(next_token)
                        self.generated_sequence.append(next_token)
                        yield next_token
                        context.append(next_token)
                        
                        # UPDATE TILING: rebuild with accumulated sequence
                        if len(self.generated_sequence) % 5 == 0:
                            self._update_tiling()
                        continue
            
            # Normal n-gram prediction with RL
            probs = None
            if len(context) >= 2:
                probs = self.transition_encoder.get_trigram_probabilities(tuple(context[-2:]))
            if probs is None and len(context) >= 1:
                probs = self.transition_encoder.get_bigram_probabilities(context[-1])
            if probs is None:
                probs = self.transition_encoder.get_unigram_probabilities()
            
            if not probs:
                next_token = np.random.choice(list(self.vsa.codebook.keys()))
            else:
                probs = self.apply_feedback_to_probs(probs, rl_weight=rl_weight)
                
                tokens = list(probs.keys())
                prob_vals = np.array(list(probs.values()))
                if temperature > 0:
                    prob_vals = np.log(prob_vals + 1e-9) / temperature
                    prob_vals = np.exp(prob_vals)
                prob_vals /= np.sum(prob_vals)
                next_token = np.random.choice(tokens, p=prob_vals)
            
            self.generation_buffer.append(next_token)
            self.generated_sequence.append(next_token)
            yield next_token
            context.append(next_token)
            
            # UPDATE TILING: rebuild with accumulated sequence
            if len(self.generated_sequence) % 5 == 0:
                self._update_tiling()
    
    def _update_tiling(self):
        """Rebuild tiling from current generated sequence."""
        if self.generated_sequence:
            try:
                self.tiling_tiles = self.tiler.build_tiling_from_tokens(
                    self.generated_sequence[-32:] if len(self.generated_sequence) > 32 
                    else self.generated_sequence
                )
            except:
                pass  # Silent fail if tiling update fails
    
    def render_current_tiling(self, filepath: str):
        """Render the current tiling to SVG."""
        if self.tiling_tiles:
            self.tiler.render_svg(self.tiling_tiles, filepath)
        else:
            print("No tiling to render. Generate some text first.")

# =====================================================================
# MAIN ENTRYPOINT WITH DYADIC OPERATIONS DEMONSTRATION
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("DYADIC OPERATIONS VSA: 2D POLARIZATION + RL + SYMPLECTIC TILING")
    print("="*80)
    print("Computer Science dyadic operations: AND, OR, XOR, BIND, BUNDLE")
    print("VSA requires two binary operations: reversible binding + distributive bundling")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=128)
    trans_encoder = TransitionEncoder(vsa)
    choice = input("[N]ew polarized model or [L]oad existing? ").strip().lower()

    if choice == "l":
        directory = input("Model directory: ").strip()
        vsa.load_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.load_model(directory)
    else:
        filename = input("Corpus filename: ")
        print("Loading polarized corpus...")
        with open(filename, encoding="utf-8") as f: 
            raw_text = f.read()
        sentences = raw_text.split(".")
        corpus = [s.split() for s in tqdm(sentences, desc="Tokenizing", ncols=80) if s.split()]
        print(f"Corpus: {len(corpus)} sequences")
        print("Learning Polarized Transitions (Multithreaded)")
        trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)
        print("Building polarized vocabulary...")
        for sentence in tqdm(corpus, desc="Polarized Vocab", ncols=80):
            for token in sentence: 
                vsa.add_to_codebook(token)
        print(f"  ✓ Polarized vocabulary: {len(vsa.codebook)} tokens")
        directory = input("Save polarized model to directory: ").strip()
        vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.save_model(directory)
    
    print("\n[RL + TILING GENERATION]")
    print("Commands: 'render', 'reset', 'quit'")
    print("Provide feedback: 'good', 'bad', 'excellent', 'terrible', etc.")

    gen = CombinedRLTilingGenerator(vsa, trans_encoder)
    
    while True:
        user_input = input("\nUSER: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.lower() == "render":
            gen.render_current_tiling("tiling_output.svg")
            print("✓ Rendered to tiling_output.svg")
            continue
        
        if user_input.lower() == "reset":
            gen.feedback_buffer = FeedbackBuffer(vsa, buffer_size=100)
            print("✓ Feedback reset")
            continue
        
        # FEEDBACK PATTERNS
        if re.search(r"\b(excellent|perfect|good|yes|great)\b", user_input.lower()):
            recent = list(gen.generation_buffer)[-5:]
            gen.feedback_buffer.add_positive_feedback(recent, reward=1.0)
            print("✓ Positive feedback applied!")
            continue
        
        if re.search(r"\b(bad|no|wrong|poor|terrible)\b", user_input.lower()):
            recent = list(gen.generation_buffer)[-5:]
            gen.feedback_buffer.add_negative_feedback(recent, penalty=-1.0)
            print("✗ Negative feedback applied!")
            continue
        
        # GENERATE + TILE
        print("AI: ", end='', flush=True)
        for token in gen.stream_generation_with_tiling(
            user_input.split(), 
            max_tokens=1000, 
            temperature=0.7,
            error_rate=0.0001,
            rl_weight=2.0
        ):
            print(token, end=' ', flush=True)
        print()
        
        # Auto-render every generation
        if gen.generated_sequence:
            gen.render_current_tiling(f"tiling_gen_{len(gen.generated_sequence)}.svg")
            print(f"  📊 Tiling saved (length: {len(gen.generated_sequence)})")

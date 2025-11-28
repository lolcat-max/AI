import numpy as np
from typing import List, Dict, Tuple, Optional, Generator  # FIXED: Added Generator
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import threading
import os
import re
from dataclasses import dataclass
KB_LEN = -1
# =====================================================================
# CORE DATA STRUCTURES
# =====================================================================
class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            super().__init__(args[0])
        else:
            super().__init__(int)

@dataclass
class TileInstance:
    token: str
    grid_pos: Tuple[int, int]
    vertices: np.ndarray
    angle: float

# =====================================================================
# VECTOR SYMBOLIC ARCHITECTURE KERNEL
# =====================================================================
class VectorSymbolicArchitecture:
    """Core VSA engine for high-dimensional vector operations."""
    
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_polarized_vector(self, normalize: bool = True) -> np.ndarray:
        dim_2d = self.dimensions // 2
        theta = np.random.uniform(0, 2 * np.pi, dim_2d)
        r = np.ones(dim_2d)
        x_channel = r * np.cos(theta)
        y_channel = r * np.sin(theta)
        vec = np.stack([x_channel, y_channel], axis=0).reshape(-1)
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def bind_polarized(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        dim = len(vec_a) // 2
        fft_a_swapped = np.ones_like(fft_a, dtype=complex)
        fft_a_swapped[:dim] = fft_b[dim:]
        fft_a_swapped[dim:] = fft_b[:dim]
        return np.real(np.fft.ifft(fft_a + fft_a_swapped))

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

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

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)

# =====================================================================
# TRANSITION ENCODER KERNEL
# =====================================================================
class TransitionEncoder:
    """N-gram transition learning with VSA binding."""
    
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
                self.bigram_vectors[key] = self.vsa.bind_polarized(vec1, vec2)

    def encode_trigram(self, token1: str, token2: str, token3: str):
        with self.lock:
            self.trigram_transitions[(token1, token2)][token3] += 1
            key = (token1, token2, token3)
            if key not in self.trigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                vec3 = self.vsa.add_to_codebook(token3)
                bound12 = self.vsa.bind_polarized(vec1, vec2)
                self.trigram_vectors[key] = self.vsa.bind_polarized(bound12, vec3)

    def _process_sequence_batch(self, sequences: List[List[str]]):
        for sequence in sequences:
            for token in sequence:
                self.encode_unigram(token)
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])
    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        """Learn n-gram transitions from corpus with parallel processing."""
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._process_sequence_batch, batches))
       
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

    def load_model(self, directory: str):
        for name in ["unigram_counts", "bigram_transitions", "trigram_transitions", 
                     "bigram_vectors", "trigram_vectors"]:
            with open(os.path.join(directory, f"{name}.pkl"), 'rb') as f:
                setattr(self, name, pickle.load(f))

# =====================================================================
# RL FEEDBACK BUFFER KERNEL (WITH 10 MODIFICATIONS)
# =====================================================================
class FeedbackBuffer:
    """Core RL feedback mechanism with non-math behavioral modifications."""
    
    def __init__(self, vsa: VectorSymbolicArchitecture, buffer_size: int = 50):
        self.vsa = vsa
        self.buffer_size = buffer_size
        self.positive_tokens = deque(maxlen=buffer_size)
        self.negative_tokens = deque(maxlen=buffer_size)
        self.token_rewards = defaultdict(float)
        # Modifications state
        self.echo_strength = defaultdict(lambda: 1.0)
        self.predator_tokens = set()
        self.prey_tokens = set()
        self.bubble_threshold = 5.0
        self.crash_tokens = set()
        self.lock = threading.Lock()

    def add_positive_feedback(self, tokens: List[str], reward: float = 1.0):
        with self.lock:
            for token in tokens:
                # Echo Chamber Amplification
                self.echo_strength[token] *= 1.5
                
                # Predator-Prey Dynamics
                current_reward = self.token_rewards.get(token, 0)
                if current_reward > 2.0:
                    self.predator_tokens.add(token)
                    self.prey_tokens.discard(token)
                    for prey in list(self.prey_tokens):
                        self.token_rewards[prey] -= reward * 0.5
                elif current_reward < -1.0:
                    self.prey_tokens.add(token)
                    self.predator_tokens.discard(token)
                    for predator in list(self.predator_tokens):
                        self.token_rewards[predator] -= reward * 0.5
                
                # Apply reward
                self.positive_tokens.append(token)
                self.token_rewards[token] += reward * self.echo_strength[token]
                
                # Market Bubble Detection
                if self.token_rewards[token] > self.bubble_threshold:
                    self.crash_tokens.add(token)

    def add_negative_feedback(self, tokens: List[str], penalty: float = -0.5):
        with self.lock:
            for token in tokens:
                self.negative_tokens.append(token)
                self.echo_strength[token] *= 0.8
                self.token_rewards[token] += penalty
                
                if self.token_rewards[token] < -1.0:
                    self.prey_tokens.add(token)
                    self.predator_tokens.discard(token)

    def get_token_reward(self, token: str) -> float:
        if token in self.crash_tokens:
            reward = self.token_rewards.get(token, 0.0)
            return -abs(reward)  # Market crash inversion
        return self.token_rewards.get(token, 0.0)
    
    def get_similar_token_reward(self, token: str, top_k: int = 5) -> float:
        if token not in self.vsa.codebook:
            return 0.0
        
        token_vec = self.vsa.codebook[token]
        similar_rewards = []
        
        for rewarded_token, reward in self.token_rewards.items():
            if rewarded_token in self.vsa.codebook:
                sim = self.vsa.similarity(token_vec, self.vsa.codebook[rewarded_token])
                if sim > 0.5:
                    similar_rewards.append(reward * sim)
        
        if similar_rewards:
            return np.mean(sorted(similar_rewards, reverse=True)[:top_k])
        return 0.0

# =====================================================================
# SYMPLECTIC ISOHEDRAL TILER KERNEL
# =====================================================================
class SymplecticIsohedralTiler:
    """Tiling generator using symplectic transforms of polarization vectors."""
    
    def __init__(self, vsa: VectorSymbolicArchitecture, grid_size: Tuple[int, int] = (16, 16), cell_size: float = 1.0):
        self.vsa = vsa
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.base_tile = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=float)
        self._token_angles: Dict[str, float] = {}

    @staticmethod
    def symplectic_rotation(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _token_angle_from_index(self, token_vec: np.ndarray) -> float:
        if token_vec.shape[0] < 2:
            return 0.0
        return float(np.arctan2(token_vec[1], token_vec[0]))

    def _get_token_angle(self, token: str) -> float:
        if token not in self._token_angles:
            vec = self.vsa.add_to_codebook(token)
            self._token_angles[token] = self._token_angle_from_index(vec)
        return self._token_angles[token]

    def _tile_origin(self, i: int, j: int) -> np.ndarray:
        return np.array([j * self.cell_size, i * self.cell_size], dtype=float)

    def build_tiling_from_tokens(self, tokens: List[str], wrap: bool = True) -> List[TileInstance]:
        if not tokens:
            raise ValueError("Need at least one token to build a tiling.")
        
        rows, cols = self.grid_size
        n_tokens = len(tokens)
        tiles = []

        for i in range(rows):
            for j in range(cols):
                idx = (i * cols + j) % n_tokens if wrap else min(i * cols + j, n_tokens - 1)
                token = tokens[idx]
                theta = self._get_token_angle(token)
                verts = (self.base_tile @ self.symplectic_rotation(theta).T) + self._tile_origin(i, j)
                tiles.append(TileInstance(token=token, grid_pos=(i, j), vertices=verts, angle=theta))
        
        return tiles

    def render_svg(self, tiles: List[TileInstance], filepath: str, scale: float = 30.0):
        rows, cols = self.grid_size
        width, height = int(cols * self.cell_size * scale) + 100, int(rows * self.cell_size * scale) + 100
        
        svg_lines = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<style>text { font-size: 10px; text-anchor: middle; }</style>',
            '<rect width="100%" height="100%" fill="white"/>'
        ]
        
        offset_x, offset_y = 50, 50
        
        for tile in tiles:
            centroid = tile.vertices.mean(axis=0)
            cx, cy = offset_x + centroid[0] * scale, offset_y + centroid[1] * scale
            color = f"hsl({int(np.degrees(tile.angle)) % 360}, 70%, 60%)"
            
            pts = [(offset_x + v[0]*scale, offset_y + v[1]*scale) for v in tile.vertices]
            pts_str = " ".join([f"{x},{y}" for x, y in pts])
            svg_lines.append(f'<polygon points="{pts_str}" fill="{color}" stroke="black" stroke-width="0.5"/>')
            svg_lines.append(f'<text x="{cx}" y="{cy}" fill="white">{tile.token}</text>')
        
        svg_lines.append('</svg>')
        
        with open(filepath, 'w') as f:
            f.write("\n".join(svg_lines))

# =====================================================================
# NEURAL GENERATION KERNEL (WITH ALL MODIFICATIONS)
# =====================================================================
class NeuralKernel:
    """
    Self-contained neural kernel combining VSA, RL, and tiling.
    All 10 non-math modifications are integrated and active.
    """
    
    def __init__(self, dimensions: int = 2048, grid_size: Tuple[int, int] = (8, 8)):
        self.vsa = VectorSymbolicArchitecture(dimensions=dimensions)
        self.transition_encoder = TransitionEncoder(self.vsa)
        self.feedback_buffer = FeedbackBuffer(self.vsa, buffer_size=100)
        self.tiler = SymplecticIsohedralTiler(self.vsa, grid_size=grid_size, cell_size=1.0)
        
        # State for non-math modifications
        self.priming_token = None
        self.generation_count = 0
        self.coherence_decay = 0.99
        self.drift_rate = 0.001
        self.dialect_a_tokens = set()
        self.dialect_b_tokens = set()
        self.semantic_categories = {}
        self.dialect_coherence_bonus = 0.3
        
        # Generation state
        self.generated_sequence = []
        self.tiling_tiles = []
        self.generation_buffer = deque(maxlen=20)
        
        print(f"Neural Kernel initialized | Dimensions: {dimensions} | Grid: {grid_size}")
    
    def train_from_corpus(self, corpus_path: str, max_workers: int = 8, batch_size: int = 100):
        """Train the kernel from a text corpus file."""
        with open(corpus_path, encoding="utf-8") as f:
            raw_text = f.read()
        sentences = raw_text.split(".")
        corpus = [s.split() for s in sentences if s.split()]
        
        self.transition_encoder.learn_transitions(corpus, max_workers=max_workers, batch_size=batch_size)
        
        for sentence in corpus:
            for token in sentence:
                self.vsa.add_to_codebook(token)
        
        self._build_semantic_categories()
    
    def _build_semantic_categories(self):
        """Build 8 semantic categories from polarized vectors."""
        categories = defaultdict(list)
        for token, vec in self.vsa.codebook.items():
            if vec.shape[0] >= 2:
                angle = np.arctan2(vec[1], vec[0])
                category_id = int((angle + np.pi) / (np.pi / 4)) % 8
                categories[f"cat_{category_id}"].append(token)
        self.semantic_categories = dict(categories)
    
    def _get_token_category(self, token: str) -> Optional[str]:
        for cat_id, tokens in self.semantic_categories.items():
            if token in tokens:
                return cat_id
        return None
    
    def _compute_semantic_plausibility(self, candidate: str, context: List[str]) -> float:
        if candidate not in self.vsa.codebook:
            return 0.0
        
        candidate_vec = self.vsa.codebook[candidate]
        recent_context = context[-3:]
        similarities = []
        
        for ctx_token in recent_context:
            if ctx_token in self.vsa.codebook:
                ctx_vec = self.vsa.codebook[ctx_token]
                sim = self.vsa.similarity(candidate_vec, ctx_vec)
                
                # Cognitive Dissonance Penalty
                if sim > 0.85:
                    sim *= 0.1
                
                # Priming Effect
                if self.priming_token and self.priming_token in self.vsa.codebook:
                    priming_boost = self.vsa.similarity(candidate_vec, self.vsa.codebook[self.priming_token])
                    sim *= (1 + priming_boost * 0.5)
                
                # Entropy Injection
                if np.random.random() < 0.05:
                    sim += np.random.normal(0, 0.1)
                
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _get_incompatible_category(self, current_category: str) -> str:
        current_id = int(current_category.split("_")[1])
        return f"cat_{(current_id + 4) % 8}"
    
    def _apply_modifications_to_probs(self, probs: Dict[str, float], rl_weight: float) -> Dict[str, float]:
        """Apply all non-math modifications to probability distribution."""
        adjusted_probs = {}
        
        for token, prob in probs.items():
            # Market Bubble & Crash
            if token in self.feedback_buffer.crash_tokens:
                total_reward = -abs(self.feedback_buffer.token_rewards.get(token, 0.0))
            else:
                direct_reward = self.feedback_buffer.get_token_reward(token)
                similar_reward = self.feedback_buffer.get_similar_token_reward(token)
                total_reward = direct_reward + (similar_reward * 0.5)
            
            # Echo Chamber Amplification
            total_reward *= self.feedback_buffer.echo_strength.get(token, 1.0)
            
            # Dialect Forking
            if token in self.dialect_a_tokens:
                total_reward += self.dialect_coherence_bonus
            elif token in self.dialect_b_tokens:
                total_reward -= self.dialect_coherence_bonus
            
            adjusted_probs[token] = prob * np.exp(total_reward * rl_weight)
        
        # Normalize
        total = sum(adjusted_probs.values())
        return {k: v/total for k, v in adjusted_probs.items()} if total > 0 else probs
    
    def generate_stream(self, seed: List[str], max_tokens: int = 50, 
                       temperature: float = 1.0, error_rate: float = 0.0001,
                       plausibility_weight: float = 0.9, rl_weight: float = 2.0) -> Generator[str, None, None]:
        """Stream generate tokens with all modifications active."""
        context = seed.copy()
        self.priming_token = None
        
        for step in range(max_tokens):
            # Creative Destruction Cycle
            if len(self.generated_sequence) % 50 == 0 and self.generated_sequence:
                if self.feedback_buffer.token_rewards:
                    top_token = max(self.feedback_buffer.token_rewards.items(), key=lambda x: x[1])[0]
                    self.feedback_buffer.token_rewards[top_token] = -5.0
                    self.vsa.codebook[top_token] = self.vsa.create_polarized_vector()
            
            # Semantic Drift
            self.generation_count += 1
            if self.generation_count % 10 == 0:
                drift_angle = np.random.normal(0, self.drift_rate)
                drift_matrix = self.tiler.symplectic_rotation(drift_angle)
                for token, vec in self.vsa.codebook.items():
                    # FIXED: Reshape as (-1, 2) for interleaved (x,y) pairs
                    reshaped = vec.reshape(-1, 2)
                    rotated = reshaped @ drift_matrix
                    self.vsa.codebook[token] = rotated.reshape(-1)
            
            force_error = np.random.random() < error_rate
            
            if force_error and len(context) >= 1:
                last_token = context[-1]
                last_category = self._get_token_category(last_token)
                
                if last_category:
                    error_category = self._get_incompatible_category(last_category)
                    candidate_tokens = self.semantic_categories.get(error_category, [])
                    
                    if candidate_tokens:
                        plausibility_scores = {token: self._compute_semantic_plausibility(token, context) for token in candidate_tokens}
                        
                        probs = {}
                        total_counts = sum(self.transition_encoder.unigram_counts.values())
                        for token in candidate_tokens:
                            base_prob = self.transition_encoder.unigram_counts.get(token, 1) / total_counts
                            plausibility = plausibility_scores.get(token, 0.0)
                            probs[token] = base_prob * (1.0 + plausibility * plausibility_weight * 10.0)
                        
                        probs = self._apply_modifications_to_probs(probs, rl_weight)
                        
                        tokens = list(probs.keys())
                        prob_vals = np.array(list(probs.values()))
                        if temperature > 0:
                            prob_vals = np.exp(np.log(prob_vals + 1e-9) / temperature)
                        prob_vals /= np.sum(prob_vals)
                        
                        next_token = np.random.choice(tokens, p=prob_vals)
                        self.generation_buffer.append(next_token)
                        self.generated_sequence.append(next_token)
                        
                        if self.priming_token is None:
                            self.priming_token = next_token
                        
                        yield next_token
                        context.append(next_token)
                        
                        # Quantum Decoherence
                        if len(self.generated_sequence) % 5 == 0:
                            for token, vec in self.vsa.codebook.items():
                                self.vsa.codebook[token] = (vec * self.coherence_decay) + np.random.normal(0, 0.01, vec.shape)
                            self._update_tiling()
                        continue
            
            # Normal generation
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
                probs = self._apply_modifications_to_probs(probs, rl_weight)
                tokens = list(probs.keys())
                prob_vals = np.array(list(probs.values()))
                if temperature > 0:
                    prob_vals = np.exp(np.log(prob_vals + 1e-9) / temperature)
                prob_vals /= np.sum(prob_vals)
                next_token = np.random.choice(tokens, p=prob_vals)
            
            self.generation_buffer.append(next_token)
            self.generated_sequence.append(next_token)
            
            if self.priming_token is None:
                self.priming_token = next_token
            
            yield next_token
            context.append(next_token)
            
            # Quantum Decoherence & Tiling update
            if len(self.generated_sequence) % 5 == 0:
                for token, vec in self.vsa.codebook.items():
                    self.vsa.codebook[token] = (vec * self.coherence_decay) + np.random.normal(0, 0.01, vec.shape)
                self._update_tiling()
    
    def _update_tiling(self):
        """Update tiling from current generation state."""
        if self.generated_sequence:
            try:
                tokens = self.generated_sequence[-32:] if len(self.generated_sequence) > 32 else self.generated_sequence
                self.tiling_tiles = self.tiler.build_tiling_from_tokens(tokens)
            except Exception:
                pass  # Silent fail
    
    def render_tiling(self, filepath: str):
        """Render current tiling to SVG file."""
        if self.tiling_tiles:
            self.tiler.render_svg(self.tiling_tiles, filepath)
    
    def apply_feedback(self, tokens: List[str], reward: float):
        """Apply reinforcement learning feedback."""
        if reward > 0:
            self.feedback_buffer.add_positive_feedback(tokens, reward)
            # Dialect forking: positive feedback moves tokens to dialect A
            for token in tokens:
                self.dialect_a_tokens.add(token)
                self.dialect_b_tokens.discard(token)
        else:
            self.feedback_buffer.add_negative_feedback(tokens, reward)
            # Dialect forking: negative feedback moves tokens to dialect B
            for token in tokens:
                self.dialect_b_tokens.add(token)
                self.dialect_a_tokens.discard(token)
    
    def get_state(self) -> Dict:
        """Export current kernel state for serialization."""
        return {
            "generated_sequence": self.generated_sequence,
            "token_rewards": dict(self.feedback_buffer.token_rewards),
            "dialect_a_tokens": list(self.dialect_a_tokens),
            "dialect_b_tokens": list(self.dialect_b_tokens),
            "generation_count": self.generation_count,
            "priming_token": self.priming_token
        }
    
    def set_state(self, state: Dict):
        """Import kernel state from serialized data."""
        self.generated_sequence = state.get("generated_sequence", [])
        self.dialect_a_tokens = set(state.get("dialect_a_tokens", []))
        self.dialect_b_tokens = set(state.get("dialect_b_tokens", []))
        self.generation_count = state.get("generation_count", 0)
        self.priming_token = state.get("priming_token", None)
        self.feedback_buffer.token_rewards.update(state.get("token_rewards", {}))

# =====================================================================
# KERNEL API (SINGLE POINT OF ENTRY)
# =====================================================================
class VSAKernelAPI:
    """Single API entry point for the neural kernel."""
    
    def __init__(self, dimensions: int = 2048, grid_size: Tuple[int, int] = (8, 8)):
        self.kernel = NeuralKernel(dimensions=dimensions, grid_size=grid_size)
    
    def train(self, corpus_path: str, max_workers: int = 8, batch_size: int = 100):
        """Train kernel from corpus file."""
        self.kernel.train_from_corpus(corpus_path, max_workers=max_workers, batch_size=batch_size)
        return self
    
    def generate(self, seed: str, max_tokens: int = 50, **kwargs) -> str:
        """Generate text from seed prompt."""
        tokens = list(self.kernel.generate_stream(seed.split(), max_tokens=max_tokens, **kwargs))
        return " ".join(tokens)
    
    def stream(self, seed: str, max_tokens: int = 50, **kwargs) -> Generator[str, None, None]:
        """Stream generate tokens."""
        return self.kernel.generate_stream(seed.split(), max_tokens=max_tokens, **kwargs)
    
    def feedback(self, tokens: List[str], reward: float):
        """Apply reinforcement feedback."""
        self.kernel.apply_feedback(tokens, reward)
    
    def render_visualization(self, filepath: str = "tiling.svg"):
        """Render tiling visualization."""
        self.kernel.render_tiling(filepath)
    
    def save(self, directory: str):
        """Save kernel state and models."""
        os.makedirs(directory, exist_ok=True)
        self.kernel.vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        self.kernel.transition_encoder.save_model(directory)
        state = self.kernel.get_state()
        with open(os.path.join(directory, "kernel_state.pkl"), 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, directory: str):
        """Load kernel state and models."""
        self.kernel.vsa.load_codebook(os.path.join(directory, "codebook.pkl"))
        self.kernel.transition_encoder.load_model(directory)
        with open(os.path.join(directory, "kernel_state.pkl"), 'rb') as f:
            state = pickle.load(f)
        self.kernel.set_state(state)
    
    def get_metrics(self) -> Dict:
        """Get kernel performance metrics."""
        return {
            "vocabulary_size": len(self.kernel.vsa.codebook),
            "unigram_count": len(self.kernel.transition_encoder.unigram_counts),
            "bigram_count": sum(len(v) for v in self.kernel.transition_encoder.bigram_transitions.values()),
            "trigram_count": sum(len(v) for v in self.kernel.transition_encoder.trigram_transitions.values()),
            "total_generated": len(self.kernel.generated_sequence),
            "positive_feedback_count": len([r for r in self.kernel.feedback_buffer.token_rewards.values() if r > 0]),
            "negative_feedback_count": len([r for r in self.kernel.feedback_buffer.token_rewards.values() if r < 0]),
            "dialect_a_size": len(self.kernel.dialect_a_tokens),
            "dialect_b_size": len(self.kernel.dialect_b_tokens),
            "crash_tokens": len(self.kernel.feedback_buffer.crash_tokens),
            "predator_tokens": len(self.kernel.feedback_buffer.predator_tokens),
            "prey_tokens": len(self.kernel.feedback_buffer.prey_tokens)
        }

# =====================================================================
# USAGE EXAMPLE
# =====================================================================
if __name__ == "__main__":
    # Quick demonstration
    api = VSAKernelAPI(dimensions=256, grid_size=(6, 6))
    
    # Minimal training data
    with open(input("Filename: "), encoding="utf-8") as f: 
            sample_corpus = f.read().split(".")[:KB_LEN]
    
    for sentence in sample_corpus:
        for token in sentence:
            api.kernel.vsa.add_to_codebook(token)
        api.kernel.transition_encoder.learn_transitions([sentence.split()], max_workers=1)
    
    api.kernel._build_semantic_categories()
    while True:
        # Generate with feedback
        print("Generating text...")
        for token in api.stream(input("USER: "), max_tokens=750, temperature=0.8):
            print(token, end=" ", flush=True)
        print()
        
    

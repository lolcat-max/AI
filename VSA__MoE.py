import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import threading
import os

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            default = args[0]
            super().__init__(default)
        else:
            super().__init__(int)

# =====================================================================
# 2D POLARIZATION VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions  # Must be even for 2D polarization
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

    def bind_polarized(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """Polarization-aware binding with 2D channel swapping."""
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        dim = len(vec_a) // 2
        
        fft_a_swapped = np.ones_like(fft_a, dtype=complex)
        fft_a_swapped[:dim] = fft_b[dim:]
        fft_a_swapped[dim:] = fft_b[:dim]
        
        result = np.fft.ifft(fft_a + fft_a_swapped)
        return np.real(result)

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
        print(f"✓ Polarized codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        print(f"✓ Polarized codebook loaded from {filepath}")

# =====================================================================
# POLARIZATION TRANSITION ENCODER (N-GRAM COUNTS)
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

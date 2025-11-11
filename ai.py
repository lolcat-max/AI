"""
Enhanced Transitioning for Streaming Text Generation
Implements smooth token-to-token transitions with n-gram bindings
NOW WITH MULTITHREADING, PROGRESS BARS, AND MODEL SAVE/LOAD
"""
KB_LEN = 999
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import pickle
import os

# =====================================================================
# VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_vector(self, normalize: bool = True) -> np.ndarray:
        vec = np.random.randn(self.dimensions)
        if normalize:
            vec = vec / np.linalg.norm(vec)
        return vec

    def bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        result = np.fft.ifft(fft_a * fft_b)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def add_to_codebook(self, symbol: str) -> np.ndarray:
        with self.lock:
            if symbol not in self.codebook:
                self.codebook[symbol] = self.create_vector()
            return self.codebook[symbol]

    def save_codebook(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
        print(f"✓ Codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        print(f"✓ Codebook loaded from {filepath}")


# =====================================================================
# TRANSITION ENCODER
# =====================================================================
class TransitionEncoder:
    """Encode and learn token transitions (bigrams, trigrams) with multithreading"""

    def __init__(self, vsa):
        self.vsa = vsa
        self.transition_memory = {}
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.lock = threading.Lock()

    def encode_bigram(self, token1: str, token2: str) -> np.ndarray:
        key = (token1, token2)
        with self.lock:
            if key not in self.bigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                self.bigram_vectors[key] = self.vsa.bind(vec1, vec2)
            return self.bigram_vectors[key]

    def encode_trigram(self, token1: str, token2: str, token3: str) -> np.ndarray:
        key = (token1, token2, token3)
        with self.lock:
            if key not in self.trigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                vec3 = self.vsa.add_to_codebook(token3)
                bound12 = self.vsa.bind(vec1, vec2)
                self.trigram_vectors[key] = self.vsa.bind(bound12, vec3)
            return self.trigram_vectors[key]

    def _process_sequence_batch(self, sequences: List[List[str]]) -> Tuple[int, int]:
        bigram_count = 0
        trigram_count = 0
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
                bigram_count += 1
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])
                trigram_count += 1
        return bigram_count, trigram_count

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning transitions from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        total_bigrams = 0
        total_trigrams = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_sequence_batch, batch): batch for batch in batches}
            with tqdm(total=len(batches), desc="Processing batches", ncols=80) as pbar:
                for future in as_completed(futures):
                    bigram_count, trigram_count = future.result()
                    total_bigrams += bigram_count
                    total_trigrams += trigram_count
                    pbar.update(1)
        print(f"  ✓ Learned {len(self.bigram_vectors)} unique bigrams ({total_bigrams} total)")
        print(f"  ✓ Learned {len(self.trigram_vectors)} unique trigrams ({total_trigrams} total)")

    def get_transition_candidates(self, context: List[str], n: int = 2) -> List[Tuple[str, float]]:
        if len(context) < 1:
            return []
        candidates = defaultdict(float)
        if n >= 2 and len(context) >= 1:
            last_token = context[-1]
            for (t1, t2), vec in self.bigram_vectors.items():
                if t1 == last_token:
                    candidates[t2] += 1.0
        if n >= 3 and len(context) >= 2:
            last_two = tuple(context[-2:])
            for (t1, t2, t3), vec in self.trigram_vectors.items():
                if (t1, t2) == last_two:
                    candidates[t3] += 1.5
        result = [(token, score) for token, score in candidates.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bigram_vectors.pkl"), 'wb') as f:
            pickle.dump(self.bigram_vectors, f)
        with open(os.path.join(directory, "trigram_vectors.pkl"), 'wb') as f:
            pickle.dump(self.trigram_vectors, f)
        print(f"✓ Transition model saved in {directory}")

    def load_model(self, directory: str):
        with open(os.path.join(directory, "bigram_vectors.pkl"), 'rb') as f:
            self.bigram_vectors = pickle.load(f)
        with open(os.path.join(directory, "trigram_vectors.pkl"), 'rb') as f:
            self.trigram_vectors = pickle.load(f)
        print(f"✓ Transition model loaded from {directory}")


# =====================================================================
# ADAPTIVE TRANSITION GENERATOR
# =====================================================================
class AdaptiveTransitionGenerator:
    def __init__(self, vsa, transition_encoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.generation_history = []

    def generate_with_adaptation(self, seed: List[str],
                                 max_tokens: int = 50,
                                 temperature: float = 0.8,
                                 adapt_rate: float = 0.3):
        context = seed.copy()
        for _ in range(max_tokens):
            candidates = self.transition_encoder.get_transition_candidates(context, n=3)
            if not candidates:
                next_token = np.random.choice(list(self.vsa.codebook.keys()))
            else:
                if np.random.random() < adapt_rate:
                    idx = min(len(candidates) - 1, int(np.random.exponential(2)))
                    next_token = candidates[idx][0]
                else:
                    next_token = candidates[0][0]
            yield next_token
            context.append(next_token)
            if len(context) >= 2:
                self.transition_encoder.encode_bigram(context[-2], context[-1])


# =====================================================================
# DEMONSTRATION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED TRANSITIONING FOR TEXT GENERATION (MULTITHREADED + SAVE/LOAD)")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=4096)
    trans_encoder = TransitionEncoder(vsa)

    choice = input("[N]ew model or [L]oad existing? ").strip().lower()

    if choice == "l":
        directory = input("Model directory: ").strip()
        vsa.load_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.load_model(directory)
    else:
        filename = input("Filename: ")
        print("Loading corpus...")

        with open(filename, encoding="utf-8") as f:
            raw_text = f.read()

        print("Splitting into sentences...")
        sentences = raw_text.split(".")[:KB_LEN]

        print("Tokenizing corpus...")
        corpus = []
        for sentence in tqdm(sentences, desc="Tokenizing", ncols=80):
            tokens = sentence.split()
            if tokens:
                corpus.append(tokens)

        print(f"Corpus loaded: {len(corpus)} sequences")

        print("[1] Learning Transition Patterns (Multithreaded)")
        print("-"*80)
        trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)

        print("Building vocabulary...")
        for sentence in tqdm(corpus, desc="Vocabulary", ncols=80):
            for token in sentence:
                vsa.add_to_codebook(token)

        print(f"  ✓ Vocabulary size: {len(vsa.codebook)} unique tokens")

        directory = input("Save model to directory: ").strip()
        vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.save_model(directory)

    print("[2] ADAPTIVE TRANSITIONS (Online Learning)")
    print("-"*80)
    adaptive_gen = AdaptiveTransitionGenerator(vsa, trans_encoder)

    while True:
        user_input = input("USER: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        tokens = []
        print("AI: ", end='')
        for token in adaptive_gen.generate_with_adaptation(
            user_input.split(),
            max_tokens=800,
            temperature=0.7,
            adapt_rate=0.9
        ):
            tokens.append(token)
            print(token, end=' ', flush=True)
        print("\n")

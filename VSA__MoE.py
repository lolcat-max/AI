"""
Enhanced Transitioning for Streaming Text Generation
Implements smooth token-to-token transitions with n-gram bindings
NOW WITH SOBOL SEQUENCES, TRUE MoE, MULTITHREADING, PROGRESS BARS, AND SAVE/LOAD
"""
KB_LEN = -1
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import pickle
import os
from scipy.stats import qmc
import math

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            super().__init__(args[0])
        else:
            super().__init__(int)

# =====================================================================
# VECTOR SYMBOLIC ARCHITECTURE WITH SOBOL SEQUENCES
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048, scramble: bool = True):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.sobol_engine = qmc.Sobol(d=min(dimensions, 21201), scramble=scramble)
        self._sobol_counter = 0
        self._sobol_buffer = None
        self._buffer_size = 0

    def _refill_sobol_buffer(self, n_points: int):
        power = max(1, math.ceil(math.log2(n_points)))
        self._buffer_size = 2 ** power
        uniform_samples = self.sobol_engine.random(self._buffer_size)
        self._sobol_buffer = qmc.scale(uniform_samples, l_bounds=-3, u_bounds=3)
        self._sobol_counter = 0

    def create_vector(self, normalize: bool = True) -> np.ndarray:
        if self._sobol_buffer is None or self._sobol_counter >= self._buffer_size:
            self._refill_sobol_buffer(1024)
        
        vec = self._sobol_buffer[self._sobol_counter].copy()
        self._sobol_counter += 1
        
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
# MIXTURE OF EXPERTS ROUTER WITH CACHING
# =====================================================================
class MixtureOfExpertsRouter:
    def __init__(self, vsa: VectorSymbolicArchitecture, expert_names: List[str]):
        self.vsa = vsa
        self.expert_vectors = {name: self.vsa.add_to_codebook(f"EXPERT_{name.upper()}") for name in expert_names}
        self._routing_cache = {}
        self._cache_lock = threading.Lock()

    def route(self, context_vector: np.ndarray) -> Dict[str, float]:
        context_key = tuple(np.round(context_vector, 4))
        with self._cache_lock:
            if context_key in self._routing_cache:
                return self._routing_cache[context_key]
        
        similarities = []
        expert_names = list(self.expert_vectors.keys())
        for name in expert_names:
            sim = self.vsa.similarity(context_vector, self.expert_vectors[name])
            similarities.append(sim)
            
        exp_sims = np.exp(np.array(similarities) - np.max(similarities))
        probabilities = exp_sims / np.sum(exp_sims)
        result = dict(zip(expert_names, probabilities))
        
        with self._cache_lock:
            self._routing_cache[context_key] = result
        return result

# =====================================================================
# TRANSITION ENCODER WITH LOCK-FREE PARALLELISM
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa):
        self.vsa = vsa
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.bigram_transitions = defaultdict(IntDefaultDict)
        self.trigram_transitions = defaultdict(IntDefaultDict)

    def encode_bigram(self, token1: str, token2: str, local_bigram=None):
        if local_bigram is not None:
            local_bigram[token1][token2] += 1
        else:
            with threading.Lock():
                self.bigram_transitions[token1][token2] += 1
                if (token1, token2) not in self.bigram_vectors:
                    vec1 = -self.vsa.add_to_codebook(token1)
                    vec2 = self.vsa.add_to_codebook(token2)
                    self.bigram_vectors[(token1, token2)] = self.vsa.bind(vec1, vec2)

    def encode_trigram(self, token1: str, token2: str, token3: str, local_trigram=None):
        if local_trigram is not None:
            local_trigram[(token1, token2)][token3] += 1
        else:
            with threading.Lock():
                self.trigram_transitions[(token1, token2)][token3] += 1
                if (token1, token2, token3) not in self.trigram_vectors:
                    vec1 = self.vsa.add_to_codebook(token1)
                    vec2 = self.vsa.add_to_codebook(token2)
                    vec3 = self.vsa.add_to_codebook(token3)
                    bound12 = self.vsa.bind(vec1, vec2)
                    self.trigram_vectors[(token1, token2, token3)] = self.vsa.bind(bound12, vec3)

    def _process_sequence_batch(self, sequences: List[List[str]]) -> Tuple[dict, dict, int, int]:
        local_bigram = defaultdict(IntDefaultDict)
        local_trigram = defaultdict(IntDefaultDict)
        bigram_count, trigram_count = 0, 0
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                local_bigram[sequence[i]][sequence[i+1]] += 1
                bigram_count += 1
            for i in range(len(sequence) - 2):
                local_trigram[(sequence[i], sequence[i+1])][sequence[i+2]] += 1
                trigram_count += 1
        return local_bigram, local_trigram, bigram_count, trigram_count

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning transitions from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        
        all_local_bigrams = []
        all_local_trigrams = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(batches), desc="Processing batches", ncols=80) as pbar:
            for local_bigram, local_trigram, bg_count, tg_count in executor.map(self._process_sequence_batch, batches):
                all_local_bigrams.append(local_bigram)
                all_local_trigrams.append(local_trigram)
                pbar.update(1)
        
        print("Merging transition counts...")
        for local_bigram in tqdm(all_local_bigrams, desc="Merging bigrams", ncols=80):
            for token1, transitions in local_bigram.items():
                for token2, count in transitions.items():
                    self.bigram_transitions[token1][token2] += count
        
        for local_trigram in tqdm(all_local_trigrams, desc="Merging trigrams", ncols=80):
            for token_pair, transitions in local_trigram.items():
                for token3, count in transitions.items():
                    self.trigram_transitions[token_pair][token3] += count
        
        total_bigrams = sum(len(v) for v in self.bigram_transitions.values())
        total_trigrams = sum(len(v) for v in self.trigram_transitions.values())
        print(f"  ✓ Learned {total_bigrams} unique bigram transitions.")
        print(f"  ✓ Learned {total_trigrams} unique trigram transitions.")

    def get_bigram_probabilities(self, last_token: str) -> Optional[Dict[str, float]]:
        if last_token not in self.bigram_transitions: return None
        candidates = self.bigram_transitions[last_token]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def get_trigram_probabilities(self, last_two_tokens: Tuple[str, str]) -> Optional[Dict[str, float]]:
        if last_two_tokens not in self.trigram_transitions: return None
        candidates = self.trigram_transitions[last_two_tokens]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bigram_transitions.pkl"), 'wb') as f:
            pickle.dump(self.bigram_transitions, f)
        with open(os.path.join(directory, "trigram_transitions.pkl"), 'wb') as f:
            pickle.dump(self.trigram_transitions, f)
        with open(os.path.join(directory, "bigram_vectors.pkl"), 'wb') as f:
            pickle.dump(self.bigram_vectors, f)
        with open(os.path.join(directory, "trigram_vectors.pkl"), 'wb') as f:
            pickle.dump(self.trigram_vectors, f)
        print(f"✓ Transition model saved in {directory}")

    def load_model(self, directory: str):
        with open(os.path.join(directory, "bigram_transitions.pkl"), 'rb') as f:
            self.bigram_transitions = pickle.load(f)
        with open(os.path.join(directory, "trigram_transitions.pkl"), 'rb') as f:
            self.trigram_transitions = pickle.load(f)
        with open(os.path.join(directory, "bigram_vectors.pkl"), 'rb') as f:
            self.bigram_vectors = pickle.load(f)
        with open(os.path.join(directory, "trigram_vectors.pkl"), 'rb') as f:
            self.trigram_vectors = pickle.load(f)
        print(f"✓ Transition model loaded from {directory}")

# =====================================================================
# MIXTURE OF EXPERTS GENERATOR WITH INCREMENTAL BUNDLING
# =====================================================================
class MoEGenerator:
    def __init__(self, vsa: VectorSymbolicArchitecture, transition_encoder: TransitionEncoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.router = MixtureOfExpertsRouter(vsa, expert_names=['bigram', 'trigram'])
        self._context_cache = {}

    def _get_context_vector(self, context: List[str], length: int) -> np.ndarray:
        cache_key = tuple(context[-length:])
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        vectors = [self.vsa.codebook[tok] for tok in cache_key if tok in self.vsa.codebook]
        if not vectors:
            vec = self.vsa.create_vector()
        else:
            vec = self.vsa.bundle(vectors)
        
        self._context_cache[cache_key] = vec
        return vec

    def stream_generation(self, seed: List[str], max_tokens: int = 50, temperature: float = 1.0):
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return

        for _ in range(max_tokens):
            bigram_probs = self.transition_encoder.get_bigram_probabilities(context[-1]) if len(context) >= 1 else None
            trigram_probs = self.transition_encoder.get_trigram_probabilities(tuple(context[-2:])) if len(context) >= 2 else None

            if trigram_probs:
                context_vec = self._get_context_vector(context, 2)
                routing_probs = self.router.route(context_vec)
            else:
                routing_probs = {'bigram': 1.0, 'trigram': 0.0}

            final_probs = defaultdict(float)
            if bigram_probs:
                for token, prob in bigram_probs.items(): final_probs[token] += routing_probs['bigram'] * prob
            if trigram_probs:
                for token, prob in trigram_probs.items(): final_probs[token] += routing_probs['trigram'] * prob

            if not final_probs:
                next_token = np.random.choice(list(self.vsa.codebook.keys()))
            else:
                tokens, probs = list(final_probs.keys()), np.array(list(final_probs.values()))
                if temperature > 0:
                    probs = np.log(probs + 1e-9) / temperature
                    probs = np.exp(probs)
                probs /= np.sum(probs)
                next_token = np.random.choice(tokens, p=probs)
            
            yield next_token
            context.append(next_token)

# =====================================================================
# DEMONSTRATION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("MIXTURE OF EXPERTS FOR TEXT GENERATION (VSA + SOBOL + MULTITHREADING)")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=256, scramble=True)
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
        sentences = raw_text.split(".")[:KB_LEN]
        corpus = [sentence.split() for sentence in tqdm(sentences, desc="Tokenizing", ncols=80) if sentence.split()]
        print(f"Corpus loaded: {len(corpus)} sequences")
        print("[1] Learning Transition Patterns (Multithreaded)")
        print("-"*80)
        trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)
        print("Building vocabulary...")
        for sentence in tqdm(corpus, desc="Vocabulary", ncols=80):
            for token in sentence: vsa.add_to_codebook(token)
        print(f"  ✓ Vocabulary size: {len(vsa.codebook)} unique tokens")
        directory = input("Save model to directory: ").strip()
        vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.save_model(directory)

    print("\n[2] GENERATING TEXT WITH MIXTURE OF EXPERTS")
    print("-"*80)
    moe_gen = MoEGenerator(vsa, trans_encoder)
    
    while True:
        user_input = input("USER: ")
        if user_input.lower() in ['quit', 'exit', 'q']: break
        print("AI: ", end='')
        for token in moe_gen.stream_generation(
            user_input.split(),
            max_tokens=350,
            temperature=0.7
        ):
            print(token, end=' ', flush=True)
        print("\n")

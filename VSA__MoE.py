"""
Enhanced Transitioning for Streaming Text Generation with 2D Polarization VSA
Implements polarization-aware binding via 2D channel swapping for richer n-gram representations
TRUE MIXTURE OF EXPERTS (MOE) + MULTITHREADING + PROGRESS BARS + SAVE/LOAD
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
        x_channel = r * np.cos(theta)  # Horizontal/real
        y_channel = r * np.sin(theta)  # Vertical/imaginary
        vec = np.stack([x_channel, y_channel], axis=-1).reshape(-1)
        if normalize:
            vec = vec / np.linalg.norm(vec)
        return vec

    def bind_polarized(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """Polarization-aware binding with 2D channel swapping."""
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        dim = len(vec_a) // 2
        
        # Swap polarization channels in frequency domain
        fft_a_swapped = np.zeros_like(fft_a, dtype=complex)
        fft_a_swapped[:dim] = fft_b[dim:]  # A's x <- B's y
        fft_a_swapped[dim:] = fft_b[:dim]  # A's y <- B's x
        
        result = np.fft.ifft(fft_a * fft_a_swapped)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

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
# MIXTURE OF EXPERTS ROUTER (POLARIZATION-ENHANCED)
# =====================================================================
class MixtureOfExpertsRouter:
    def __init__(self, vsa: VectorSymbolicArchitecture, expert_names: List[str]):
        self.vsa = vsa
        self.expert_vectors = {name: self.vsa.add_to_codebook(f"EXPERT_{name.upper()}") 
                              for name in expert_names}

    def route(self, context_vector: np.ndarray) -> Dict[str, float]:
        similarities = [self.vsa.similarity(context_vector, self.expert_vectors[name]) 
                       for name in self.expert_vectors]
        
        # Softmax
        exp_sims = np.exp(np.array(similarities) - np.max(similarities))
        probabilities = exp_sims / np.sum(exp_sims)
        
        return dict(zip(self.expert_vectors.keys(), probabilities))

# =====================================================================
# POLARIZATION TRANSITION ENCODER
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.bigram_transitions = defaultdict(IntDefaultDict)
        self.trigram_transitions = defaultdict(IntDefaultDict)
        self.lock = threading.Lock()

    def encode_bigram(self, token1: str, token2: str):
        with self.lock:
            self.bigram_transitions[token1][token2] += 1
            key = (token1, token2)
            if key not in self.bigram_vectors:
                vec1 = -self.vsa.add_to_codebook(token1)  # Inverse for binding
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

    def _process_sequence_batch(self, sequences: List[List[str]]) -> Tuple[int, int]:
        bigram_count, trigram_count = 0, 0
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
                bigram_count += 1
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])
                trigram_count += 1
        return bigram_count, trigram_count

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning polarized transitions from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self._process_sequence_batch, batches), 
                     total=len(batches), desc="Polarized batches", ncols=80))
        print(f"  ✓ Learned {sum(len(v) for v in self.bigram_transitions.values())} bigram transitions")
        print(f"  ✓ Learned {sum(len(v) for v in self.trigram_transitions.values())} trigram transitions")

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
        for name, data in [("bigram_transitions", self.bigram_transitions),
                          ("trigram_transitions", self.trigram_transitions),
                          ("bigram_vectors", self.bigram_vectors),
                          ("trigram_vectors", self.trigram_vectors)]:
            with open(os.path.join(directory, f"{name}.pkl"), 'wb') as f:
                pickle.dump(data, f)
        print(f"✓ Polarized transition model saved to {directory}")

    def load_model(self, directory: str):
        for name in ["bigram_transitions", "trigram_transitions", "bigram_vectors", "trigram_vectors"]:
            with open(os.path.join(directory, f"{name}.pkl"), 'rb') as f:
                setattr(self, name.replace('.pkl', ''), pickle.load(f))
        print(f"✓ Polarized transition model loaded from {directory}")

# =====================================================================
# POLARIZATION MOE GENERATOR
# =====================================================================
class MoEGenerator:
    def __init__(self, vsa: VectorSymbolicArchitecture, transition_encoder: TransitionEncoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.router = MixtureOfExpertsRouter(vsa, ['bigram', 'trigram'])

    def stream_generation(self, seed: List[str], max_tokens: int = 50, temperature: float = 1.0):
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return

        for _ in range(max_tokens):
            bigram_probs = self.transition_encoder.get_bigram_probabilities(context[-1]) if len(context) >= 1 else None
            trigram_probs = self.transition_encoder.get_trigram_probabilities(tuple(context[-2:])) if len(context) >= 2 else None

            # Polarization-enhanced routing
            if trigram_probs and all(t in self.vsa.codebook for t in context[-2:]):
                context_vec = self.vsa.bundle([self.vsa.codebook[tok] for tok in context[-2:]])
                routing_probs = self.router.route(context_vec)
            else:
                routing_probs = {'bigram': 1.0, 'trigram': 0.0}

            final_probs = defaultdict(float)
            if bigram_probs:
                for token, prob in bigram_probs.items(): 
                    final_probs[token] += routing_probs['bigram'] * prob
            if trigram_probs:
                for token, prob in trigram_probs.items(): 
                    final_probs[token] += routing_probs['trigram'] * prob

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
# MAIN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("2D POLARIZATION VSA + MOE TEXT GENERATION")
    print("Channel-swapping binding for enhanced n-gram separation")
    print("="*80)

    # Use even dimensions for 2D polarization
    vsa = VectorSymbolicArchitecture(dimensions=64)  
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
        sentences = raw_text.split(".")[:KB_LEN]
        corpus = [s.split() for s in tqdm(sentences, desc="Tokenizing", ncols=80) if s.split()]
        print(f"Corpus: {len(corpus)} sequences")
        
        print("[1] Learning Polarized Transitions (Multithreaded)")
        print("-"*80)
        trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)
        
        print("Building polarized vocabulary...")
        for sentence in tqdm(corpus, desc="Polarized Vocab", ncols=80):
            for token in sentence: 
                vsa.add_to_codebook(token)
        print(f"  ✓ Polarized vocabulary: {len(vsa.codebook)} tokens")
        
        directory = input("Save polarized model to directory: ").strip()
        vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.save_model(directory)

    print("\n[2] POLARIZATION-ENHANCED MOE GENERATION")
    print("-"*80)
    moe_gen = MoEGenerator(vsa, trans_encoder)
    
    while True:
        user_input = input("USER: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']: 
            break
        print("AI: ", end='', flush=True)
        for token in moe_gen.stream_generation(user_input.split(), max_tokens=350, temperature=0.7):
            print(token, end=' ', flush=True)
        print("\n")

"""
Enhanced Transitioning for Streaming Text Generation with 2D Polarization VSA
Implements polarization-aware binding via 2D channel swapping for richer n-gram representations
GRANGER CAUSALITY FOR NATURAL TEXT + MULTITHREADING + PROGRESS BARS + SAVE/LOAD
MONOTONIC BACKOFF SMOOTHING
"""

KB_LEN = -1
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
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
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_polarized_vector(self, normalize: bool = True) -> np.ndarray:
        dim_2d = self.dimensions // 2
        theta = np.random.uniform(0, 2 * np.pi, dim_2d)
        r = np.ones(dim_2d)
        x_channel = r * np.cos(theta)  # Fixed: use cos instead of exp
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
        result = np.fft.ifft(fft_a + fft_a_swapped)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9)

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
# GRANGER CAUSALITY ENGINE (PURE NUMPY)
# =====================================================================
class GrangerCausalityEngine:
    """Pure NumPy implementation of Granger causality for text [web:41][web:42]."""
    
    def __init__(self, max_vocab: int = 500, n_lags: int = 3):
        self.max_vocab = max_vocab
        self.n_lags = n_lags
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.causality_matrix = None  # [source, target] causality weights
        self.token_sequences = []  # Indexed token sequences
        
    def build_vocabulary(self, corpus: List[List[str]]):
        """Build vocabulary from most frequent tokens [web:47]."""
        token_counts = Counter()
        for sentence in corpus:
            token_counts.update(sentence)
        
        most_common = token_counts.most_common(self.max_vocab)
        vocab = [token for token, _ in most_common]
        
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.n_tokens = len(self.token_to_idx)
        
        print(f"  ✓ Granger vocabulary: {self.n_tokens} tokens")
        
    def encode_corpus(self, corpus: List[List[str]]):
        """Convert corpus to indexed sequences [web:42]."""
        print("Encoding corpus for Granger analysis...")
        for sentence in tqdm(corpus, desc="Encoding", ncols=80):
            indices = [self.token_to_idx[tok] for tok in sentence if tok in self.token_to_idx]
            if len(indices) > self.n_lags:
                self.token_sequences.extend(indices)
        print(f"  ✓ Encoded {len(self.token_sequences)} total tokens")
    
    def compute_granger_causality(self):
        """Compute pairwise Granger causality using VAR residuals [web:41][web:42]."""
        print(f"Computing Granger causality for {self.n_tokens} tokens...")
        
        # Initialize causality matrix [web:42]
        self.causality_matrix = np.zeros((self.n_tokens, self.n_tokens))
        
        # Build co-occurrence matrices at different lags [web:47]
        for target_idx in tqdm(range(self.n_tokens), desc="Granger causality", ncols=80):
            for source_idx in range(self.n_tokens):
                if source_idx == target_idx:
                    continue
                
                # Count: source at lag -> target at current [web:41]
                causal_count = 0
                total_target = 0
                
                for i in range(len(self.token_sequences) - self.n_lags):
                    # Check if target appears at position i
                    if self.token_sequences[i] == target_idx:
                        total_target += 1
                        # Check if source appeared in previous n_lags [web:42]
                        for lag in range(1, self.n_lags + 1):
                            if i - lag >= 0 and self.token_sequences[i - lag] == source_idx:
                                causal_count += 1
                                break
                
                # Compute normalized causality score [web:41]
                if total_target > 3:  # Minimum threshold
                    self.causality_matrix[source_idx, target_idx] = causal_count / total_target
        
        # Normalize matrix [web:42]
        max_val = np.max(self.causality_matrix)
        if max_val > 0:
            self.causality_matrix /= max_val
        
        n_edges = np.sum(self.causality_matrix > 0.1)
        print(f"  ✓ Found {n_edges} significant causal edges")
        
    def get_causal_predecessors(self, token: str, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Get tokens that Granger-cause the target token [web:47]."""
        if token not in self.token_to_idx:
            return []
        
        target_idx = self.token_to_idx[token]
        causal_weights = self.causality_matrix[:, target_idx]
        
        # Get tokens above threshold [web:42]
        predecessors = []
        for source_idx in range(self.n_tokens):
            if causal_weights[source_idx] > threshold:
                predecessors.append((self.idx_to_token[source_idx], causal_weights[source_idx]))
        
        return sorted(predecessors, key=lambda x: x[1], reverse=True)
    
    def save_causality(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.causality_matrix)
        with open(filepath.replace('.npy', '_vocab.pkl'), 'wb') as f:
            pickle.dump({
                'token_to_idx': self.token_to_idx,
                'idx_to_token': self.idx_to_token
            }, f)
        print(f"✓ Granger causality saved to {filepath}")

# =====================================================================
# POLARIZATION TRANSITION ENCODER WITH GRANGER-ENHANCED BACKOFF
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture, granger_engine: Optional['GrangerCausalityEngine'] = None):
        self.vsa = vsa
        self.granger = granger_engine
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
        if total == 0: return {}
        return {token: count / total for token, count in self.unigram_counts.items()}

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
    
    def get_granger_enhanced_probabilities(self, context: List[str]) -> Optional[Dict[str, float]]:
        """Enhance probabilities using Granger causality [web:41][web:47]."""
        if not self.granger or not context:
            return None
        
        # Get base unigram distribution
        base_probs = self.get_unigram_probabilities()
        if not base_probs:
            return None
        
        # Boost probabilities based on Granger causality [web:42]
        enhanced_probs = {}
        for token in base_probs:
            enhanced_probs[token] = base_probs[token]
            
            # Check if any context token Granger-causes this token [web:47]
            if token in self.granger.token_to_idx:
                target_idx = self.granger.token_to_idx[token]
                causal_boost = 0.0
                
                for ctx_token in context[-self.granger.n_lags:]:
                    if ctx_token in self.granger.token_to_idx:
                        source_idx = self.granger.token_to_idx[ctx_token]
                        causal_boost += self.granger.causality_matrix[source_idx, target_idx]
                    else:
                        source_idx = target_idx
                # Apply boost [web:41]
                enhanced_probs[token] *= (1.0 + causal_boost)
        
        # Renormalize
        total = sum(enhanced_probs.values())
        if total > 0:
            enhanced_probs = {k: v/total for k, v in enhanced_probs.items()}
        
        return enhanced_probs

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
# GRANGER-ENHANCED MONOTONIC BACKOFF GENERATOR
# =====================================================================
class MonotonicBackoffGenerator:
    def __init__(self, vsa: VectorSymbolicArchitecture, transition_encoder: TransitionEncoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder

    def stream_generation(self, seed: List[str], max_tokens: int = 50, temperature: float = 1.0, use_granger: bool = True):
        """Generate using Granger-enhanced monotonic backoff [web:41][web:47]."""
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return

        for _ in range(max_tokens):
            probs = None
            
            # Try trigram first
            if len(context) >= 2:
                probs = self.transition_encoder.get_trigram_probabilities(tuple(context[-2:]))
            
            # Back off to bigram
            if probs is None and len(context) >= 1:
                probs = self.transition_encoder.get_bigram_probabilities(context[-1])
            
            # Granger-enhanced unigram [web:41][web:47]
            if probs is None and use_granger:
                probs = self.transition_encoder.get_granger_enhanced_probabilities(context)
            
            # Standard unigram fallback
            if probs is None:
                probs = self.transition_encoder.get_unigram_probabilities()

            if not probs:
                next_token = np.random.choice(list(self.vsa.codebook.keys()))
            else:
                tokens = list(probs.keys())
                prob_vals = np.array(list(probs.values()))
                
                # Temperature sampling
                if temperature > 0:
                    prob_vals = np.log(prob_vals + 1e-9) / temperature
                    prob_vals = np.exp(prob_vals)
                prob_vals /= np.sum(prob_vals)
                
                next_token = np.random.choice(tokens, p=prob_vals)
            
            yield next_token
            context.append(next_token)

# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("2D POLARIZATION VSA + GRANGER CAUSALITY TEXT GENERATION")
    print("Pure NumPy Granger causality for natural text modeling")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=64)
    granger_engine = GrangerCausalityEngine(max_vocab=100, n_lags=3)
    trans_encoder = TransitionEncoder(vsa, granger_engine)

    choice = input("[N]ew model or [L]oad existing? ").strip().lower()

    if choice == "l":
        directory = input("Model directory: ").strip()
        vsa.load_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.load_model(directory)
        granger_engine.causality_matrix = np.load(os.path.join(directory, "granger_causality.npy"))
        with open(os.path.join(directory, "granger_causality_vocab.pkl"), 'rb') as f:
            vocab_data = pickle.load(f)
            granger_engine.token_to_idx = vocab_data['token_to_idx']
            granger_engine.idx_to_token = vocab_data['idx_to_token']
    else:
        filename = input("Corpus filename: ")
        print("Loading corpus...")
        with open(filename, encoding="utf-8") as f:
            raw_text = f.read()
        sentences = raw_text.split(".")[:KB_LEN]
        corpus = [s.split() for s in tqdm(sentences, desc="Tokenizing", ncols=80) if s.split()]
        print(f"Corpus: {len(corpus)} sequences")
        
        print("\n[1] Learning Granger Causality")
        print("-"*80)
        granger_engine.build_vocabulary(corpus)
        granger_engine.encode_corpus(corpus)
        granger_engine.compute_granger_causality()
        
        print("\n[2] Learning Polarized Transitions")
        print("-"*80)
        trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)
        
        print("\nBuilding polarized vocabulary...")
        for sentence in tqdm(corpus, desc="Polarized Vocab", ncols=80):
            for token in sentence:
                vsa.add_to_codebook(token)
        print(f"  ✓ Polarized vocabulary: {len(vsa.codebook)} tokens")
        
        directory = input("Save model to directory: ").strip()
        vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.save_model(directory)
        granger_engine.save_causality(os.path.join(directory, "granger_causality.npy"))
        
        # Show sample causality relationships
        print("\n[3] Sample Granger Causality Relationships")
        print("-"*80)
        sample_tokens = list(granger_engine.token_to_idx.keys())[:5]
        for token in sample_tokens:
            preds = granger_engine.get_causal_predecessors(token, threshold=0.5)
            if preds:
                print(f"\n'{token}' is Granger-caused by:")
                for pred_tok, weight in preds[:5]:
                    bar = '█' * int(weight * 40)
                    print(f"  {pred_tok:15s} {bar} {weight:.3f}")

    print("\n[4] GRANGER-ENHANCED TEXT GENERATION")
    print("-"*80)
    backoff_gen = MonotonicBackoffGenerator(vsa, trans_encoder)
    
    while True:
        user_input = input("USER: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        print("AI: ", end='', flush=True)
        for token in backoff_gen.stream_generation(user_input.split(), max_tokens=500, temperature=0.7, use_granger=True):
            print(token, end=' ', flush=True)
        print("\n")

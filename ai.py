"""
Enhanced Transitioning for Streaming Text Generation
Implements smooth token-to-token transitions with n-gram bindings
NOW WITH MULTITHREADING AND PROGRESS BARS
"""
KB_LEN = 9999
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()  # Thread-safe codebook access
        
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


class TransitionEncoder:
    """Encode and learn token transitions (bigrams, trigrams) with multithreading"""
    
    def __init__(self, vsa):
        self.vsa = vsa
        self.transition_memory = {}
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.lock = threading.Lock()  # Thread-safe transition storage
        
    def encode_bigram(self, token1: str, token2: str) -> np.ndarray:
        """Encode transition between two tokens (thread-safe)"""
        key = (token1, token2)
        
        with self.lock:
            if key not in self.bigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                self.bigram_vectors[key] = self.vsa.bind(vec1, vec2)
            return self.bigram_vectors[key]
    
    def encode_trigram(self, token1: str, token2: str, token3: str) -> np.ndarray:
        """Encode transition across three tokens (thread-safe)"""
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
        """Process a batch of sequences in a worker thread"""
        bigram_count = 0
        trigram_count = 0
        
        for sequence in sequences:
            # Learn bigrams
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
                bigram_count += 1
            
            # Learn trigrams
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])
                trigram_count += 1
        
        return bigram_count, trigram_count
    
    def learn_transitions(self, corpus: List[List[str]], 
                          max_workers: int = 8, 
                          batch_size: int = 100):
        """Learn transitions from training corpus with multithreading"""
        print("Learning transitions from corpus...")
        
        # Split corpus into batches for parallel processing
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        
        total_bigrams = 0
        total_trigrams = 0
        
        # Process batches in parallel with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_sequence_batch, batch): batch 
                for batch in batches
            }
            
            with tqdm(total=len(batches), desc="Processing batches", ncols=80) as pbar:
                for future in as_completed(futures):
                    bigram_count, trigram_count = future.result()
                    total_bigrams += bigram_count
                    total_trigrams += trigram_count
                    pbar.update(1)
        
        print(f"  ✓ Learned {len(self.bigram_vectors)} unique bigrams ({total_bigrams} total)")
        print(f"  ✓ Learned {len(self.trigram_vectors)} unique trigrams ({total_trigrams} total)")
    
    def get_transition_candidates(self, context: List[str], 
                                  n: int = 2) -> List[Tuple[str, float]]:
        """Get candidate next tokens based on transition patterns"""
        if len(context) < 1:
            return []
        
        candidates = defaultdict(float)
        
        # Check bigram transitions
        if n >= 2 and len(context) >= 1:
            last_token = context[-1]
            for (t1, t2), vec in self.bigram_vectors.items():
                if t1 == last_token:
                    candidates[t2] += 1.0
        
        # Check trigram transitions
        if n >= 3 and len(context) >= 2:
            last_two = tuple(context[-2:])
            for (t1, t2, t3), vec in self.trigram_vectors.items():
                if (t1, t2) == last_two:
                    candidates[t3] += 1.5
        
        result = [(token, score) for token, score in candidates.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result


class AdaptiveTransitionGenerator:
    """Adaptive transitions that learn during generation"""
    
    def __init__(self, vsa, transition_encoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.generation_history = []
        
    def generate_with_adaptation(self, seed: List[str],
                                 max_tokens: int = 50,
                                 temperature: float = 0.8,
                                 adapt_rate: float = 0.3):
        """Generate text that adapts transitions during generation"""
        context = seed.copy()
        
        for i in range(max_tokens):
            candidates = self.transition_encoder.get_transition_candidates(
                context, n=3
            )
            
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


class SequentialEncoder:
    def __init__(self, vsa):
        self.vsa = vsa


# ============================================================================
# DEMONSTRATION - ENHANCED TRANSITIONS WITH MULTITHREADING
# ============================================================================

print("="*80)
print("ENHANCED TRANSITIONING FOR TEXT GENERATION (MULTITHREADED)")
print("="*80)

# Initialize
vsa = VectorSymbolicArchitecture(dimensions=4096)
seq_encoder = SequentialEncoder(vsa)
trans_encoder = TransitionEncoder(vsa)

# Load and tokenize corpus with progress bar
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
    if tokens:  # Skip empty sequences
        corpus.append(tokens)

print(f"Corpus loaded: {len(corpus)} sequences")

print("[1] Learning Transition Patterns (Multithreaded)")
print("-"*80)

# Learn transitions with multithreading
trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)

# Build vocabulary with progress bar
print("Building vocabulary...")
for sentence in tqdm(corpus, desc="Vocabulary", ncols=80):
    for token in sentence:
        vsa.add_to_codebook(token)

print(f"  ✓ Vocabulary size: {len(vsa.codebook)} unique tokens")

print("[2] ADAPTIVE TRANSITIONS (Online Learning)")
print("-"*80)

adaptive_gen = AdaptiveTransitionGenerator(vsa, trans_encoder)

while True:
    user_input = input("USER: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
        
    tokens = []
    print("AI: ", end='')
    
    # Generate with progress tracking
    for i, token in enumerate(adaptive_gen.generate_with_adaptation(
        user_input.split(), 
        max_tokens=800, 
        temperature=0.7, 
        adapt_rate=0.9
    )):
        tokens.append(token)
        print(token, end=' ', flush=True)
    
    print()

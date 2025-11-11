"""
Enhanced Transitioning for Streaming Text Generation
Implements smooth token-to-token transitions with n-gram bindings
"""
KB_LEN = 999
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class TransitionEncoder:
    """Encode and learn token transitions (bigrams, trigrams)"""
    
    def __init__(self, vsa):
        self.vsa = vsa
        self.transition_memory = {}  # Store learned transitions
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        
    def encode_bigram(self, token1: str, token2: str) -> np.ndarray:
        """Encode transition between two tokens"""
        key = (token1, token2)
        
        if key not in self.bigram_vectors:
            vec1 = self.vsa.add_to_codebook(token1)
            vec2 = self.vsa.add_to_codebook(token2)
            # Bind tokens to capture transition
            self.bigram_vectors[key] = self.vsa.bind(vec1, vec2)
        
        return self.bigram_vectors[key]
    
    def encode_trigram(self, token1: str, token2: str, token3: str) -> np.ndarray:
        """Encode transition across three tokens"""
        key = (token1, token2, token3)
        
        if key not in self.trigram_vectors:
            vec1 = self.vsa.add_to_codebook(token1)
            vec2 = self.vsa.add_to_codebook(token2)
            vec3 = self.vsa.add_to_codebook(token3)
            # Bind all three with positional weighting
            bound12 = self.vsa.bind(vec1, vec2)
            self.trigram_vectors[key] = self.vsa.bind(bound12, vec3)
        
        return self.trigram_vectors[key]
    
    def learn_transitions(self, corpus: List[List[str]]):
        """Learn transitions from training corpus"""
        print("Learning transitions from corpus...")
        
        for sequence in corpus:
            # Learn bigrams
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
            
            # Learn trigrams
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])
        
        print(f"  ✓ Learned {len(self.bigram_vectors)} bigrams")
        print(f"  ✓ Learned {len(self.trigram_vectors)} trigrams")
    
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
                    # This bigram starts with our last token
                    candidates[t2] += 1.0
        
        # Check trigram transitions
        if n >= 3 and len(context) >= 2:
            last_two = tuple(context[-2:])
            for (t1, t2, t3), vec in self.trigram_vectors.items():
                if (t1, t2) == last_two:
                    # This trigram continues our context
                    candidates[t3] += 1.5  # Weight trigrams higher
        
        # Convert to list and sort
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
            # Get candidates from learned transitions
            candidates = self.transition_encoder.get_transition_candidates(
                context, n=3
            )
            
            if not candidates:
                # Fallback to random selection
                next_token = np.random.choice(list(self.vsa.codebook.keys()))
            else:
                # Adaptive selection
                if np.random.random() < adapt_rate:
                    # Explore: try less common transitions
                    idx = min(len(candidates) - 1, int(np.random.exponential(2)))
                    next_token = candidates[idx][0]
                else:
                    # Exploit: use common transitions
                    next_token = candidates[0][0]
            
            yield next_token
            context.append(next_token)
            
            # Learn this new transition
            if len(context) >= 2:
                self.transition_encoder.encode_bigram(context[-2], context[-1])


# ============================================================================
# DEMONSTRATION - ENHANCED TRANSITIONS
# ============================================================================

print("="*80)
print("ENHANCED TRANSITIONING FOR TEXT GENERATION")
print("="*80)

# Rebuild VSA
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        
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
        if symbol not in self.codebook:
            self.codebook[symbol] = self.create_vector()
        return self.codebook[symbol]

class SequentialEncoder:
    def __init__(self, vsa):
        self.vsa = vsa

# Initialize
vsa = VectorSymbolicArchitecture(dimensions=4096)
seq_encoder = SequentialEncoder(vsa)
trans_encoder = TransitionEncoder(vsa)
corpus = []
# Training corpus with better structure
with open(input("Filename: "), encoding="utf-8") as f:
    training_corpus = f.read().split(".")[:KB_LEN]
for sentence in training_corpus:
    corpus.append(sentence.split())

print("\n[1] Learning Transition Patterns")
print("-"*80)

# Tokenize and learn
corpus_tokenized = corpus
trans_encoder.learn_transitions(corpus_tokenized)

# Build vocabulary
for sentence in corpus_tokenized:
    for token in sentence:
        vsa.add_to_codebook(token)

print("\n ADAPTIVE TRANSITIONS (Online Learning)")
print("-"*80)

adaptive_gen = AdaptiveTransitionGenerator(vsa, trans_encoder)

while True:
    tokens = []
    for i, token in enumerate(adaptive_gen.generate_with_adaptation(
        input("USER: ").split(), max_tokens=800, temperature=0.7, adapt_rate=0.9
    )):
        tokens.append(token)
        print(token, end=' ')
    print()
    print()

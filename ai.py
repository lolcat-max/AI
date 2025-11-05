import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
KB_LEN = 99999
class PureVSA_TextGenerator:
    """
    Pure Vector Symbolic Architecture (VSA) text generation system.
    Uses only VSA operations: binding, bundling, and permutation.
    
    Based on principles from Kanerva's Hyperdimensional Computing:
    - All entities are high-dimensional vectors (10,000+ dimensions)
    - Information is distributed across dimensions
    - Computations use only vector operations
    - Relations evaluated via vector similarity
    """
    
    def __init__(self, vector_dim: int = 10000):
        """
        Initialize pure VSA system.
        
        Args:
            vector_dim: Dimensionality of hypervectors (typically 10,000)
        """
        self.vector_dim = vector_dim
        
        # VSA memory structures
        self.atomic_vectors = {}  # Base hypervectors for words
        self.sequence_memory = []  # Stored sequence patterns
        self.position_vectors = {}  # Position encodings
        
        # Transition memory for better continuation
        self.transition_memory = defaultdict(Counter)  # word/ngram -> next words
        
        # Pre-generate position vectors for sequence encoding
        self._initialize_position_vectors(max_positions=100)
    
    def _generate_random_hypervector(self, seed: str) -> np.ndarray:
        """
        Generate a random bipolar hypervector deterministically.
        
        VSA principle: Random high-dimensional vectors are quasi-orthogonal.
        With D=10,000, can represent ~2^D distinct concepts.
        """
        hash_seed = int(hashlib.sha256(seed.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(hash_seed)
        
        # Bipolar vectors {-1, +1} are common in MAP (Multiply-Add-Permute) VSA
        return rng.choice([-1.0, 1.0], size=self.vector_dim)
    
    def _initialize_position_vectors(self, max_positions: int):
        """Generate position vectors for sequential encoding."""
        for i in range(max_positions):
            self.position_vectors[i] = self._generate_random_hypervector(f"POS_{i}")
    
    def _bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        VSA Binding operation (element-wise multiplication).
        
        Properties:
        - Associative and commutative
        - Approximately involutory: bind(bind(A, B), B) ≈ A
        - Creates new vector dissimilar to inputs
        """
        return np.multiply(vec1, vec2)
    
    def _unbind(self, bound_vec: np.ndarray, key_vec: np.ndarray) -> np.ndarray:
        """
        VSA Unbinding (inverse of binding).
        For bipolar vectors, unbinding = binding (self-inverse).
        """
        return self._bind(bound_vec, key_vec)
    
    def _bundle(self, vectors: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        VSA Bundling operation (superposition via addition).
        
        Properties:
        - Similar to any constituent vector
        - Noise-resistant: can recover components even with degradation
        - Commutative and associative
        """
        if not vectors:
            return np.zeros(self.vector_dim)
        
        bundled = np.sum(vectors, axis=0)
        
        if normalize:
            # Binarize to maintain bipolar property
            bundled = np.sign(bundled)
            bundled[bundled == 0] = 1  # Handle exact zeros
        
        return bundled
    
    def _permute(self, vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        VSA Permutation operation (coordinate rotation).
        
        Used to create sequence-sensitive representations.
        Different permutations create orthogonal vectors.
        """
        return np.roll(vec, shift)
    
    def _similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity between hypervectors.
        
        For bipolar vectors, this is proportional to Hamming similarity.
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get or create atomic hypervector for a word."""
        word = word.lower()
        if word not in self.atomic_vectors:
            self.atomic_vectors[word] = self._generate_random_hypervector(word)
        return self.atomic_vectors[word]
    
    def encode_sequence(self, words: List[str]) -> np.ndarray:
        """
        Encode a sequence using VSA operations.
        
        Method: Bind each word to its position vector, then bundle.
        This preserves both content and order information.
        """
        bound_vectors = []
        
        for i, word in enumerate(words):
            if i >= len(self.position_vectors):
                # Generate more position vectors if needed
                self.position_vectors[i] = self._generate_random_hypervector(f"POS_{i}")
            
            word_vec = self.get_word_vector(word)
            pos_vec = self.position_vectors[i]
            
            # Bind word to position
            bound = self._bind(word_vec, pos_vec)
            bound_vectors.append(bound)
        
        # Bundle all bound pairs into single hypervector
        return self._bundle(bound_vectors)
    
    def encode_ngram_sequence(self, words: List[str], n: int = 3) -> Dict[Tuple[str, ...], np.ndarray]:
        """
        Encode sequences as n-gram VSA patterns.
        Each n-gram is encoded using sequential binding and permutation.
        """
        ngram_vectors = {}
        
        for i in range(len(words) - n + 1):
            ngram_words = tuple(words[i:i+n])
            
            # Encode n-gram using permutation chaining
            # Method: word1 ⊗ ρ(word2) ⊗ ρ²(word3) ...
            vec = self.get_word_vector(ngram_words[0])
            
            for j in range(1, n):
                next_word = self.get_word_vector(ngram_words[j])
                # Permute to encode position
                permuted = self._permute(next_word, shift=j)
                vec = self._bind(vec, permuted)
            
            ngram_vectors[ngram_words] = vec
        
        return ngram_vectors
    
    def train_on_corpus(self, text: str):
        """
        Train the VSA system on a text corpus.
        
        Creates a distributed memory of sequential patterns.
        """
        import re
        
        # Tokenize
        sentences = re.split(r'[.\n]+', text.lower())
        sentences = [s.strip()+"." for s in sentences if s.strip()]
        
        print(f"Training on {len(sentences)} sentences...")
        
        # Build sequence memory using VSA encoding
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                continue
            
            # Build transition memory (for longer generation)
            for n in range(1, min(4, len(words))):
                for i in range(len(words) - n):
                    context = ' '.join(words[i:i+n])
                    next_word = words[i+n]
                    self.transition_memory[context][next_word] += 1
            
            # Encode full sequence
            seq_vector = self.encode_sequence(words)
            self.sequence_memory.append({
                'vector': seq_vector,
                'words': words,
                'length': len(words)
            })
            
            # Also encode n-grams for local patterns
            ngram_vecs = self.encode_ngram_sequence(words, n=3)
            for ngram, vec in ngram_vecs.items():
                self.sequence_memory.append({
                    'vector': vec,
                    'words': list(ngram),
                    'length': len(ngram)
                })
        
        print(f"Stored {len(self.sequence_memory)} VSA patterns in memory")
        print(f"Built transition memory with {len(self.transition_memory)} contexts")
    
    def predict_next_word_hybrid(self, context: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Hybrid prediction using both VSA and transition memory.
        Combines symbolic VSA operations with statistical patterns.
        """
        if not context:
            return []
        
        candidates = Counter()
        
        # Method 1: Transition memory (fast, statistical)
        for n in range(min(len(context), 3), 0, -1):
            context_key = ' '.join(context[-n:])
            if context_key in self.transition_memory:
                for word, count in self.transition_memory[context_key].items():
                    candidates[word] += count * (n / 3.0)  # Weight by context length
        
        # Method 2: VSA similarity (slower, semantic)
        if len(candidates) < top_k:
            context_vec = self.encode_sequence(context[-3:])
            
            # Check all patterns for semantic matches
            for pattern in self.sequence_memory:
                if pattern['length'] <= len(context):
                    continue
                
                # Compare context to pattern prefix
                pattern_context = pattern['words'][:len(context)]
                pattern_vec = self.encode_sequence(pattern_context)
                similarity = self._similarity(context_vec, pattern_vec)
                
                if similarity > 0.4:  # Lower threshold for more variety
                    # Get next word from pattern
                    if len(pattern['words']) > len(context):
                        next_word = pattern['words'][len(context)]
                        candidates[next_word] += similarity * 100  # Scale to compete with counts
        
        if not candidates:
            return []
        
        # Normalize and return top-k
        max_score = max(candidates.values())
        normalized = [(word, score/max_score) for word, score in candidates.items()]
        return sorted(normalized, key=lambda x: x[1], reverse=True)[:top_k]
    
    def generate_text(self, seed: str, max_words: int = 500, temperature: float = 0.8, 
                     min_words: int = 50, stop_on_period: bool = False) -> str:
        """
        Generate text using pure VSA operations with enhanced continuation.
        
        Args:
            seed: Starting text
            max_words: Maximum words to generate
            temperature: Sampling randomness (0=greedy, 1=uniform)
            min_words: Minimum words before allowing stop
            stop_on_period: Whether to stop at sentence end
        
        Returns:
            Generated text
        """
        words = seed.lower().split()
        consecutive_failures = 0
        
        for iteration in range(max_words):
            # Use hybrid prediction
            predictions = self.predict_next_word_hybrid(words[-5:], top_k=15)
            
            if not predictions:
                consecutive_failures += 1
                
                # Try with shorter context
                if len(words) > 1:
                    predictions = self.predict_next_word_hybrid(words[-2:], top_k=15)
                
                if not predictions and consecutive_failures > 3:
                    print(f"[Stopped after {iteration} words - no predictions]")
                    break
                
                if not predictions:
                    continue
            else:
                consecutive_failures = 0
            
            # Sample with temperature
            if temperature > 0 and len(predictions) > 1:
                scores = np.array([score for _, score in predictions])
                scores = np.maximum(scores, 0.001)
                
                # Apply temperature
                scores = scores ** (1.0 / temperature)
                probs = scores / scores.sum()
                
                next_word = np.random.choice([w for w, _ in predictions], p=probs)
            else:
                next_word = predictions[0][0]
            
            words.append(next_word)
            
            # Optional early stopping at sentence boundaries
            if stop_on_period and iteration >= min_words:
                if next_word in ['.', '!', '?']:
                    break
        
        return ' '.join(words)
    
    def query_memory(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Query the VSA memory for similar sequences."""
        query_words = query.lower().split()
        query_vec = self.encode_sequence(query_words)
        
        similarities = []
        for pattern in self.sequence_memory:
            if pattern['length'] >= len(query_words):
                sim = self._similarity(query_vec, pattern['vector'])
                similarities.append((' '.join(pattern['words']), sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]


# --- Demonstration ---
if __name__ == "__main__":
    print("="*60)
    print("Pure VSA Text Generation System")
    print("Based on Hyperdimensional Computing principles")
    print("="*60)
    
    # Initialize
    vsa = PureVSA_TextGenerator(vector_dim=10000)
    
    # Load corpus
    filename = input("\nCorpus filename (or press Enter for example): ").strip()
    
    if filename:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                corpus = f.read()[:KB_LEN]
            print(f"Loaded from {filename}")
        except Exception as e:
            print(f"Error: {e}")
            filename = None
    
    if not filename:
        corpus = """the quick brown fox jumps over the lazy dog.
        a dog is a loyal companion. the cat sat on the mat.
        birds fly in the sky. fish swim in the ocean.
        the sun rises in the east. the moon shines at night.
        people walk on the street. cars drive on the road.
        trees grow in the forest. flowers bloom in spring."""
    
    # Train
    vsa.train_on_corpus(corpus)
    
    while True:
        user_input = input("\n> ").strip()
       
        generated = vsa.generate_text(user_input, max_words=800, 
                                        temperature=0.7, min_words=100)
        print(f"\nGenerated ({len(generated.split())} words):\n{generated}")
           
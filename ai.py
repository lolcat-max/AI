import torch
import numpy as np
from collections import Counter, defaultdict
import os
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FLOAT64 = True
torch_dtype = torch.float64 if USE_FLOAT64 else torch.float32

print(f"Using {device}, precision {torch_dtype}")


# ================================================================
# WORD FEATURE APPROXIMATOR
# ================================================================

class WordFeatureApproximator:
    """
    Computes multi-dimensional feature similarity between words.
    Words approximate features of other words through:
    - Lexical similarity (character overlap, edit distance)
    - Frequency correlation (similar corpus distributions)
    - Phonetic patterns (sound structure)
    - Contextual co-occurrence (distributional semantics)
    - Morphological features (prefix/suffix patterns)
    """
    def __init__(self, corpus_tokens, word_freq):
        self.word_freq = word_freq
        self.total_words = len(corpus_tokens)
        
        # Build co-occurrence matrix for distributional similarity
        self.cooccurrence = self._build_cooccurrence(corpus_tokens, window=2)
        
        # Cache feature vectors for performance
        self.feature_cache = {}
        
    def _build_cooccurrence(self, tokens, window=2):
        """Build word co-occurrence matrix for distributional features."""
        cooccur = defaultdict(Counter)
        for i, word in enumerate(tokens):
            context_start = max(0, i - window)
            context_end = min(len(tokens), i + window + 1)
            for j in range(context_start, context_end):
                if i != j:
                    cooccur[word][tokens[j]] += 1
        return dict(cooccur)
    
    def extract_features(self, word):
        """
        Extract comprehensive feature vector for a word.
        Returns normalized feature array capturing multiple word properties.
        """
        if word in self.feature_cache:
            return self.feature_cache[word]
        
        features = []
        
        # 1. Lexical features (character-level patterns)
        features.append(len(word) / 20.0)  # Normalized length
        features.append(sum(1 for c in word if c.isalpha()) / (len(word) + 1))  # Alpha ratio
        features.append(sum(1 for c in word if c in 'aeiou') / (len(word) + 1))  # Vowel ratio
        
        # 2. Character distribution features
        char_counts = Counter(word)
        features.append(len(char_counts) / (len(word) + 1))  # Character diversity
        features.append(max(char_counts.values()) / (len(word) + 1) if char_counts else 0)  # Max char frequency
        
        # 3. Frequency-based features
        freq = self.word_freq.get(word, 1)
        features.append(np.log(freq + 1) / np.log(self.total_words + 1))  # Log-normalized frequency
        features.append(1.0 / (freq + 1))  # Rarity score
        
        # 4. Phonetic features (approximate sound patterns)
        features.append(1 if word[0] in 'aeiou' else 0 if word else 0)  # Starts with vowel
        features.append(1 if word[-1] in 'aeiou' else 0 if word else 0)  # Ends with vowel
        features.append(word.count('th') + word.count('ch') + word.count('sh'))  # Digraph count
        
        # 5. Morphological features
        features.append(1 if word.endswith('ing') else 0)  # Progressive form
        features.append(1 if word.endswith('ed') else 0)  # Past tense
        features.append(1 if word.endswith('s') else 0)  # Plural/3rd person
        features.append(1 if word.startswith('un') or word.startswith('re') else 0)  # Common prefixes
        
        # 6. Positional character features (encoding word structure)
        if len(word) >= 1:
            features.append(ord(word[0]) / 122.0)  # First character (normalized)
        else:
            features.append(0)
        if len(word) >= 2:
            features.append(ord(word[1]) / 122.0)  # Second character
        else:
            features.append(0)
        if len(word) >= 1:
            features.append(ord(word[-1]) / 122.0)  # Last character
        else:
            features.append(0)
        
        feature_vector = np.array(features)
        self.feature_cache[word] = feature_vector
        return feature_vector
    
    def compute_similarity(self, word1, word2):
        """
        Compute multi-feature similarity between two words.
        Returns value in [0, 1] where 1 = identical features, 0 = completely different.
        """
        # Extract feature vectors
        vec1 = self.extract_features(word1)
        vec2 = self.extract_features(word2)
        
        # Compute cosine similarity between feature vectors
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0
        else:
            cosine_sim = dot_product / (norm1 * norm2)
        
        # Add distributional similarity based on co-occurrence
        distributional_sim = self._distributional_similarity(word1, word2)
        
        # Compute character overlap (Jaccard similarity)
        set1 = set(word1)
        set2 = set(word2)
        if len(set1) == 0 and len(set2) == 0:
            jaccard_sim = 1.0
        elif len(set1 | set2) == 0:
            jaccard_sim = 0.0
        else:
            jaccard_sim = len(set1 & set2) / len(set1 | set2)
        
        # Weighted combination of similarity metrics
        combined_similarity = (
            0.5 * max(0, cosine_sim) +  # Feature vector similarity
            0.3 * distributional_sim +   # Context similarity
            0.2 * jaccard_sim            # Character overlap
        )
        
        return float(np.clip(combined_similarity, 0, 1))
    
    def _distributional_similarity(self, word1, word2):
        """
        Compute distributional similarity based on shared context words.
        Implements simplified word2vec-style distributional semantics.
        """
        if word1 not in self.cooccurrence or word2 not in self.cooccurrence:
            return 0.0
        
        context1 = self.cooccurrence[word1]
        context2 = self.cooccurrence[word2]
        
        # Find shared context words
        shared_contexts = set(context1.keys()) & set(context2.keys())
        
        if not shared_contexts:
            return 0.0
        
        # Compute similarity based on shared context weights
        similarity_sum = sum(
            min(context1[ctx], context2[ctx]) / max(context1[ctx], context2[ctx])
            for ctx in shared_contexts
        )
        
        return similarity_sum / (len(shared_contexts) + 1)
    
    def find_similar_words(self, target_word, candidates, top_k=5):
        """
        Find the k most similar words to target_word from candidates list.
        Returns list of (word, similarity_score) tuples sorted by similarity.
        """
        similarities = [
            (cand, self.compute_similarity(target_word, cand))
            for cand in candidates
            if cand != target_word
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def approximate_word_features(self, word, reference_words):
        """
        Approximate a word's features as a weighted combination of similar words' features.
        This allows words to 'borrow' features from semantically/lexically related words.
        """
        if not reference_words:
            return self.extract_features(word)
        
        # Get own features
        own_features = self.extract_features(word)
        
        # Compute similarity to all reference words
        similarities = [(ref, self.compute_similarity(word, ref)) for ref in reference_words]
        similarities = [(ref, sim) for ref, sim in similarities if sim > 0.1]  # Filter low similarity
        
        if not similarities:
            return own_features
        
        # Weight own features heavily, but blend in similar words' features
        total_weight = 1.0 + sum(sim for _, sim in similarities)
        approximated_features = own_features.copy()
        
        for ref_word, similarity in similarities:
            ref_features = self.extract_features(ref_word)
            approximated_features += (similarity / total_weight) * ref_features
        
        approximated_features = approximated_features / (1 + len(similarities) * 0.3)  # Normalize
        
        return approximated_features


# ================================================================
# CENTRAL KERNEL PROCESSOR
# ================================================================

class CentralKernel:
    """
    Central kernel that applies convolution-like operations to all arrays.
    Acts as a unified processing core for coherence, eigenvalue, and spatial data.
    """
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.kernel = self._create_kernel()
        
    def _create_kernel(self):
        """Create the convolution kernel based on type."""
        if self.kernel_size == 3:
            kernel = np.array([1, 2, 1]) / 4.0
        elif self.kernel_size == 5:
            kernel = np.array([1, 4, 6, 4, 1]) / 16.0
        else:
            kernel = np.bartlett(self.kernel_size)
            kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def convolve_1d(self, array, mode='same'):
        if len(array) == 0:
            return array
        arr = np.array(array)
        if mode == 'same':
            pad_width = self.kernel_size // 2
            arr_padded = np.pad(arr, pad_width, mode='edge')
            result = np.convolve(arr_padded, self.kernel, mode='valid')
        else:
            result = np.convolve(arr, self.kernel, mode=mode)
        if mode == 'same' and len(result) != len(array):
            result = result[:len(array)]
        return result
    
    def process_scores(self, scores):
        if len(scores) < self.kernel_size:
            return scores
        filtered = self.convolve_1d(scores, mode='same')
        filtered = np.clip(filtered, 0, 1)
        return filtered.tolist()
    
    def process_eigenvalues(self, eigenvalues):
        if len(eigenvalues) < 2:
            return eigenvalues
        filtered = self.convolve_1d(eigenvalues, mode='same')
        return filtered
    
    def process_vector(self, vector):
        if len(vector) < self.kernel_size:
            return vector
        filtered = self.convolve_1d(vector, mode='same')
        return filtered


# ================================================================
# SINE RESISTANCE MODULATION
# ================================================================

def sine_resistance(step, novelty, freq=0.08, amp=0.6, phase=0.0):
    oscillation = np.sin(2 * np.pi * freq * step + phase)
    resistance = 1.0 - amp * novelty * max(0.0, oscillation)
    return max(0.1, resistance)


# ================================================================
# EIGENVALUE ISOMORPHISM MODEL
# ================================================================

class EigenIsomorphism:
    def __init__(self, dim=4, kernel=None):
        self.dim = dim
        self.W = np.eye(dim)
        self.last_input = np.zeros(dim)
        self.kernel = kernel

    def update(self, input_vector):
        eigvals, eigvecs = np.linalg.eig(self.W)
        if self.kernel is not None and len(input_vector) >= self.kernel.kernel_size:
            input_vector = self.kernel.process_vector(input_vector[:self.dim])
        delta = np.tanh(0.6 * np.dot(eigvecs.T, input_vector[:self.dim]))
        new_eigvals = eigvals + 0.05 * delta[:len(eigvals)]
        if self.kernel is not None:
            new_eigvals = self.kernel.process_eigenvalues(new_eigvals)
        self.W = eigvecs @ np.diag(new_eigvals) @ np.linalg.inv(eigvecs)
        self.last_input = input_vector
        return np.real(new_eigvals), np.real(eigvecs)

    def project(self, vec):
        eigvals, eigvecs = np.linalg.eig(self.W)
        return np.real(np.dot(eigvecs, np.dot(np.diag(eigvals), np.dot(np.linalg.inv(eigvecs), vec))))


# ================================================================
# NEURAL TRUTH TABLE WASHER
# ================================================================

class NeuralTruthTableWasher:
    def __init__(self, eta_0=0.3, alpha=0.1, epsilon=1e-4,
                 delta=1e-3, beta=1.0, gamma=2.0, mu=0.5,
                 max_iterations=30, kernel=None):
        self.eta_0 = eta_0
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.max_iterations = max_iterations
        self.dtype = torch_dtype
        self.device = device
        self.history = []
        self.kernel = kernel

    def calculate_error(self, T, T_expected):
        T = torch.tensor(T, dtype=self.dtype, device=self.device)
        Texp = torch.tensor(T_expected, dtype=self.dtype, device=self.device)
        return torch.sum((T - Texp) ** 2).item()

    def wash_iteration(self, T, T_expected, eta):
        Tnew = []
        for i in range(len(T)):
            grad = 2 * (T[i] - T_expected[i])
            val = T[i] - eta * grad
            val = max(0.0, min(1.0, val))
            if abs(val - T_expected[i]) < 0.05:
                val = T_expected[i]
            Tnew.append(val)
        return Tnew

    def wash(self, T_contaminated, T_expected):
        Tcur = T_contaminated.copy()
        if self.kernel is not None and len(Tcur) >= self.kernel.kernel_size:
            Tcur = self.kernel.process_scores(Tcur)
        for k in range(self.max_iterations):
            eta = self.eta_0 * np.exp(-self.alpha * k)
            Tnext = self.wash_iteration(Tcur, T_expected, eta)
            err = self.calculate_error(Tnext, T_expected)
            self.history.append(err)
            if err < self.delta:
                break
            Tcur = Tnext
        return Tcur, {"final_error": err, "iterations": k+1}


# ================================================================
# REASONING ENGINE
# ================================================================

class ReasoningEngine:
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.truth_washer = NeuralTruthTableWasher(kernel=kernel)
        self.eigen_system = EigenIsomorphism(kernel=kernel)

    def reason_step(self, coherence_scores, input_vector):
        if self.kernel is not None and len(coherence_scores) >= self.kernel.kernel_size:
            coherence_scores = self.kernel.process_scores(coherence_scores)
        eigvals, eigvecs = self.eigen_system.update(input_vector)
        padded_scores = coherence_scores[:4]
        while len(padded_scores) < 4:
            padded_scores.append(0.5)
        washed, metrics = self.truth_washer.wash(
            padded_scores,
            [1.0 if c > 0.5 else 0.0 for c in padded_scores]
        )
        modulated = []
        scale = 1 + 0.1 * np.mean(eigvals)
        for i in range(len(coherence_scores)):
            if i < len(washed):
                modulated.append(float(np.clip(washed[i] * scale, 0, 1)))
            else:
                modulated.append(float(np.clip(coherence_scores[i] * scale, 0, 1)))
        if self.kernel is not None and len(modulated) >= self.kernel.kernel_size:
            modulated = self.kernel.process_scores(modulated)
        return modulated, np.mean(eigvals), metrics


# ================================================================
# ENHANCED QUANTUM FEATURES WITH WORD APPROXIMATION
# ================================================================

class SchrodingerQuantumFeatures:
    def __init__(self, word_approximator=None):
        self.word_approximator = word_approximator
        
    def extract_quantum_features(self, segment, word_freq, total_words):
        xs = np.array([len(w) for w in segment])
        fs = np.array([word_freq.get(w, 1) for w in segment])
        
        # Base variance calculation
        var = np.var(xs / (fs + 1))
        coherence = 1.0 / (1.0 + var)
        
        # Enhanced: If word approximator available, boost coherence based on word similarities
        if self.word_approximator is not None and len(segment) >= 2:
            # Compute average pairwise similarity in segment
            similarities = []
            for i in range(len(segment) - 1):
                sim = self.word_approximator.compute_similarity(segment[i], segment[i+1])
                similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                # Boost coherence for similar adjacent words
                coherence = coherence * (1.0 + 0.3 * avg_similarity)
                coherence = min(1.0, coherence)
        
        return {"coherence": coherence}


# ================================================================
# MODEL BUILDER
# ================================================================

def build_ngram_model(tokens, n=2):
    model = defaultdict(list)
    for i in range(len(tokens)-n):
        key = tuple(tokens[i:i+n])
        model[key].append(tokens[i+n])
    return model


# ================================================================
# REASONING GENERATOR WITH WORD FEATURE APPROXIMATION
# ================================================================

class ReasoningGenerator:
    def __init__(self, tokens, model, kernel_size=3):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        
        # Initialize word feature approximator
        print("🔬 Building word feature approximator...")
        self.word_approximator = WordFeatureApproximator(tokens, self.word_freq)
        
        # Initialize quantum features with approximator
        self.feature = SchrodingerQuantumFeatures(word_approximator=self.word_approximator)
        
        # Initialize central kernel
        self.central_kernel = CentralKernel(
            kernel_size=kernel_size,
        )
        
        # Initialize reasoning engine with kernel
        self.engine = ReasoningEngine(kernel=self.central_kernel)
        
        # Sine resistance parameters
        self.sine_freq = 0.08
        self.sine_amp = 0.6
        self.sine_phase = 0.0
        
        print(f"🤖 Generator ready with kernel and word approximation!")

    def calculate_novelty(self, word):
        freq = self.word_freq.get(word, 1)
        novelty = 1.0 - np.log(freq + 1) / np.log(self.total_words + 1)
        return float(np.clip(novelty, 0, 1))

    def generate(self, seed, length=50):
        seed_words = seed.lower().split()[:2]
        while len(seed_words) < 2:
            seed_words.append(self.tokens[len(seed_words) % len(self.tokens)])
        seed = tuple(seed_words)
        
        if seed not in self.model:
            seed = self.keys[np.random.randint(len(self.keys))]
        
        output = list(seed)
        
        print(f"\n🌀 Generating {length} words with feature approximation...")
        print(f"   Seed: {' '.join(seed)}\n")
        
        step_count = 0
        
        while len(output) < length:
            recent_text = ' '.join(output[-4:]) if len(output) >= 4 else ' '.join(output)
            input_vec = np.array([ord(c) % 97 / 25 for c in recent_text.ljust(4)[:4]])
            input_vec = self.central_kernel.process_vector(input_vec)

            candidates = self.model.get(seed, [])
            candidates = [w for w in candidates if any(c.isalnum() for c in w)]
            
            if not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue

            coherence_scores = []
            novelty_scores = []
            resistance_factors = []
            
            for cand in candidates:
                # Extract quantum features (now enhanced with word approximation)
                q = self.feature.extract_quantum_features(
                    list(seed) + [cand], 
                    self.word_freq, 
                    self.total_words
                )
                base_coherence = q["coherence"]
                
                # Boost coherence for candidates similar to recent words
                if len(output) >= 1:
                    recent_word = output[-1]
                    similarity = self.word_approximator.compute_similarity(recent_word, cand)
                    base_coherence = base_coherence * (1.0 + 0.2 * similarity)
                
                novelty = self.calculate_novelty(cand)
                resistance_factor = sine_resistance(
                    step_count, 
                    novelty, 
                    freq=self.sine_freq, 
                    amp=self.sine_amp, 
                    phase=self.sine_phase
                )
                
                adjusted_coherence = base_coherence * resistance_factor
                
                coherence_scores.append(adjusted_coherence)
                novelty_scores.append(novelty)
                resistance_factors.append(resistance_factor)

            if len(coherence_scores) >= self.central_kernel.kernel_size:
                coherence_scores = self.central_kernel.process_scores(coherence_scores)

            modulated, eigmean, metrics = self.engine.reason_step(coherence_scores, input_vec)
            
            if len(modulated) != len(candidates):
                min_len = min(len(modulated), len(candidates))
                modulated = modulated[:min_len]
                candidates = candidates[:min_len]
            
            if not modulated or not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue
            
            if len(modulated) >= self.central_kernel.kernel_size:
                modulated = self.central_kernel.process_scores(modulated)
            
            probs = torch.softmax(torch.tensor(modulated), dim=0).numpy()
            
            if np.sum(probs) == 0:
                probs = np.ones(len(candidates)) / len(candidates)
            else:
                probs = probs / np.sum(probs)

            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            seed = tuple(output[-2:])
            step_count += 1

        return " ".join(output)


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n=== Eigenvalue-Isomorphic Neural Reasoner ===")
    print("    with Word Feature Approximation\n")
    
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus)
    print(f"Loaded {len(corpus):,} tokens, model size: {len(model):,}")

    generator = ReasoningGenerator(corpus, model, kernel_size=3)
    
    while True:
        seed = input("\nUSER: ")
        if seed.lower() in ['quit', 'exit']:
            break
            
        generated = generator.generate(seed, length=500)
        print("\n=== AI Response ===\n")
        print(generated)
        print(f"\n[Total: {len(generated.split())} words]")


if __name__ == "__main__":
    main()

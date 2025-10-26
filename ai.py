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
# PROPERTY CONSTRUCTOR
# ================================================================

class PropertyConstructor:
    """
    Constructs semantic properties and conceptual understanding from text.
    Uses word feature approximation to build property networks.
    """
    def __init__(self, word_approximator):
        self.word_approximator = word_approximator
        self.property_graph = defaultdict(dict)
        self.concept_embeddings = {}
        
    def extract_properties(self, text):
        """Extract semantic properties from generated text."""
        words = text.lower().split()
        
        properties = {
            'semantic_clusters': self._build_semantic_clusters(words),
            'syntactic_patterns': self._extract_syntactic_patterns(words),
            'thematic_coherence': self._measure_thematic_coherence(words),
            'entity_relations': self._extract_entity_relations(words)
        }
        
        return properties
    
    def _build_semantic_clusters(self, words, threshold=0.4):
        """Group words into semantic clusters based on feature similarity."""
        clusters = []
        visited = set()
        
        for i, word in enumerate(words):
            if word in visited:
                continue
                
            cluster = [word]
            visited.add(word)
            
            # Find similar words within context window
            context_window = words[max(0, i-10):min(len(words), i+10)]
            
            for other_word in context_window:
                if other_word not in visited:
                    similarity = self.word_approximator.compute_similarity(word, other_word)
                    if similarity > threshold:
                        cluster.append(other_word)
                        visited.add(other_word)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _extract_syntactic_patterns(self, words):
        """Identify recurring syntactic patterns via n-gram analysis."""
        patterns = defaultdict(int)
        
        # Bigram patterns
        for i in range(len(words) - 1):
            pattern = (words[i], words[i+1])
            patterns[pattern] += 1
        
        # Trigram patterns
        for i in range(len(words) - 2):
            pattern = (words[i], words[i+1], words[i+2])
            patterns[pattern] += 1
        
        # Return top patterns
        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:20]
    
    def _measure_thematic_coherence(self, words):
        """Measure how consistently themes are maintained across the text."""
        if len(words) < 20:
            return 0.5
        
        # Split into segments
        segment_size = 10
        segments = [words[i:i+segment_size] for i in range(0, len(words)-segment_size, segment_size)]
        
        # Measure inter-segment similarity
        coherence_scores = []
        for i in range(len(segments) - 1):
            seg1_words = set(segments[i])
            seg2_words = set(segments[i+1])
            
            # Calculate average similarity between segments
            similarities = []
            for w1 in list(seg1_words)[:5]:  # Sample to avoid O(n²)
                for w2 in list(seg2_words)[:5]:
                    similarities.append(self.word_approximator.compute_similarity(w1, w2))
            
            if similarities:
                coherence_scores.append(np.mean(similarities))
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _extract_entity_relations(self, words):
        """Extract entity relationships via co-occurrence patterns."""
        relations = []
        
        # Simple pattern: [entity1] [relation_verb] [entity2]
        for i in range(len(words) - 2):
            # Heuristic: longer words are entities
            if len(words[i]) > 3 and len(words[i+2]) > 3:
                relation = (words[i], words[i+1], words[i+2])
                relations.append(relation)
        
        return relations[:10]  # Return top relations
    
    def build_knowledge_graph(self, properties):
        """Construct a knowledge graph from extracted properties."""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add semantic clusters as nodes
        for cluster in properties['semantic_clusters']:
            node = {
                'id': f"cluster_{len(graph['nodes'])}",
                'words': cluster,
                'type': 'semantic_cluster'
            }
            graph['nodes'].append(node)
        
        # Add entity relations as edges
        for entity1, relation, entity2 in properties['entity_relations']:
            edge = {
                'source': entity1,
                'target': entity2,
                'relation': relation
            }
            graph['edges'].append(edge)
        
        return graph


# ================================================================
# ENVIRONMENT CORRELATOR
# ================================================================

class EnvironmentCorrelator:
    """
    Correlates environmental conditions with generation quality.
    Adapts generation parameters based on environmental feedback.
    """
    def __init__(self, reasoning_generator):
        self.generator = reasoning_generator
        self.environment_state = {
            'coherence_history': [],
            'novelty_history': [],
            'diversity_history': [],
            'quality_trend': 0.0
        }
        self.adaptation_threshold = 0.3
        
    def monitor_environment(self, generated_text):
        """Monitor environmental conditions during generation."""
        words = generated_text.split()
        
        # Measure current state
        coherence = self._measure_coherence(words)
        novelty = self._measure_novelty(words)
        diversity = len(set(words)) / len(words) if words else 0
        
        # Update history
        self.environment_state['coherence_history'].append(coherence)
        self.environment_state['novelty_history'].append(novelty)
        self.environment_state['diversity_history'].append(diversity)
        
        # Compute quality trend
        if len(self.environment_state['coherence_history']) >= 5:
            recent_quality = np.mean(self.environment_state['coherence_history'][-5:])
            previous_quality = np.mean(self.environment_state['coherence_history'][-10:-5]) if len(self.environment_state['coherence_history']) >= 10 else recent_quality
            self.environment_state['quality_trend'] = recent_quality - previous_quality
        
        return self.environment_state
    
    def _measure_coherence(self, words):
        """Measure coherence of generated text."""
        if len(words) < 5:
            return 0.5
        
        segments = [words[i:i+5] for i in range(0, min(len(words)-4, 20))]
        coherences = [
            self.generator.feature.extract_quantum_features(seg, self.generator.word_freq, self.generator.total_words)['coherence']
            for seg in segments
        ]
        return np.mean(coherences)
    
    def _measure_novelty(self, words):
        """Measure novelty of generated words."""
        if not words:
            return 0.5
        
        novelties = [self.generator.calculate_novelty(w) for w in words[:20]]
        return np.mean(novelties)
    
    def adapt_parameters(self):
        """Adapt generation parameters based on environment state."""
        adaptations = {}
        
        # If quality is declining, increase novelty exploration
        if self.environment_state['quality_trend'] < -self.adaptation_threshold:
            adaptations['sine_amp'] = min(0.8, self.generator.sine_amp + 0.1)
            adaptations['sine_freq'] = min(0.12, self.generator.sine_freq + 0.02)
            print("🔧 Environment declining - increasing exploration")
        
        # If quality is improving, exploit current strategy
        elif self.environment_state['quality_trend'] > self.adaptation_threshold:
            adaptations['sine_amp'] = max(0.4, self.generator.sine_amp - 0.05)
            adaptations['sine_freq'] = max(0.05, self.generator.sine_freq - 0.01)
            print("🎯 Environment improving - exploiting current strategy")
        
        # Apply adaptations
        for param, value in adaptations.items():
            setattr(self.generator, param, value)
        
        return adaptations
    
    def get_environment_summary(self):
        """Generate summary of environmental conditions."""
        if not self.environment_state['coherence_history']:
            return {
                'avg_coherence': 0.5,
                'avg_novelty': 0.5,
                'avg_diversity': 0.5,
                'quality_trend': 0.0,
                'status': 'initializing'
            }
        
        summary = {
            'avg_coherence': np.mean(self.environment_state['coherence_history']),
            'avg_novelty': np.mean(self.environment_state['novelty_history']),
            'avg_diversity': np.mean(self.environment_state['diversity_history']),
            'quality_trend': self.environment_state['quality_trend'],
            'status': 'improving' if self.environment_state['quality_trend'] > 0 else 'declining'
        }
        
        return summary


# ================================================================
# ENVIRONMENT CONTEXT INJECTOR
# ================================================================

class EnvironmentContextInjector:
    """
    Injects environmental context directly into user input.
    Fuses environment state with seed to create conditioned input.
    """
    def __init__(self, environment_correlator):
        self.environment_correlator = environment_correlator
        self.injection_strength = 0.5  # How much environment influences input
        
    def create_environment_tokens(self, env_state):
        """Convert environment state into token-like representations."""
        env_tokens = []
        
        # Coherence tokens
        coherence = env_state.get('avg_coherence', 0.5)
        if coherence > 0.7:
            env_tokens.append('__HIGH_COHERENCE__')
        elif coherence < 0.3:
            env_tokens.append('__LOW_COHERENCE__')
        else:
            env_tokens.append('__MID_COHERENCE__')
        
        # Novelty tokens
        novelty = env_state.get('avg_novelty', 0.5)
        if novelty > 0.6:
            env_tokens.append('__NOVEL__')
        elif novelty < 0.4:
            env_tokens.append('__FAMILIAR__')
        
        # Diversity tokens
        diversity = env_state.get('avg_diversity', 0.5)
        if diversity > 0.6:
            env_tokens.append('__DIVERSE__')
        elif diversity < 0.4:
            env_tokens.append('__REPETITIVE__')
        
        # Trend tokens
        trend = env_state.get('quality_trend', 0.0)
        if trend > 0.1:
            env_tokens.append('__IMPROVING__')
        elif trend < -0.1:
            env_tokens.append('__DECLINING__')
        
        return env_tokens
    
    def inject_environment_into_seed(self, seed, env_state):
        """Inject environment context directly into seed input."""
        # Get environment tokens
        env_tokens = self.create_environment_tokens(env_state)
        
        # Create conditioned seed: [ENV_CONTEXT] + [USER_SEED]
        env_prefix = ' '.join(env_tokens)
        conditioned_seed = f"{env_prefix} {seed}"
        
        print(f"\n🔬 Environment Injection:")
        print(f"   Original Seed: {seed}")
        print(f"   Environment Context: {env_prefix}")
        print(f"   Conditioned Input: {conditioned_seed}")
        
        return conditioned_seed
    
    def create_environment_vector(self, env_state):
        """Create dense vector representation of environment state."""
        vec = np.array([
            env_state.get('avg_coherence', 0.5),
            env_state.get('avg_novelty', 0.5),
            env_state.get('avg_diversity', 0.5),
            env_state.get('quality_trend', 0.0) + 0.5,  # Normalize to [0, 1]
        ])
        
        return vec
    
    def modulate_input_vector(self, input_vector, env_state):
        """Modulate the input vector with environment state."""
        env_vec = self.create_environment_vector(env_state)
        
        # Ensure same dimensionality
        if len(input_vector) != len(env_vec):
            # Pad or truncate
            min_len = min(len(input_vector), len(env_vec))
            env_vec = env_vec[:min_len]
            input_vector = input_vector[:min_len]
        
        # Context-gating: weighted fusion
        modulated = (1 - self.injection_strength) * input_vector + self.injection_strength * env_vec
        
        return modulated


# ================================================================
# PERFECTION DETECTOR
# ================================================================

class PerfectionDetector:
    """
    Detects states of 'perfection' in generation output and triggers
    self-modification of system properties.
    """
    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        self.quality_history = []
        self.perfection_events = []
        self.baseline_metrics = None
        
    def detect_perfection(self, output_metrics):
        """Detect if current output represents 'perfection'."""
        # Composite quality score
        quality_score = (
            0.4 * output_metrics.get('avg_coherence', 0) +
            0.3 * output_metrics.get('thematic_coherence', 0) +
            0.2 * output_metrics.get('avg_diversity', 0) +
            0.1 * (1.0 - abs(output_metrics.get('avg_novelty', 0.5) - 0.5) * 2)
        )
        
        self.quality_history.append(quality_score)
        
        # Need sufficient history to establish baseline
        if len(self.quality_history) < 20:
            return False, quality_score
        
        # Calculate dynamic threshold
        threshold = np.percentile(self.quality_history, self.threshold_percentile)
        
        # Perfection: current quality exceeds 95th percentile
        is_perfect = quality_score > threshold
        
        if is_perfect:
            perfection_event = {
                'timestamp': datetime.now(),
                'quality_score': quality_score,
                'threshold': threshold,
                'metrics': output_metrics.copy()
            }
            self.perfection_events.append(perfection_event)
            print(f"\n✨ PERFECTION DETECTED! Score: {quality_score:.4f} (threshold: {threshold:.4f})")
        
        return is_perfect, quality_score
    
    def get_perfection_patterns(self):
        """Analyze perfection events to extract common patterns."""
        if len(self.perfection_events) < 3:
            return None
        
        # Extract metrics from perfection events
        perfect_coherences = [e['metrics'].get('avg_coherence', 0) for e in self.perfection_events]
        perfect_novelties = [e['metrics'].get('avg_novelty', 0) for e in self.perfection_events]
        perfect_diversities = [e['metrics'].get('avg_diversity', 0) for e in self.perfection_events]
        
        patterns = {
            'optimal_coherence': np.mean(perfect_coherences),
            'optimal_novelty': np.mean(perfect_novelties),
            'optimal_diversity': np.mean(perfect_diversities),
            'coherence_std': np.std(perfect_coherences),
            'novelty_std': np.std(perfect_novelties),
            'diversity_std': np.std(perfect_diversities)
        }
        
        return patterns


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
# ENVIRONMENT-AWARE GENERATOR
# ================================================================

class EnvironmentAwareGenerator:
    """
    Enhanced generator with environment injection at every step.
    No switches - both environment and seed always active.
    """
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
        self.central_kernel = CentralKernel(kernel_size=kernel_size)
        
        # Initialize reasoning engine with kernel
        self.engine = ReasoningEngine(kernel=self.central_kernel)
        
        # Sine resistance parameters
        self.sine_freq = 0.08
        self.sine_amp = 0.6
        self.sine_phase = 0.0
        
        # Environment injector (set externally)
        self.env_injector = None
        
        print(f"🤖 Generator ready with kernel and word approximation!")

    def calculate_novelty(self, word):
        freq = self.word_freq.get(word, 1)
        novelty = 1.0 - np.log(freq + 1) / np.log(self.total_words + 1)
        return float(np.clip(novelty, 0, 1))

    def generate_with_environment_injection(self, seed, env_state, length=200):
        """Generate with continuous environment injection."""
        # Inject environment into seed
        if self.env_injector and env_state:
            conditioned_seed = self.env_injector.inject_environment_into_seed(seed, env_state)
        else:
            conditioned_seed = seed
        
        seed_words = conditioned_seed.lower().split()
        
        # Find valid seed from conditioned input
        valid_seed = None
        for i in range(len(seed_words) - 1):
            potential_seed = tuple(seed_words[i:i+2])
            if potential_seed in self.model:
                valid_seed = potential_seed
                break
        
        if not valid_seed:
            # Fallback to original seed words (skip env tokens)
            original_words = seed.lower().split()[:2]
            while len(original_words) < 2:
                original_words.append(self.tokens[len(original_words) % len(self.tokens)])
            valid_seed = tuple(original_words)
            
            if valid_seed not in self.model:
                valid_seed = self.keys[np.random.randint(len(self.keys))]
        
        output = list(valid_seed)
        
        print(f"\n🌀 Generating {length} words with environment injection...")
        print(f"   Effective Seed: {' '.join(valid_seed)}\n")
        
        step_count = 0
        
        while len(output) < length:
            # Create base input vector
            recent_text = ' '.join(output[-4:]) if len(output) >= 4 else ' '.join(output)
            input_vec = np.array([ord(c) % 97 / 25 for c in recent_text.ljust(4)[:4]])
            
            # INJECT ENVIRONMENT: Modulate input vector with environment state
            if self.env_injector and env_state:
                input_vec = self.env_injector.modulate_input_vector(input_vec, env_state)
            
            # Process through kernel
            input_vec = self.central_kernel.process_vector(input_vec)

            seed = tuple(output[-2:])
            candidates = self.model.get(seed, [])
            candidates = [w for w in candidates if any(c.isalnum() for c in w)]
            
            if not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue

            coherence_scores = []
            
            for cand in candidates:
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
                
                # ENVIRONMENT MODULATION: Adjust scores based on environment
                if self.env_injector and env_state:
                    # If environment is declining, boost novelty
                    if env_state.get('quality_trend', 0) < -0.1:
                        novelty = novelty * 1.3
                    # If environment is improving, boost coherence
                    elif env_state.get('quality_trend', 0) > 0.1:
                        base_coherence = base_coherence * 1.2
                
                resistance_factor = sine_resistance(
                    step_count, 
                    novelty, 
                    freq=self.sine_freq, 
                    amp=self.sine_amp, 
                    phase=self.sine_phase
                )
                
                adjusted_coherence = base_coherence * resistance_factor
                coherence_scores.append(adjusted_coherence)

            if len(coherence_scores) >= self.central_kernel.kernel_size:
                coherence_scores = self.central_kernel.process_scores(coherence_scores)

            # ENVIRONMENT INJECTION: Pass environment-modulated input to reasoning
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
            step_count += 1

        return " ".join(output)


# ================================================================
# SIMULTANEOUS ENVIRONMENT SYSTEM
# ================================================================

class SimultaneousEnvironmentSystem:
    """
    System where environment correlation and generation happen simultaneously.
    No switches - both always active together. Self-modifies when perfection detected.
    """
    def __init__(self, generator, property_constructor, environment_correlator):
        self.generator = generator
        self.property_constructor = property_constructor
        self.environment_correlator = environment_correlator
        self.perfection_detector = PerfectionDetector()
        
        # Create environment injector
        self.env_injector = EnvironmentContextInjector(environment_correlator)
        
        # Link injector to generator
        self.generator.env_injector = self.env_injector
        
        # Baseline parameters for self-modification
        self.baseline_params = {
            'sine_amp': generator.sine_amp,
            'sine_freq': generator.sine_freq,
            'sine_phase': generator.sine_phase,
        }
        
        self.meta_learning_rate = 0.1
        self.modification_history = []
        
    def generate_with_simultaneous_correlation(self, seed, length=200):
        """Generate text with environment correlation happening simultaneously."""
        print(f"\n{'='*70}")
        print(f"SIMULTANEOUS ENVIRONMENT CORRELATION SYSTEM")
        print(f"Environment + Seed Injection Active Throughout Generation")
        print(f"{'='*70}\n")
        
        # Get current environment state
        env_summary = self.environment_correlator.get_environment_summary()
        
        # Phase 1: Simultaneous Generation with Environment Injection
        print("🔄 Phase 1: Simultaneous Generation + Environment Correlation")
        print("   ├─ User Seed: Active ✓")
        print("   ├─ Environment Context: Active ✓")
        print("   └─ Continuous Fusion: Enabled ✓\n")
        
        generated_text = self.generator.generate_with_environment_injection(
            seed, 
            env_summary, 
            length=length
        )
        
        # Phase 2: Update Environment State
        print("\n🌍 Phase 2: Environment State Update")
        env_state = self.environment_correlator.monitor_environment(generated_text)
        updated_summary = self.environment_correlator.get_environment_summary()
        
        print(f"   Coherence: {updated_summary['avg_coherence']:.3f}")
        print(f"   Novelty: {updated_summary['avg_novelty']:.3f}")
        print(f"   Diversity: {updated_summary['avg_diversity']:.3f}")
        print(f"   Trend: {updated_summary['status']}")
        
        # Phase 3: Extract Properties
        print("\n🔍 Phase 3: Property Construction")
        properties = self.property_constructor.extract_properties(generated_text)
        
        print(f"   Semantic Clusters: {len(properties['semantic_clusters'])}")
        print(f"   Thematic Coherence: {properties['thematic_coherence']:.3f}")
        
        # Phase 4: Knowledge Graph
        print("\n🕸️  Phase 4: Knowledge Graph Construction")
        knowledge_graph = self.property_constructor.build_knowledge_graph(properties)
        print(f"   Nodes: {len(knowledge_graph['nodes'])}")
        print(f"   Edges: {len(knowledge_graph['edges'])}")
        
        # Phase 5: Perfection Detection + Self-Modification
        print("\n✨ Phase 5: Perfection Detection")
        
        output_metrics = {
            'avg_coherence': updated_summary['avg_coherence'],
            'avg_novelty': updated_summary['avg_novelty'],
            'avg_diversity': updated_summary['avg_diversity'],
            'thematic_coherence': properties['thematic_coherence']
        }
        
        is_perfect, quality_score = self.perfection_detector.detect_perfection(output_metrics)
        
        modified = False
        if is_perfect:
            patterns = self.perfection_detector.get_perfection_patterns()
            if patterns:
                print("\n🧬 Phase 6: Self-Modification Triggered")
                modifications = self._compute_modifications(patterns)
                self._apply_modifications(modifications)
                
                # Optimize kernel if perfection detected
                self._optimize_kernel_properties(patterns)
                modified = True
        
        if not modified:
            # Adaptive parameter tuning
            print("\n⚙️  Phase 6: Adaptive Parameter Tuning")
            adaptations = self.environment_correlator.adapt_parameters()
        else:
            adaptations = {}
        
        # Compile output
        output = {
            'generated_text': generated_text,
            'environment_state': updated_summary,
            'properties': properties,
            'knowledge_graph': knowledge_graph,
            'adaptations': adaptations,
            'self_modifications': modified,
            'quality_score': quality_score,
            'perfection_detected': is_perfect
        }
        
        return output
    
    def _compute_modifications(self, patterns):
        """Compute parameter modifications from perfection patterns."""
        modifications = {}
        current_params = {
            'sine_amp': self.generator.sine_amp,
            'sine_freq': self.generator.sine_freq,
            'sine_phase': self.generator.sine_phase,
        }
        
        # Adjust based on patterns
        if patterns['optimal_coherence'] > 0.7:
            new_amp = current_params['sine_amp'] * (1 - self.meta_learning_rate * 0.3)
            modifications['sine_amp'] = np.clip(new_amp, 0.3, 0.9)
        elif patterns['optimal_coherence'] < 0.5:
            new_amp = current_params['sine_amp'] * (1 + self.meta_learning_rate * 0.5)
            modifications['sine_amp'] = np.clip(new_amp, 0.3, 0.9)
        
        if patterns['optimal_diversity'] > 0.6:
            new_freq = current_params['sine_freq'] * (1 + self.meta_learning_rate * 0.4)
            modifications['sine_freq'] = np.clip(new_freq, 0.04, 0.15)
        
        optimal_novelty = patterns['optimal_novelty']
        if abs(optimal_novelty - 0.5) < 0.1:
            phase_shift = 0.1 * np.sin(optimal_novelty * np.pi)
            new_phase = (current_params['sine_phase'] + phase_shift) % (2 * np.pi)
            modifications['sine_phase'] = new_phase
        
        return modifications
    
    def _apply_modifications(self, modifications):
        """Apply self-modifications."""
        for param, value in modifications.items():
            old_value = getattr(self.generator, param)
            setattr(self.generator, param, value)
            print(f"   ✅ {param}: {old_value:.4f} → {value:.4f}")
            
            self.modification_history.append({
                'timestamp': datetime.now(),
                'parameter': param,
                'old_value': old_value,
                'new_value': value
            })
    
    def _optimize_kernel_properties(self, patterns):
        """Optimize central kernel properties based on perfection patterns."""
        print("\n🔧 Optimizing Kernel Properties...")
        
        coherence_std = patterns.get('coherence_std', 0)
        
        if coherence_std < 0.05:
            # Low variance - tighter kernel works
            new_kernel_size = max(3, self.generator.central_kernel.kernel_size - 2)
            print(f"   🎯 Tightening kernel: {self.generator.central_kernel.kernel_size} → {new_kernel_size}")
        elif coherence_std > 0.15:
            # High variance - broader smoothing needed
            new_kernel_size = min(7, self.generator.central_kernel.kernel_size + 2)
            print(f"   📊 Broadening kernel: {self.generator.central_kernel.kernel_size} → {new_kernel_size}")
        else:
            return
        
        # Recreate kernel with new size
        self.generator.central_kernel = CentralKernel(kernel_size=new_kernel_size)
        self.generator.engine.kernel = self.generator.central_kernel
        self.generator.engine.truth_washer.kernel = self.generator.central_kernel
        self.generator.engine.eigen_system.kernel = self.generator.central_kernel
        
        print("   ✅ Kernel properties updated")
    
    def display_output(self, output):
        """Display comprehensive output."""
        print(f"\n{'='*70}")
        print("GENERATED TEXT")
        print(f"{'='*70}\n")
        print(output['generated_text'])
        
        print(f"\n{'='*70}")
        print("ENVIRONMENT-CONDITIONED ANALYSIS")
        print(f"{'='*70}\n")
        
        # Quality metrics
        print("📊 Quality Metrics:")
        print(f"   Overall Score: {output['quality_score']:.4f}")
        if output['perfection_detected']:
            print(f"   Status: ✨ PERFECTION DETECTED ✨")
        else:
            print(f"   Status: Standard generation")
        
        # Semantic clusters
        print("\n🔹 Semantic Clusters:")
        for i, cluster in enumerate(output['properties']['semantic_clusters'][:3]):
            print(f"   Cluster {i+1}: {', '.join(cluster[:5])}")
        
        # Entity relations
        print("\n🔹 Entity Relations:")
        for e1, rel, e2 in output['properties']['entity_relations'][:5]:
            print(f"   {e1} --[{rel}]--> {e2}")
        
        # Self-modification summary
        if output['self_modifications']:
            print(f"\n🧬 Self-Modifications:")
            print(f"   Total Perfection Events: {len(self.perfection_detector.perfection_events)}")
            print(f"   Total Modifications: {len(self.modification_history)}")
        
        print(f"\n{'='*70}\n")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*70)
    print("SIMULTANEOUS ENVIRONMENT CORRELATION SYSTEM")
    print("Environment + Seed Always Active Together")
    print("Self-Modifying When Perfection Detected")
    print("="*70 + "\n")
    
    # Load corpus
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus, n=2)
    print(f"📚 Loaded {len(corpus):,} tokens, model size: {len(model):,}")

    # Initialize components
    print("\n🔧 Initializing simultaneous correlation system...")
    
    # Use environment-aware generator
    generator = EnvironmentAwareGenerator(corpus, model, kernel_size=3)
    property_constructor = PropertyConstructor(generator.word_approximator)
    environment_correlator = EnvironmentCorrelator(generator)
    
    # Create simultaneous system
    system = SimultaneousEnvironmentSystem(
        generator, 
        property_constructor, 
        environment_correlator
    )
    
    print("✅ Simultaneous correlation system ready!")
    print("💡 Environment context injected into every generation step")
    print("🧬 Self-modification enabled for perfection events\n")
    
    # Interactive loop
    generation_count = 0
    while True:
        seed = input("\nUSER SEED: ")
        if seed.lower() in ['quit', 'exit']:
            break
        
        generation_count += 1
        print(f"\n[Generation #{generation_count}]")
        
        # Generate with simultaneous environment correlation
        output = system.generate_with_simultaneous_correlation(seed, length=200)
        
        # Display results
        system.display_output(output)
        
        # Show environment injection details
        print("\n🔬 Environment Injection Details:")
        env_state = output['environment_state']
        env_vec = system.env_injector.create_environment_vector(env_state)
        print(f"   Environment Vector: {env_vec}")
        print(f"   Injection Strength: {system.env_injector.injection_strength:.2f}")
        print(f"   Context Tokens: {system.env_injector.create_environment_tokens(env_state)}")
        
        # Modification history every 5 generations
        if generation_count % 5 == 0 and system.modification_history:
            view = input("\nView modification history? (y/n): ").strip().lower()
            if view == 'y':
                print("\n📊 MODIFICATION HISTORY:")
                for i, mod in enumerate(system.modification_history[-10:]):
                    print(f"   {i+1}. {mod['timestamp'].strftime('%H:%M:%S')} - "
                          f"{mod['parameter']}: {mod['old_value']:.4f} → {mod['new_value']:.4f}")


if __name__ == "__main__":
    main()

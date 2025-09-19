import random
import math
import numpy as np
from collections import defaultdict
from datetime import datetime

KB_LEN = 99999

# -------------------------
# Cybernetic Feedback Module
# -------------------------
class CyberneticFeedbackModule:
    """
    Implements a cybernetic control loop for adjusting word transition weights.
    - error_signal: deviation between expected and actual transitions
    - feedback_gain: scales corrective adjustment
    - stability_margin: prevents runaway amplification
    - adaptation_rate: how quickly the system adapts to new signals
    """
    def __init__(self, feedback_gain=0.3, stability_margin=0.1, adaptation_rate=0.05):
        self.feedback_gain = feedback_gain
        self.stability_margin = stability_margin
        self.adaptation_rate = adaptation_rate
        self.error_history = []

    def compute_feedback(self, predicted_prob, actual_prob):
        error = actual_prob - predicted_prob
        self.error_history.append(error)
        correction = self.feedback_gain * error
        correction = max(min(correction, 1 - self.stability_margin), -1 + self.stability_margin)
        return correction

    def adapt_gain(self):
        """Adaptive tuning: if errors oscillate too much, lower gain."""
        if len(self.error_history) < 5:
            return
        variance = np.var(self.error_history[-5:])
        if variance > 0.05:  # too unstable
            self.feedback_gain *= (1 - self.adaptation_rate)
        else:  # stable, can increase slightly
            self.feedback_gain *= (1 + self.adaptation_rate)


# -------------------------
# Neural Enhanced Markov
# -------------------------
class NeuralEnhancedMarkov:
    """
    Markov text generator with cybernetic control.
    - Uses feedback loops to regulate probability distributions.
    - Tracks entropy, complexity, and generation history.
    """

    def __init__(self, convergence_threshold=1e-6):
        self.convergence_threshold = convergence_threshold
        self.model = {}
        self.series_sum = 0
        self.word_frequencies = defaultdict(int)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.generation_history = []
        self.performance_metrics = {}
        self.seed_word = None
        self.spike_patterns = defaultdict(list)

        # Cybernetic module
        self.cybernetics = CyberneticFeedbackModule()
    
    def simulate_neural_spike_patterns(self):
        for word, frequency in self.word_frequencies.items():
            base_rate = min(frequency / max(self.word_frequencies.values()), 1.0)
            spikes = [t for t in range(100) if random.random() < base_rate * 0.1]
            self.spike_patterns[word] = spikes
    
    def build_semantic_clusters(self):
        co_occurrence = defaultdict(lambda: defaultdict(float))
        for word, transitions in self.transition_matrix.items():
            for next_word, count in transitions.items():
                co_occurrence[word][next_word] += count
                co_occurrence[next_word][word] += count * 0.5
        self.semantic_clusters = {}
        processed_words = set()
        for word in self.vocabulary:
            if word in processed_words: continue
            cluster = [word]
            word_neighbors = set(co_occurrence[word].keys())
            for other_word in self.vocabulary:
                if other_word != word and other_word not in processed_words:
                    other_neighbors = set(co_occurrence[other_word].keys())
                    intersection = len(word_neighbors & other_neighbors)
                    union = len(word_neighbors | other_neighbors)
                    similarity = intersection / union if union > 0 else 0
                    if similarity > 0.8:
                        cluster.append(other_word)
                        processed_words.add(other_word)
            if len(cluster) > 1:
                cluster_name = f"cluster_{len(self.semantic_clusters)}"
                self.semantic_clusters[cluster_name] = cluster
            processed_words.add(word)
    # -------------------------
    # Seed control
    # -------------------------
    def set_seed(self, seed_input):
        processed_seed = seed_input.strip().lower()
        if not processed_seed:
            self.seed_word = None
            print("Seed cleared. Using random start.")
            return
        seed_words = processed_seed.split()
        if all(word in self.vocabulary for word in seed_words):
            self.seed_word = seed_words if len(seed_words) > 1 else seed_words[0]
            print(f"Seed set to: '{seed_input.strip()}'")
        else:
            self.seed_word = None
            print(f"Warning: The seed phrase '{seed_input.strip()}' contains word(s) not found in vocabulary. Using random start.")

    # -------------------------
    # Model building
    # -------------------------
    def build_enhanced_model(self, text):
        processed_text = text.lower()
        words = processed_text.split()
        if len(words) < 2:
            return
        self.vocabulary.update(words)
        for i in range(len(words) - 1):
            current, nxt = words[i], words[i + 1]
            self.transition_matrix[current][nxt] += 1
            self.word_frequencies[current] += 1
        self.series_sum = self._calculate_infinite_series()
        self.model = self._build_weighted_model()
        self._calculate_performance_metrics()

    def _calculate_infinite_series(self):
        """Simplified weighting — frequency-based only."""
        total_sum = 0
        for depth in range(1, 20):  # fixed recursion depth
            level_sum = 0
            for word, next_words in self.transition_matrix.items():
                word_freq = self.word_frequencies[word]
                for nxt, count in next_words.items():
                    if word_freq == 0:
                        continue
                    contrib = (word_freq * count) / (depth + 1)
                    level_sum += contrib
            total_sum += level_sum * (depth + 1) ** -1.2
        return max(total_sum, 1e-6)

    def _build_weighted_model(self):
        weighted_model = {}
        epsilon = 1e-10
        for word, next_words in self.transition_matrix.items():
            word_freq = self.word_frequencies[word]
            transitions, weights = [], []
            for nxt, count in next_words.items():
                base_weight = epsilon * self.series_sum
                freq_weight = (word_freq * count) / (word_freq + 1e-9)
                rarity_penalty = 1.0 / (count + 1)
                weight = max(base_weight, freq_weight) * (1 - 0.8 * rarity_penalty)
                transitions.append(nxt)
                weights.append(weight)
            total = sum(weights)
            normalized = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

            # Apply cybernetic correction to weights
            corrected_weights = []
            for prob in normalized:
                correction = self.cybernetics.compute_feedback(prob, 1.0 / len(normalized))
                corrected = prob + correction
                corrected_weights.append(max(corrected, 0.0))

            total_corrected = sum(corrected_weights)
            final_weights = [w / total_corrected for w in corrected_weights] if total_corrected > 0 else normalized

            weighted_model[word] = list(zip(transitions, final_weights))
        return weighted_model

    # -------------------------
    # Metrics
    # -------------------------
    def _calculate_performance_metrics(self):
        self.performance_metrics = {
            'vocabulary_size': len(self.vocabulary),
            'unique_transitions': sum(len(v) for v in self.model.values()),
            'avg_transitions_per_word': sum(len(v) for v in self.model.values()) / len(self.model) if self.model else 0,
            'series_convergence_sum': self.series_sum,
            'model_complexity': len(self.vocabulary) * self.series_sum,
            'entropy': self._calculate_entropy(),
            'build_timestamp': datetime.now().isoformat(),
            'cybernetic_gain': self.cybernetics.feedback_gain
        }

    def _calculate_entropy(self):
        total_entropy = 0
        for word, transitions in self.model.items():
            for _, prob in transitions:
                if prob > 0:
                    total_entropy += -prob * math.log2(prob)
        return total_entropy / len(self.model) if self.model else 0

    # -------------------------
    # Text generation
    # -------------------------
    def generate_with_neural_enhancement(self, max_words=500, neural_strength=0.3, start_word=None):
        if not self.model:
            return "No model available."

        if start_word is None and self.seed_word is not None:
            start_word = self.seed_word

        if isinstance(start_word, list):
            sentence = [word.capitalize() for word in start_word]
            current_word = start_word[-1]
        else:
            current_word = start_word.lower() if start_word and start_word.lower() in self.model else random.choice(list(self.model.keys()))
            sentence = [current_word.capitalize()]

        generation_path = [current_word]

        for _ in range(max_words - 1):
            if current_word in self.model and self.model[current_word]:
                words, weights = zip(*self.model[current_word])
                if random.random() < neural_strength:
                    inverse = [(1.0 - w) * w for w in weights]
                    total_inv = sum(inverse)
                    creativity_weights = [w / total_inv for w in inverse] if total_inv > 0 else weights
                    next_word = np.random.choice(words, p=creativity_weights)
                else:
                    next_word = np.random.choice(words, p=weights)
                sentence.append(next_word)
                generation_path.append(next_word)
                current_word = next_word
            else:
                break

        generated_text = " ".join(sentence)
        if generated_text[-1] not in ".?!":
            generated_text += "."

        # Adapt cybernetic feedback gain dynamically
        self.cybernetics.adapt_gain()

        self.generation_history.append({
            'text': generated_text,
            'path': generation_path,
            'neural_strength': neural_strength,
            'length': len(sentence),
            'timestamp': datetime.now().isoformat()
        })

        return generated_text


# Usage Example
with open(input("Filename: "), 'r', encoding='utf-8') as f:
    corpus = f.read()[:KB_LEN]
    generator = NeuralEnhancedMarkov()
    generator.build_enhanced_model(corpus)
    generator.build_semantic_clusters()
    generator.simulate_neural_spike_patterns()
    while True:
        generator.set_seed(input("USER: "))
        print(f"Neural Enhanced: {generator.generate_with_neural_enhancement(max_words=550, neural_strength=0.8)}")

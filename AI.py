import random

import math
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
KB_LEN = 99999
class NeuralEnhancedMarkov:
    """
    Advanced Markov text generator that integrates mathematical weighting,
    semantic clustering, spike-pattern simulation, and context-aware generation.
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
        self.n_gram_model = defaultdict(lambda: defaultdict(int))
        self.semantic_clusters = {}
        self.spike_patterns = defaultdict(list)
        self.seed_word = None

    def set_seed(self, seed_input):
        """
        Set the seed word(s) for text generation.
        Handles both single words and multi-word phrases.
        """
        processed_seed = seed_input.strip().lower()
        if not processed_seed:
            self.seed_word = None
            print("Seed cleared. Using random start.")
            return

        # Split the input into words
        seed_words = processed_seed.split()
        
        # Check if all words in the phrase are in the vocabulary
        if all(word in self.vocabulary for word in seed_words):
            # Set the seed to a list if it's a phrase, otherwise a single word
            self.seed_word = seed_words if len(seed_words) > 1 else seed_words[0]
            print(f"Seed set to: '{seed_input.strip()}'")
        else:
            # If any word is not found, print a warning and use random start
            self.seed_word = None
            print(f"Warning: The seed phrase '{seed_input.strip()}' contains word(s) not found in vocabulary. Using random start.")


    def build_enhanced_model(self, text):
        processed_text = text.lower()
        words = processed_text.split()
        if len(words) < 2: return
        self.vocabulary.update(words)
        for i in range(len(words) - 1):
            current, next = words[i], words[i + 1]
            self.transition_matrix[current][next] += 1
            self.word_frequencies[current] += 1
        self.series_sum = self._calculate_infinite_series()
        self.model = self._build_weighted_model()
        self._calculate_performance_metrics()

    def _calculate_infinite_series(self):
        total_sum = 0
        n = 0
        while n < 1000:
            term_sum = 0
            for word, next_words in self.transition_matrix.items():
                i = self.word_frequencies[next_words[0]]
                for next_word, count in next_words.items():
                    j = count
                    term_sum += (i * j) / (n + 1) ** 0.5
            total_sum += term_sum
            n += 1
        return total_sum if total_sum > 0 else 1.0

    def _build_weighted_model(self):
        weighted_model = {}
        epsilon = 1e-10
        for word, next_words in self.transition_matrix.items():
            word_freq = self.word_frequencies[word]
            transitions, weights = [], []
            for next_word, count in next_words.items():
                i, j = word_freq, count
                base_weight = epsilon * self.series_sum
                freq_weight = (i * j * count) / word_freq
                rarity_penalty = 1.0 / (count + 1)
                weight = max(base_weight, freq_weight) * (1 - rarity_penalty * 0.1)
                transitions.append(next_word)
                weights.append(weight)
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)
            weighted_model[word] = list(zip(transitions, normalized_weights))
        return weighted_model

    def _calculate_performance_metrics(self):
        self.performance_metrics = {
            'vocabulary_size': len(self.vocabulary),
            'unique_transitions': sum(len(transitions) for transitions in self.model.values()),
            'avg_transitions_per_word': sum(len(transitions) for transitions in self.model.values()) / len(self.model) if self.model else 0,
            'series_convergence_sum': self.series_sum,
            'model_complexity': len(self.vocabulary) * self.series_sum,
            'entropy': self._calculate_entropy(),
            'build_timestamp': datetime.now().isoformat()
        }

    def _calculate_entropy(self):
        total_entropy = 0
        for word, transitions in self.model.items():
            word_entropy = 0
            for _, prob in transitions:
                if prob > 0:
                    word_entropy += -prob * math.log2(prob)
            total_entropy += word_entropy
        return total_entropy / len(self.model) if self.model else 0

    def generate_text(self, max_words=20, creativity_factor=0.3, start_word=None):
        if not self.model: return "No model available."
        
        # Use seed word if available and no start_word specified
        if start_word is None and self.seed_word is not None:
            start_word = self.seed_word
            
        current_word = start_word.lower() if start_word and start_word.lower() in self.model else random.choice(list(self.model.keys()))
        while not current_word[0].isalpha():
            current_word = random.choice(list(self.model.keys()))
        sentence = [current_word.capitalize()]
        generation_path = [current_word]
        for _ in range(max_words - 1):
            if current_word in self.model and self.model[current_word]:
                words, weights = zip(*self.model[current_word])
                if random.random() < creativity_factor:
                    inverse_weights = [(1.0 - w) * w for w in weights]
                    total_inverse = sum(inverse_weights)
                    creativity_weights = [w / total_inverse for w in inverse_weights] if total_inverse > 0 else weights
                    next_word = np.random.choice(words, p=creativity_weights)
                else:
                    next_word = np.random.choice(words, p=weights)
                sentence.append(next_word)
                generation_path.append(next_word)
                current_word = next_word
            else:
                break
        generated_text = " ".join(sentence)
        if generated_text[-1] not in ".?!": generated_text += "."
        self.generation_history.append({
            'text': generated_text, 'path': generation_path, 'creativity_factor': creativity_factor,
            'length': len(sentence), 'timestamp': datetime.now().isoformat()
        })
        return generated_text

    def add_n_gram_enhancement(self, n=2):
        if n < 2: return
        words = list(self.vocabulary)
        for i in range(len(words) - n):
            n_gram = tuple(words[i:i+n])
            next_word = words[i+n]
            self.n_gram_model[n_gram][next_word] += 1

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

    def simulate_neural_spike_patterns(self):
        for word, frequency in self.word_frequencies.items():
            base_rate = min(frequency / max(self.word_frequencies.values()), 1.0)
            spikes = [t for t in range(100) if random.random() < base_rate * 0.1]
            self.spike_patterns[word] = spikes

    def neural_weighted_selection(self, current_word, available_words, weights):
        if current_word not in self.spike_patterns:
            return np.random.choice(available_words, p=weights)
        current_spikes = self.spike_patterns[current_word]
        neural_weights = []
        for word in available_words:
            if word in self.spike_patterns:
                target_spikes = self.spike_patterns[word]
                synchrony = sum(1 for st in current_spikes for tt in target_spikes if abs(st-tt)<=5)
                max_sync = min(len(current_spikes), len(target_spikes))
                sync_weight = synchrony / max_sync if max_sync > 0 else 0
            else:
                sync_weight = 0.1
            neural_weights.append(sync_weight)
        if sum(neural_weights) > 0:
            combined_weights = [w * 0.7 + nw * 0.3 for w, nw in zip(weights, neural_weights)]
            total = sum(combined_weights)
            final_weights = [w / total for w in combined_weights] if total > 0 else weights
        else:
            final_weights = weights
        return np.random.choice(available_words, p=final_weights)

    def generate_with_neural_enhancement(self, max_words=20, neural_strength=0.5):
        if not self.model: return "No model available."
        if not self.semantic_clusters: self.build_semantic_clusters()
        if not self.spike_patterns: self.simulate_neural_spike_patterns()
        
        # Handle multi-word seeds
        if self.seed_word is not None:
            if isinstance(self.seed_word, list):
                sentence = [self.seed_word[0].capitalize()] + self.seed_word[1:]
                current_word = self.seed_word[-1].lower()
            else:
                current_word = self.seed_word
                sentence = [current_word.capitalize()]
        else:
            current_word = random.choice(list(self.model.keys()))
            sentence = [current_word.capitalize()]
        
        for _ in range(max_words - len(sentence)):
            if current_word in self.model and self.model[current_word]:
                words, weights = zip(*self.model[current_word])
                next_word = self.neural_weighted_selection(current_word, words, weights) if random.random()<neural_strength else np.random.choice(words, p=weights)
                sentence.append(next_word)
                current_word = next_word
            else:
                break
        generated_text = " ".join(sentence)
        if generated_text[-1] not in ".?!": generated_text += "."
        return generated_text

# Usage Example
with open(input("Filename: "), 'r', encoding='utf-8') as f:
    corpus = f.read()[:KB_LEN]

while True:

    generator = NeuralEnhancedMarkov()
    generator.build_enhanced_model(corpus)
    generator.add_n_gram_enhancement(n=2)
    generator.build_semantic_clusters()
    generator.simulate_neural_spike_patterns()
    generator.set_seed(input("USER: "))

    print(f"Neural Enhanced: {generator.generate_with_neural_enhancement(max_words=550, neural_strength=0.8)}")

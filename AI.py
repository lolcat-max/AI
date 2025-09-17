import random
import math
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
from copy import deepcopy

KB_LEN = 99999

def validate_probabilities(probs, words=None):
    """Validate and fix probability arrays to prevent NaN errors."""
    probs = np.array(probs, dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.maximum(probs, 1e-10)
    total = np.sum(probs)
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    if not np.all(np.isfinite(probs)) or not np.isclose(np.sum(probs), 1.0):
        probs = np.ones(len(probs)) / len(probs)
    return probs

class NeuralEnhancedMarkov:
    """
    Advanced Markov text generator with catastrophic forgetting simulation
    and feed-forward memory consolidation mechanisms.
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
        self.semantic_clusters = {}
        self.spike_patterns = defaultdict(list)
        self.seed_word = None
        self.memory_snapshots = []
        self.forgetting_rate = 0.1
        self.consolidation_strength = 0.3
        self.memory_importance = defaultdict(float)
        self.rehearsal_buffer = []
        self.learning_episodes = 0
        self.elastic_weights = defaultdict(float)
        self.synaptic_intelligence = defaultdict(float)
        self.progressive_networks = []

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

    def create_memory_snapshot(self):
        snapshot = {
            'model': deepcopy(self.model),
            'word_frequencies': dict(self.word_frequencies),
            'transition_matrix': {k: dict(v) for k, v in self.transition_matrix.items()},
            'vocabulary': set(self.vocabulary),
            'timestamp': datetime.now().isoformat(),
            'episode': self.learning_episodes
        }
        self.memory_snapshots.append(snapshot)
        print(f"Memory snapshot {len(self.memory_snapshots)} created at episode {self.learning_episodes}")

    def calculate_memory_importance(self):
        total_transitions = sum(sum(next_words.values()) for next_words in self.transition_matrix.values()) + 1e-9
        for word, next_words in self.transition_matrix.items():
            for next_word, count in next_words.items():
                frequency_importance = count / total_transitions
                rarity_bonus = 1.0 / (count + 1)
                cluster_bonus = self._get_cluster_importance(word, next_word)
                importance = frequency_importance * (1 + rarity_bonus * 0.5 + cluster_bonus * 0.3)
                self.memory_importance[(word, next_word)] = importance

    def _get_cluster_importance(self, word1, word2):
        for cluster in self.semantic_clusters.values():
            if word1 in cluster and word2 in cluster:
                return 1.0
            elif word1 in cluster or word2 in cluster:
                return 0.5
        return 0.0

    def apply_catastrophic_forgetting(self):
        if not self.memory_snapshots:
            return
        print(f"Applying catastrophic forgetting (rate: {self.forgetting_rate:.2f})...")
        forgetting_impacts = {}
        for word, next_words in self.transition_matrix.items():
            for next_word, count in next_words.items():
                transition_key = (word, next_word)
                base_forgetting = self.forgetting_rate * count
                importance = self.memory_importance.get(transition_key, 0.0)
                consolidation_protection = importance * self.consolidation_strength
                elastic_protection = self.elastic_weights.get(transition_key, 0.0) * 0.2
                actual_forgetting = max(0, base_forgetting - consolidation_protection - elastic_protection)
                forgetting_impacts[transition_key] = actual_forgetting
        transitions_forgotten = 0
        for (word, next_word), forgetting_amount in forgetting_impacts.items():
            if forgetting_amount > 0:
                original_count = self.transition_matrix[word][next_word]
                new_count = max(1, int(original_count - forgetting_amount))
                self.transition_matrix[word][next_word] = new_count
                self.word_frequencies[word] = max(1, self.word_frequencies[word] - int(forgetting_amount))
                if new_count < original_count:
                    transitions_forgotten += 1
        print(f"Catastrophic forgetting applied: {transitions_forgotten} transitions degraded")

    def rehearsal_learning(self, rehearsal_size=50):
        if len(self.rehearsal_buffer) < rehearsal_size:
            return
        print(f"Performing rehearsal learning with {rehearsal_size} samples...")
        rehearsal_samples = random.sample(self.rehearsal_buffer, min(rehearsal_size, len(self.rehearsal_buffer)))
        for word, next_word, original_strength in rehearsal_samples:
            if word in self.transition_matrix and next_word in self.transition_matrix[word]:
                boost_amount = int(original_strength * 0.1)
                self.transition_matrix[word][next_word] += boost_amount
                self.word_frequencies[word] += boost_amount

    def progressive_network_expansion(self):
        if len(self.progressive_networks) >= 5:
            return
        new_column = {
            'vocabulary': set(self.vocabulary),
            'transitions': deepcopy(self.transition_matrix),
            'creation_episode': self.learning_episodes,
            'specialization': self._calculate_specialization_score()
        }
        self.progressive_networks.append(new_column)
        print(f"Progressive network column {len(self.progressive_networks)} added")

    def _calculate_specialization_score(self):
        if not self.vocabulary:
            return 0.0
        vocab_size = len(self.vocabulary)
        total_transitions = sum(len(next_words) for next_words in self.transition_matrix.values())
        avg_transitions_per_word = total_transitions / (len(self.transition_matrix) + 1e-9)
        specialization = min(1.0, avg_transitions_per_word / 10.0)
        return specialization

    def build_enhanced_model(self, text, enable_forgetting=True):
        if self.model:
            self.create_memory_snapshot()
            if enable_forgetting:
                self.apply_catastrophic_forgetting()
                self.progressive_network_expansion()
        processed_text = text.lower()
        words = processed_text.split()
        if len(words) < 2:
            return
        print(f"Learning from {len(words)} words (Episode {self.learning_episodes + 1})...")
        for i in range(len(words) - 1):
            current, next_word = words[i], words[i + 1]
            original_count = self.transition_matrix[current][next_word]
            if original_count > 0:
                self.rehearsal_buffer.append((current, next_word, original_count))
        if len(self.rehearsal_buffer) > 1000:
            self.rehearsal_buffer = self.rehearsal_buffer[-1000:]
        new_vocabulary = set(words)
        self.vocabulary.update(new_vocabulary)
        for i in range(len(words) - 1):
            current, next_word = words[i], words[i + 1]
            self.transition_matrix[current][next_word] += 1
            self.word_frequencies[current] += 1
        self.calculate_memory_importance()
        self._update_elastic_weights()
        if enable_forgetting:
            self.rehearsal_learning()
        sample_sequence = words[:10]
        self.series_sum = 1.0
        self.model = self._build_weighted_model()
        self._calculate_performance_metrics()
        self.learning_episodes += 1
        print(f"Model updated. Episode {self.learning_episodes} complete.")

    def _update_elastic_weights(self):
        for word, next_words in self.transition_matrix.items():
            for next_word, count in next_words.items():
                transition_key = (word, next_word)
                importance = self.memory_importance.get(transition_key, 0.0)
                frequency_factor = count / (self.word_frequencies[word] + 1e-9)
                self.elastic_weights[transition_key] = importance * frequency_factor

    def _build_weighted_model(self):
        weighted_model = {}
        epsilon = 1e-9
        for word, next_words in self.transition_matrix.items():
            if not next_words:
                continue
            word_freq = max(self.word_frequencies[word], 1)
            transitions, weights = [], []
            for next_word, count in next_words.items():
                count = max(count, 1)
                base_weight = epsilon * max(self.series_sum, 1.0)
                freq_weight = (word_freq * count * count) / (word_freq + epsilon)
                rarity_penalty = 1.0 / (count + 1)
                transition_key = (word, next_word)
                consolidation_bonus = self.memory_importance.get(transition_key, 0.0) * 0.2
                weight = max(base_weight, freq_weight) * (1 - rarity_penalty * 0.8) + consolidation_bonus
                weight = max(weight, epsilon)
                transitions.append(next_word)
                weights.append(weight)
            if transitions and weights:
                weights = validate_probabilities(weights)
                weighted_model[word] = list(zip(transitions, weights))
        return weighted_model

    def _calculate_performance_metrics(self):
        forgetting_metrics = {
            'snapshots_count': len(self.memory_snapshots),
            'rehearsal_buffer_size': len(self.rehearsal_buffer),
            'progressive_columns': len(self.progressive_networks),
            'avg_memory_importance': np.mean(list(self.memory_importance.values())) if self.memory_importance else 0,
            'elastic_weights_count': len(self.elastic_weights),
            'learning_episodes': self.learning_episodes,
        }
        self.performance_metrics = {
            'vocabulary_size': len(self.vocabulary),
            'unique_transitions': sum(len(transitions) for transitions in self.model.values()),
            'avg_transitions_per_word': sum(len(transitions) for transitions in self.model.values()) / (len(self.model) + 1e-9) if self.model else 0,
            'series_convergence_sum': self.series_sum,
            'model_complexity': len(self.vocabulary) * self.series_sum,
            'entropy': self._calculate_entropy(),
            'build_timestamp': datetime.now().isoformat(),
            **forgetting_metrics
        }

    def _calculate_entropy(self):
        total_entropy = 0
        epsilon = 1e-9
        for word, transitions in self.model.items():
            word_entropy = 0
            for _, prob in transitions:
                if prob > 0:
                    word_entropy += -prob * math.log2(prob)
            total_entropy += word_entropy
        return total_entropy / (len(self.model) + epsilon) if self.model else 0

    def analyze_forgetting_impact(self):
        if len(self.memory_snapshots) < 2:
            return "Need at least 2 learning episodes to analyze forgetting."
        current_vocab = self.vocabulary
        current_transitions = set((w, nw) for w, next_words in self.transition_matrix.items() for nw in next_words.keys())
        first_snapshot = self.memory_snapshots[0]
        original_vocab = first_snapshot['vocabulary']
        original_transitions = set((w, nw) for w, next_words in first_snapshot['transition_matrix'].items() for nw in next_words.keys())
        vocab_retained = len(current_vocab & original_vocab) / (len(original_vocab) + 1e-9) * 100 if original_vocab else 0
        transitions_retained = len(current_transitions & original_transitions) / (len(original_transitions) + 1e-9) * 100 if original_transitions else 0
        vocab_new = len(current_vocab - original_vocab)
        transitions_new = len(current_transitions - original_transitions)
        analysis = f"""
Catastrophic Forgetting Analysis:
  Learning Episodes: {self.learning_episodes}
  Memory Snapshots: {len(self.memory_snapshots)}
Retention Rates:
  Vocabulary retained: {vocab_retained:.1f}% ({len(current_vocab & original_vocab)}/{len(original_vocab)})
  Transitions retained: {transitions_retained:.1f}% ({len(current_transitions & original_transitions)}/{len(original_transitions)})
New Knowledge:
  New vocabulary: {vocab_new} words
  New transitions: {transitions_new} patterns
Memory Mechanisms:
  Rehearsal buffer: {len(self.rehearsal_buffer)} samples
  Progressive networks: {len(self.progressive_networks)} columns
  Elastic weights: {len(self.elastic_weights)} constraints
  Avg memory importance: {np.mean(list(self.memory_importance.values())):.4f}
        """
        return analysis.strip()

    def generate_text(self, max_words=20, creativity_factor=0.3, start_word=None):
        if not self.model:
            return "No model available."
        if start_word is None and self.seed_word is not None:
            start_word = self.seed_word
        try:
            if isinstance(start_word, list):
                sentence = [start_word[0].capitalize()] + start_word[1:]
                current_word = start_word[-1].lower()
            else:
                if start_word and start_word.lower() in self.model:
                    current_word = start_word.lower()
                    sentence = [current_word.capitalize()]
                else:
                    current_word = random.choice(list(self.model.keys()))
                    sentence = [current_word.capitalize()]
        except (IndexError, ValueError):
            if self.model:
                current_word = random.choice(list(self.model.keys()))
                sentence = [current_word.capitalize()]
            else:
                return "No valid starting word available."
        generation_path = [current_word]
        for _ in range(max_words - len(sentence)):
            if current_word not in self.model or not self.model[current_word]:
                break
            try:
                words, weights = zip(*self.model[current_word])
                weights = validate_probabilities(weights, words)
                progressive_options = self._consult_progressive_networks(current_word)
                if progressive_options:
                    words, weights = self._blend_generation_options(words, weights, progressive_options)
                    weights = validate_probabilities(weights)
                if random.random() < creativity_factor:
                    inverse_weights = [(1.0 - w + 1e-9) * (w + 1e-9) for w in weights]
                    creativity_weights = validate_probabilities(inverse_weights)
                    next_word = np.random.choice(words, p=creativity_weights)
                else:
                    next_word = np.random.choice(words, p=weights)
                sentence.append(next_word)
                generation_path.append(next_word)
                current_word = next_word
            except (ValueError, IndexError) as e:
                print(f"Generation error at word '{current_word}': {e}")
                break
        generated_text = " ".join(sentence)
        if generated_text and generated_text[-1] not in ".?!":
            generated_text += "."
        self.generation_history.append({
            'text': generated_text, 'path': generation_path, 'creativity_factor': creativity_factor,
            'length': len(sentence), 'timestamp': datetime.now().isoformat(),
            'used_progressive_networks': len(self.progressive_networks) > 0
        })
        return generated_text

    def _consult_progressive_networks(self, current_word):
        progressive_options = []
        for network in self.progressive_networks:
            if current_word in network['transitions']:
                next_words = network['transitions'][current_word]
                next_total = sum(next_words.values())
                if next_total > 0:
                    for next_word, count in next_words.items():
                        weight = count / next_total
                        progressive_options.append((next_word, weight * network['specialization']))
        return progressive_options

    def _blend_generation_options(self, current_words, current_weights, progressive_options):
        if not progressive_options:
            return current_words, current_weights
        current_weights = validate_probabilities(current_weights)
        combined_options = {}
        for word, weight in zip(current_words, current_weights):
            if np.isfinite(weight) and weight > 0:
                combined_options[word] = weight * 0.8
        for word, weight in progressive_options:
            if np.isfinite(weight) and weight > 0:
                if word in combined_options:
                    combined_options[word] += weight * 0.2
                else:
                    combined_options[word] = weight * 0.3
        if not combined_options:
            return current_words, current_weights
        words = list(combined_options.keys())
        weights = [combined_options[w] for w in words]
        weights = validate_probabilities(weights)
        return words, weights

    def build_semantic_clusters(self):
        co_occurrence = defaultdict(lambda: defaultdict(float))
        for word, transitions in self.transition_matrix.items():
            for next_word, count in transitions.items():
                co_occurrence[word][next_word] += count
                co_occurrence[next_word][word] += count * 0.5
        self.semantic_clusters = {}
        processed_words = set()
        for word in self.vocabulary:
            if word in processed_words:
                continue
            cluster = [word]
            word_neighbors = set(co_occurrence[word].keys())
            for other_word in self.vocabulary:
                if other_word != word and other_word not in processed_words:
                    other_neighbors = set(co_occurrence[other_word].keys())
                    intersection = len(word_neighbors & other_neighbors)
                    union = len(word_neighbors | other_neighbors)
                    similarity = intersection / (union + 1e-9) if union > 0 else 0
                    if similarity > 0.8:
                        cluster.append(other_word)
                        processed_words.add(other_word)
            if len(cluster) > 1:
                cluster_name = f"cluster_{len(self.semantic_clusters)}"
                self.semantic_clusters[cluster_name] = cluster
            processed_words.add(word)

    def simulate_neural_spike_patterns(self):
        if not self.word_frequencies:
            return
        max_freq = max(self.word_frequencies.values())
        if max_freq == 0:
            return
        for word, frequency in self.word_frequencies.items():
            base_rate = min(frequency / max_freq, 1.0)
            spikes = [t for t in range(100) if random.random() < base_rate * 0.1]
            self.spike_patterns[word] = spikes

    def neural_weighted_selection(self, current_word, available_words, weights):
        weights = validate_probabilities(weights)
        if current_word not in self.spike_patterns or not self.spike_patterns[current_word]:
            return np.random.choice(available_words, p=weights)
        current_spikes = self.spike_patterns[current_word]
        neural_weights = []
        for word in available_words:
            if word in self.spike_patterns and self.spike_patterns[word]:
                target_spikes = self.spike_patterns[word]
                try:
                    synchrony = sum(1 for st in current_spikes for tt in target_spikes if abs(st-tt) <= 5)
                    max_sync = min(len(current_spikes), len(target_spikes))
                    sync_weight = synchrony / (max_sync + 1e-9) if max_sync > 0 else 0.1
                except:
                    sync_weight = 0.1
            else:
                sync_weight = 0.1
            neural_weights.append(max(sync_weight, 1e-9))
        try:
            neural_weights = validate_probabilities(neural_weights)
            combined_weights = [w * 0.7 + nw * 0.3 for w, nw in zip(weights, neural_weights)]
            final_weights = validate_probabilities(combined_weights)
        except:
            final_weights = weights
        return np.random.choice(available_words, p=final_weights)

    def generate_with_neural_enhancement(self, max_words=20, neural_strength=0.5):
        if not self.model:
            return "No model available."
        if not self.semantic_clusters:
            self.build_semantic_clusters()
        if not self.spike_patterns:
            self.simulate_neural_spike_patterns()
        if self.seed_word is not None:
            if isinstance(self.seed_word, list):
                sentence = [self.seed_word[0].capitalize()] + self.seed_word[1:]
                current_word = self.seed_word[-1].lower()
            else:
                current_word = self.seed_word.lower()
                sentence = [current_word.capitalize()]
        else:
            current_word = random.choice(list(self.model.keys()))
            sentence = [current_word.capitalize()]
        for _ in range(max_words - len(sentence)):
            if current_word in self.model and self.model[current_word]:
                words, weights = zip(*self.model[current_word])
                weights = validate_probabilities(weights)
                progressive_options = self._consult_progressive_networks(current_word)
                if progressive_options:
                    words, weights = self._blend_generation_options(words, weights, progressive_options)
                    weights = validate_probabilities(weights)
                if random.random() < neural_strength:
                    next_word = self.neural_weighted_selection(current_word, words, weights)
                else:
                    next_word = np.random.choice(words, p=weights)
                sentence.append(next_word)
                current_word = next_word
            else:
                break
        generated_text = " ".join(sentence)
        if generated_text and generated_text[-1] not in ".?!":
            generated_text += "."
        return generated_text

# EXAMPLE USAGE:
if __name__ == "__main__":
    generator = NeuralEnhancedMarkov()
    generator.forgetting_rate = 0.15
    generator.consolidation_strength = 0.4
    print("=== Neural-Enhanced Markov Text Generator ===")
    print("Demonstrating memory consolidation and catastrophic forgetting.")

    # First learning episode
    print("\n--- Episode 1: Learning from Primary Text ---")
    filename1 = input("Enter the path for the primary text file: ")
    try:
        with open(filename1, 'r', encoding='utf-8') as f:
            corpus1 = f.read()[:KB_LEN]
        generator.build_enhanced_model(corpus1, enable_forgetting=True)
        generator.build_semantic_clusters()
        generator.simulate_neural_spike_patterns()
        print("\nAfter first learning:")
        print(generator.analyze_forgetting_impact())
    except FileNotFoundError:
        print(f"Error: File not found at '{filename1}'. Skipping first learning episode.")

    # Optional second learning episode
    print("\n--- Episode 2: Learning from Secondary Text ---")
    filename2 = input("Enter the path for the secondary text file (or press Enter to skip): ")
    if filename2.strip():
        try:
            with open(filename2, 'r', encoding='utf-8') as f:
                corpus2 = f.read()[:KB_LEN]
            if generator.model:
                generator.build_enhanced_model(corpus2, enable_forgetting=True)
                generator.build_semantic_clusters()
                generator.simulate_neural_spike_patterns()
                print("\nAfter second learning:")
                print(generator.analyze_forgetting_impact())
            else:
                print("Cannot perform second learning episode as the first one failed.")
        except FileNotFoundError:
            print(f"Error: File not found at '{filename2}'. Skipping second learning episode.")
    else:
        print("Skipping second learning episode.")
    print("\n--- Text Generation ---")

    while True:
        if generator.model:
            seed_input = input("Enter a seed word or phrase for generation (or press Enter for random): ")
            generator.set_seed(seed_input)
            print("Neural-Enhanced Generation")
            text2 = generator.generate_with_neural_enhancement(max_words=550, neural_strength=0.7)
            print(f"Generated Text:\n{text2}\n")
        else:
            print("Cannot generate text. No model was built.")

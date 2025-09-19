import random
import math
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

KB_LEN = 99999

# -------------------------
# Temporal-Spatial Swatch Module
# -------------------------
class TemporalSpatialSwatchModule:
    """
    Implements temporal-spatial swatches for context-aware text generation.
    Swatches capture patterns across time and positional contexts.
    """
    def __init__(self, temporal_window=100, spatial_radius=50, swatch_decay=0.1):
        self.temporal_window = temporal_window  # How far back to look in time (in hours)
        self.spatial_radius = spatial_radius    # Context window around current position
        self.swatch_decay = swatch_decay        # How quickly temporal influence decays
        self.temporal_swatches = defaultdict(lambda: defaultdict(list))
        self.spatial_swatches = defaultdict(lambda: defaultdict(dict))
        self.composite_swatches = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.swatch_history = []

    def create_temporal_swatch(self, word_sequence, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        time_segment = self._get_time_segment(timestamp)
        for i in range(len(word_sequence) - 1):
            current_word = word_sequence[i]
            next_word = word_sequence[i + 1]
            self.temporal_swatches[time_segment][current_word].append({
                'next_word': next_word,
                'timestamp': timestamp,
                'position': i
            })

    def create_spatial_swatch(self, word_sequence, focus_position):
        start_pos = max(0, focus_position - self.spatial_radius)
        end_pos = min(len(word_sequence), focus_position + self.spatial_radius + 1)
        context_window = word_sequence[start_pos:end_pos]
        focus_word = word_sequence[focus_position] if focus_position < len(word_sequence) else None
        if focus_word:
            for i, context_word in enumerate(context_window):
                relative_position = i - (focus_position - start_pos)
                if relative_position != 0:
                    self.spatial_swatches[focus_word][context_word][relative_position] = \
                        self.spatial_swatches[focus_word][context_word].get(relative_position, 0) + 1

    def create_composite_swatch(self, word_sequence, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        time_segment = self._get_time_segment(timestamp)
        for pos in range(len(word_sequence)):
            word = word_sequence[pos]
            spatial_context = self._get_spatial_context(word_sequence, pos)
            for context_word, distance in spatial_context.items():
                weight = self._calculate_composite_weight(distance, timestamp)
                self.composite_swatches[time_segment][word][context_word] += weight

    def _get_time_segment(self, timestamp):
        return f"hour_{timestamp.hour}"

    def _get_spatial_context(self, word_sequence, position):
        context = {}
        start_pos = max(0, position - self.spatial_radius)
        end_pos = min(len(word_sequence), position + self.spatial_radius + 1)
        for i in range(start_pos, end_pos):
            if i != position:
                distance = abs(i - position)
                context[word_sequence[i]] = distance
        return context

    def _calculate_composite_weight(self, spatial_distance, timestamp):
        spatial_weight = 1.0 / (1.0 + spatial_distance)
        temporal_weight = self.swatch_decay ** (datetime.now() - timestamp).seconds
        return spatial_weight * temporal_weight

    def get_temporal_prediction(self, current_word, current_time=None):
        if current_time is None:
            current_time = datetime.now()
        time_segment = self._get_time_segment(current_time)
        if time_segment in self.temporal_swatches and current_word in self.temporal_swatches[time_segment]:
            recent_transitions = []
            for transition in self.temporal_swatches[time_segment][current_word]:
                age = (current_time - transition['timestamp']).total_seconds()
                if age <= self.temporal_window * 3600:
                    weight = self.swatch_decay ** (age / 3600)
                    recent_transitions.append((transition['next_word'], weight))
            if recent_transitions:
                total_weight = sum(weight for _, weight in recent_transitions)
                return [(word, weight / total_weight) for word, weight in recent_transitions]
        return []

    def get_spatial_prediction(self, current_word, spatial_context):
        if current_word not in self.spatial_swatches:
            return []
        predictions = defaultdict(float)
        for context_word in spatial_context:
            if context_word in self.spatial_swatches[current_word]:
                for position, count in self.spatial_swatches[context_word][current_word].items():
                    weight = count / (1.0 + abs(position))
                    predictions[context_word] += weight
        total_weight = sum(predictions.values())
        if total_weight > 0:
            return [(word, weight / total_weight) for word, weight in predictions.items()]
        return []

    def generate_swatch_summary(self):
        summary = {
            'temporal_swatches': len(self.temporal_swatches),
            'spatial_swatches': len(self.spatial_swatches),
            'composite_swatches': len(self.composite_swatches),
            'total_patterns': sum(len(v) for v in self.temporal_swatches.values()) +
                              sum(len(v) for v in self.spatial_swatches.values()) +
                              sum(len(v) for v in self.composite_swatches.values()),
            'swatch_diversity': len(set().union(*[v.keys() for v in self.temporal_swatches.values()])),
            'generation_timestamp': datetime.now().isoformat()
        }
        return summary

# -------------------------
# Cybernetic Feedback Module
# -------------------------
class CyberneticFeedbackModule:
    """Enhanced with swatch-aware feedback."""
    def __init__(self, feedback_gain=0.3, stability_margin=0.1, adaptation_rate=0.05):
        self.feedback_gain = feedback_gain
        self.stability_margin = stability_margin
        self.adaptation_rate = adaptation_rate
        self.error_history = []
        self.swatch_corrections = defaultdict(float)

    def compute_feedback(self, predicted_prob, actual_prob, swatch_context=None):
        error = actual_prob - predicted_prob
        self.error_history.append(error)
        correction = self.feedback_gain * error
        if swatch_context:
            swatch_adjustment = self.swatch_corrections.get(swatch_context, 0.0)
            correction += swatch_adjustment * 0.1
        correction = max(min(correction, 1 - self.stability_margin), -1 + self.stability_margin)
        return correction

    def update_swatch_corrections(self, swatch_context, error):
        self.swatch_corrections[swatch_context] += error * self.adaptation_rate

    def adapt_gain(self):
        if len(self.error_history) < 15:
            return
        variance = np.var(self.error_history[-15:])
        if variance > 0.05:
            self.feedback_gain *= (1 - self.adaptation_rate)
        else:
            self.feedback_gain *= (1 + self.adaptation_rate)

# -------------------------
# Neural Enhanced Markov
# -------------------------
class NeuralEnhancedMarkov:
    """
    A Markov text generator with an enhanced "neural" architecture.
    Uses temporal-spatial swatches to inform predictions and feedback.
    """
    def __init__(self, convergence_threshold=1e-6):
        self.convergence_threshold = convergence_threshold
        self.synapses = defaultdict(lambda: defaultdict(int))
        self.model = {}
        self.series_sum = 0
        self.word_frequencies = defaultdict(int)
        self.vocabulary = set()
        self.generation_history = []
        self.performance_metrics = {}
        self.seed_word = None
        self.spike_patterns = defaultdict(list)
        self.cybernetics = CyberneticFeedbackModule()
        self.swatch_module = TemporalSpatialSwatchModule()

        # For cybernetic monitor / anomaly loop
        self.event_log = []  # persistent log for anomaly events
        # prototypes (random init; replace with learned prototypes as needed)
        self.P1 = np.random.randn(16)
        self.P2 = np.random.randn(16)
        # controller placeholder
        self.controller = SimpleController()
    def extract_text_features(self, text_segment):
        """Extract features from generated text for cybernetic analysis"""
        words = text_segment.split()
        
        # Calculate basic text metrics
        features = {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_words': len(set(words)),
            'repetition_score': 1 - (len(set(words)) / len(words)) if words else 0
        }
        
        # Use existing swatch analysis
        if hasattr(self, 'temporal_spatial_swatch'):
            swatch_scores = self.temporal_spatial_swatch.get_composite_swatch_weights(words[-1] if words else "")
            features['swatch_confidence'] = max(swatch_scores.values()) if swatch_scores else 0
        
        return features

    def add_swatches_from_corpus(self, corpus):
        """Build swatches for every sentence in corpus."""
        for sentence in corpus.split('.'):
            words = sentence.strip().lower().split()
            if words:
                self.swatch_module.create_temporal_swatch(words)
                for i in range(len(words)):
                    self.swatch_module.create_spatial_swatch(words, i)
                self.swatch_module.create_composite_swatch(words)

    def simulate_neural_spike_patterns(self):
        for word, frequency in self.word_frequencies.items():
            base_rate = min(frequency / max(self.word_frequencies.values()), 1.0)
            spikes = [t for t in range(100) if random.random() < base_rate * 0.1]
            self.spike_patterns[word] = spikes

    def build_semantic_clusters(self):
        co_occurrence = defaultdict(lambda: defaultdict(float))
        for word, transitions in self.synapses.items():
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
            print(f"Warning: The seed phrase '{seed_input.strip()}' contains words not found in vocabulary. Using random start.")

    def learn(self, text, epochs=10, learning_rate=0.1):
        """Training loop with swatch-enhanced feedback."""
        processed_text = text.lower()
        words = processed_text.split()
        if len(words) < 2:
            return
        print(f"Starting training for {epochs} epochs...")

        # Swatch building
        self.add_swatches_from_corpus(text)

        # Initialize synapses with raw counts
        for i in range(len(words) - 1):
            current, nxt = words[i], words[i + 1]
            self.synapses[current][nxt] += 1
            self.word_frequencies[current] += 1
            self.vocabulary.add(current)
            self.vocabulary.add(nxt)

        self.series_sum = self._calculate_infinite_series()

        # Training loop with swatch context
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i+1]
                neuron_output = self.synapses.get(current_word, {})
                if not neuron_output:
                    continue
                raw_outputs = np.array(list(neuron_output.values()), dtype=float)
                activated_outputs = 1 / (1 + np.exp(-raw_outputs))
                total_activated = np.sum(activated_outputs)
                predicted_probs = activated_outputs / total_activated if total_activated > 0 else np.zeros_like(activated_outputs)
                actual_prob = 1.0
                idx = list(neuron_output.keys()).index(next_word)
                predicted_prob = predicted_probs[idx]
                error = actual_prob - predicted_prob
                total_loss += abs(error)
                # Swatch context for feedback
                time_segment = self.swatch_module._get_time_segment(datetime.now())
                swatch_context = f"{current_word}_{time_segment}"
                correction = self.cybernetics.compute_feedback(predicted_prob, actual_prob, swatch_context=swatch_context)
                self.cybernetics.update_swatch_corrections(swatch_context, error)
                for j, (key, value) in enumerate(neuron_output.items()):
                    adj = learning_rate * error * predicted_probs[j]
                    self.synapses[current_word][key] += adj
            avg_loss = total_loss / (len(words) - 1)
            print(f"Epoch {epoch + 1}/{epochs}: Avg Loss = {avg_loss:.4f}")
        self._finalize_model()
        self._calculate_performance_metrics()

    def _calculate_infinite_series(self):
        total_sum = 0
        for depth in range(1, 2000):
            level_sum = 0
            for word, next_words in self.synapses.items():
                word_freq = self.word_frequencies[word]
                for nxt, count in next_words.items():
                    if word_freq == 0:
                        continue
                    contrib = (word_freq * count) / (depth + 1)
                    level_sum += contrib
            total_sum += level_sum * (depth + 1) ** -1.2
        return max(total_sum, 1e-6)

    def _finalize_model(self):
        self.model = {}
        for word, next_words in self.synapses.items():
            transitions, weights = [], []
            for nxt, count in next_words.items():
                activated_weight = 1 / (1 + math.exp(-count))
                transitions.append(nxt)
                weights.append(activated_weight)
            total = sum(weights)
            normalized = [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)
            corrected_weights = []
            time_segment = self.swatch_module._get_time_segment(datetime.now())
            for prob in normalized:
                swatch_context = f"{word}_{time_segment}"
                correction = self.cybernetics.compute_feedback(prob, 1.0 / len(normalized), swatch_context=swatch_context)
                corrected = prob + correction
                corrected_weights.append(max(corrected, 0.0))
            total_corrected = sum(corrected_weights)
            final_weights = [w / total_corrected for w in corrected_weights] if total_corrected > 0 else normalized
            self.model[word] = list(zip(transitions, final_weights))

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

    def generate_with_neural_enhancement(self, max_words=500, neural_strength=0.3, start_word=None):
        if not self.model:
            return "No model available. Please run the learn() method first."
        if start_word is None and self.seed_word is not None:
            start_word = self.seed_word
        if isinstance(start_word, list):
            sentence = [word for word in start_word]
            current_word = start_word[-1]
        else:
            current_word = start_word.lower() if start_word and start_word.lower() in self.model else random.choice(list(self.model.keys()))
            sentence = [current_word]
        generation_path = [current_word]
        for i in range(max_words - 1):
            # Predict using swatches first
            swatch_preds = self.swatch_module.get_temporal_prediction(current_word)
            if swatch_preds:
                words, weights = zip(*swatch_preds)
                next_word = np.random.choice(words, p=weights)
            else:
                try:
                    words, weights = zip(*self.model[current_word])
                    if random.random() < neural_strength:
                        inverse = [(1.0 - w) * w for w in weights]
                        total_inv = sum(inverse)
                        creativity_weights = [w / total_inv for w in inverse] if total_inv > 0 else weights
                        next_word = np.random.choice(words, p=creativity_weights)
                    else:
                        next_word = np.random.choice(words, p=weights)
                except:
                    False
            if next_word not in sentence[-1]:
                sentence.append(next_word)
            generation_path.append(next_word)
            current_word = next_word
        generated_text = " ".join(sentence)
        if generated_text[-1] not in ".?!":
            generated_text += "."
        self.cybernetics.adapt_gain()
        self.generation_history.append({
            'text': generated_text,
            'path': generation_path,
            'neural_strength': neural_strength,
            'length': len(sentence),
            'timestamp': datetime.now().isoformat()
        })
        return generated_text

    # -------------------------
    # --- Cybernetic pseudocode integrated as runnable loop
    # -------------------------
    def read_window_features(self):
        """Placeholder: return a fixed-dim feature vector (replace with real sensor fusion)."""
        return np.random.randn(16)

    def autoencoder_recon_error(self, r):
        """Placeholder: small positive scalar representing reconstruction error."""
        return float(np.abs(np.random.randn()) * 0.5 + 0.1)

    def dist_to_prototypes(self, r):
        d1 = np.linalg.norm(r - self.P1)
        d2 = np.linalg.norm(r - self.P2)
        return d1, d2

    def physio_arousal(self):
        """Placeholder: return a z-scored arousal metric (can be heart-rate variability, GSR, etc.)."""
        return float((np.random.randn() * 0.5))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def log_append(self, r, s, meta=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'features': r.tolist() if isinstance(r, np.ndarray) else r,
            'score': float(s),
            'meta': meta or {}
        }
        self.event_log.append(entry)

    def perform_action(self, action):
        """Placeholder action performer: replace with real intervention e.g., play sound, push notification."""
        print(f"[{datetime.now().isoformat()}] Performing action: {action}")

    def user_confirms_event(self):
        """Ask user for confirmation that the event matched a prototype. (blocking UX; swap to GUI in practice)"""
        try:
            resp = input("Confirm event matched a prototype? (y/n) [enter to skip]: ").strip().lower()
            return resp == 'y'
        except KeyboardInterrupt:
            return False

    def assign_prototype_and_update(self, prototype_index, r, eta=0.05):
        """Update prototype Pi using small learning rate after confirmation."""
        if prototype_index == 1:
            self.P1 = (1 - eta) * self.P1 + eta * r
            print("Updated P1 prototype.")
        elif prototype_index == 2:
            self.P2 = (1 - eta) * self.P2 + eta * r
            print("Updated P2 prototype.")
        else:
            print("Unknown prototype index; no update performed.")

    def cybernetic_event_loop(self,
                             alpha1=1.0, alpha2=1.0, alpha3=0.5,
                             T1=0.6,
                             step_delay=0.5,
                             max_iterations=None):
        """
        Implements:
        while True:
            r = read_window_features()
            ae = autoencoder_recon_error(r)
            d1, d2 = dist(r, P1), dist(r, P2)
            a = physio_arousal()
            s = sigmoid(alpha1*norm(d_min)+alpha2*norm(ae)+alpha3*a)
            log_append(r, s, meta)
            if s > T1:
                action = controller.choose_action(context=r)
                if action:
                    perform(action)
            if user_confirms_event():
                assign_prototype_and_update(Pi, r)
        """
        import time
        iterations = 0
        print("Starting cybernetic event loop (ctrl-c to stop)...")
        try:
            while True:
                # stop if reached max iterations (useful for testing)
                if max_iterations is not None and iterations >= max_iterations:
                    print("Max iterations reached; exiting loop.")
                    break

                r = self.read_window_features()
                ae = self.autoencoder_recon_error(r)
                d1, d2 = self.dist_to_prototypes(r)
                d_min = min(d1, d2)
                a = self.physio_arousal()

                # normalize components with simple running-scale (here we just scale by constants/placeholders)
                norm_d = d_min / (1.0 + d_min)
                norm_ae = ae / (1.0 + ae)
                # combine
                s = self.sigmoid(alpha1 * norm_d + alpha2 * norm_ae + alpha3 * a)

                # log it
                meta = {'d1': d1, 'd2': d2, 'ae': ae, 'arousal': a}
                self.log_append(r, s, meta=meta)

                # threshold action
                if s > T1:
                    action = self.controller.choose_action(context=r, score=s)
                    if action:
                        self.perform_action(action)

                # user confirmation -> prototype update (blocking)
                if s > T1 * 1.2:  # slightly stricter to avoid too many prompts
                    confirmed = self.user_confirms_event()
                    if confirmed:
                        # decide which prototype to update
                        pi = 1 if d1 < d2 else 2
                        self.assign_prototype_and_update(pi, r)

                iterations += 1
                time.sleep(step_delay)
        except KeyboardInterrupt:
            print("Cybernetic event loop stopped by user.")
    def evaluate_cybernetic_score(self, features, alpha1=1.0, alpha2=1.0, alpha3=0.5):
        """
        Implements the cybernetic scoring from your pseudocode:
        s = sigmoid(alpha1*norm(d_min)+alpha2*norm(ae)+alpha3*a)
        """
        import numpy as np
        
        # Extract window features (r) from text features
        r = np.array([
            features.get('word_count', 0),
            features.get('avg_word_length', 0), 
            features.get('unique_words', 0),
            features.get('repetition_score', 0)
        ])
        
        # Autoencoder reconstruction error (ae)
        # Simulate autoencoder error based on text coherence
        expected_patterns = np.array([10, 5, 8, 0.2])  # Expected good text patterns
        ae = np.linalg.norm(r - expected_patterns) / np.linalg.norm(expected_patterns)
        
        # Distance to prototypes P1 and P2 (d1, d2)
        # P1 = "good text" prototype, P2 = "bad text" prototype  
        P1 = np.array([15, 6, 12, 0.1])  # Good text features
        P2 = np.array([3, 3, 2, 0.8])    # Bad text features
        
        d1 = np.linalg.norm(r - P1)
        d2 = np.linalg.norm(r - P2)
        d_min = min(d1, d2)
        
        # Physiological arousal (a) - simulate based on text variety
        a = features.get('swatch_confidence', 0.5)
        
        # Normalize values
        d_min_norm = d_min / (np.linalg.norm(P1) + np.linalg.norm(P2))
        ae_norm = min(ae, 1.0)  # Cap at 1.0
        a_norm = min(a, 1.0)    # Cap at 1.0
        
        # Calculate cybernetic score using sigmoid
        raw_score = alpha1 * d_min_norm + alpha2 * ae_norm + alpha3 * a_norm
        s = 1 / (1 + np.exp(-raw_score))  # Sigmoid function
        
        # Store metadata for logging
        self.last_cybernetic_meta = {
            'features': r,
            'autoencoder_error': ae,
            'distance_to_P1': d1,
            'distance_to_P2': d2,
            'arousal': a,
            'raw_score': raw_score,
            'sigmoid_score': s
        }
        
        return s
    def adjust_generation_parameters(self, cybernetic_score):
        """
        Adjust generation parameters based on cybernetic feedback score.
        Implements the action controller from your pseudocode.
        """
        # Store current parameters if not already stored
        if not hasattr(self, 'base_neural_strength'):
            self.base_neural_strength = 0.8
            self.base_swatch_weight = 1.0
            self.base_learning_rate = 0.4
        
        # Adjust parameters based on cybernetic score
        if cybernetic_score > 0.8:
            # High score - text quality is good, maintain current settings
            self.current_neural_strength = self.base_neural_strength
            self.current_swatch_weight = self.base_swatch_weight
            #print("High quality detected - maintaining parameters")
            
        elif cybernetic_score > 0.6:
            # Medium score - slight enhancement needed
            self.current_neural_strength = min(self.base_neural_strength * 1.2, 1.0)
            self.current_swatch_weight = min(self.base_swatch_weight * 1.1, 2.0)
            #print("Medium quality - enhancing neural strength")
            
        elif cybernetic_score > 0.4:
            # Low score - significant adjustment needed
            self.current_neural_strength = min(self.base_neural_strength * 1.5, 1.0)
            self.current_swatch_weight = min(self.base_swatch_weight * 1.3, 2.0)
            #print("Low quality - boosting generation parameters")
            
        else:
            # Very low score - reset and try different approach
            self.current_neural_strength = 0.9
            self.current_swatch_weight = 1.5
            # Could also trigger re-learning or different seed selection
            print("Very low quality - resetting parameters")
        
        # Apply temporal-spatial swatch adjustments if available
        if hasattr(self, 'temporal_spatial_swatch'):
            self.temporal_spatial_swatch.swatch_decay *= (1.1 if cybernetic_score < 0.5 else 0.95)
        
        return {
            'neural_strength': self.current_neural_strength,
            'swatch_weight': self.current_swatch_weight,
            'action_taken': f"Adjusted for score {cybernetic_score:.3f}"
        }

# -------------------------
# Simple controller used by the cybernetic loop
# -------------------------
class SimpleController:
    """
    A trivial controller to choose actions based on context. Replace with a contextual bandit or policy.
    """
    def __init__(self):
        self.actions = ['grounding_cue', 'play_metronome', 'breathing_exercise', None]

    def choose_action(self, context=None, score=0.0):
        # higher score -> choose more intrusive/helpful action
        if score < 0.7:
            return None
        # probability-weighted pick biased by score
        weights = [max(0.0, score - 0.6) * (i+1) for i in range(len(self.actions))]
        total = sum(weights)
        if total <= 0:
            return None
        probs = [w/total for w in weights]
        return np.random.choice(self.actions, p=probs)

# -------------------------
# Usage Example
# -------------------------
try:
    filename = input("Filename: ")
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = f.read()[:KB_LEN]
        generator = NeuralEnhancedMarkov()
        generator.learn(corpus, epochs=15, learning_rate=0.4) # Train the model over 15 epochs
        generator.build_semantic_clusters()
        generator.simulate_neural_spike_patterns()

        while True:
            # Generate text segment
            text_segment = generator.generate_with_neural_enhancement(max_words=500, start_word=input("USER: ").split())
            
            # Extract features from generated text for cybernetic analysis
            features = generator.extract_text_features(text_segment)
            
            # Apply cybernetic evaluation
            score = generator.evaluate_cybernetic_score(features)
            generator.adjust_generation_parameters(cybernetic_score=score)
            print(text_segment)
       
except FileNotFoundError:
    print("Error: The specified file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

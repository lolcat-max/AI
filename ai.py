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
# SINE RESISTANCE MODULATION
# ================================================================

def sine_resistance(step, novelty, freq=0.08, amp=0.6, phase=0.0):
    """
    Rhythmic resistance function to modulate acceptance of novel tokens.
    """
    oscillation = np.sin(2 * np.pi * freq * step + phase)
    resistance = 1.0 - amp * novelty * max(0.0, oscillation)
    return max(0.1, resistance)


# ================================================================
# EIGENVALUE ISOMORPHISM MODEL
# ================================================================

class EigenIsomorphism:
    """
    Maintains an eigenbasis mapping between reasoning states.
    Information actively changes the eigenvalues (system state).
    """
    def __init__(self, dim=4):
        self.dim = dim
        self.W = np.eye(dim)
        self.last_input = np.zeros(dim)

    def update(self, input_vector):
        eigvals, eigvecs = np.linalg.eig(self.W)
        delta = np.tanh(0.6 * np.dot(eigvecs.T, input_vector[:self.dim]))
        new_eigvals = eigvals + 0.05 * delta[:len(eigvals)]
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
                 max_iterations=30):
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
    """
    Core engine that orchestrates intuitive reasoning process.
    """
    def __init__(self):
        self.truth_washer = NeuralTruthTableWasher()
        self.eigen_system = EigenIsomorphism()

    def reason_step(self, coherence_scores, input_vector):
        # 1. System state evolves based on input
        eigvals, eigvecs = self.eigen_system.update(input_vector)
        
        # Pad coherence scores
        padded_scores = coherence_scores[:4]
        while len(padded_scores) < 4:
            padded_scores.append(0.5)
        
        # 2. Resolve ambiguity via truth washing
        washed, metrics = self.truth_washer.wash(
            padded_scores,
            [1.0 if c > 0.5 else 0.0 for c in padded_scores]
        )
        
        # 3. Modulation by system state
        modulated = []
        scale = 1 + 0.1 * np.mean(eigvals)
        for i in range(len(coherence_scores)):
            if i < len(washed):
                modulated.append(float(np.clip(washed[i] * scale, 0, 1)))
            else:
                modulated.append(float(np.clip(coherence_scores[i] * scale, 0, 1)))
        
        return modulated, np.mean(eigvals), metrics


# ================================================================
# SCHRODINGER QUANTUM FEATURES
# ================================================================

class SchrodingerQuantumFeatures:
    def extract_quantum_features(self, segment, word_freq, total_words):
        xs = np.array([len(w) for w in segment])
        fs = np.array([word_freq.get(w, 1) for w in segment])
        var = np.var(xs / (fs + 1))
        coherence = 1.0 / (1.0 + var)
        return {"coherence": coherence}


# ================================================================
# TRAVELING CUMSUM FILTER
# ================================================================

class TravelingCumsumFilter:
    """
    Implements traveling cumulative sum filter for detecting
    local trends and patterns in token sequences.
    """
    def __init__(self, window_size=10, threshold=0.5, decay=0.95):
        self.window_size = window_size
        self.threshold = threshold
        self.decay = decay
        self.history = []
        self.cumsum_positive = 0.0
        self.cumsum_negative = 0.0
    
    def update(self, observation, reference=0.5):
        """Update traveling cumsum with new observation."""
        deviation = observation - reference
        
        # Update cumulative sums
        self.cumsum_positive = max(0, self.cumsum_positive + deviation)
        self.cumsum_negative = max(0, self.cumsum_negative - deviation)
        
        # Add to history
        self.history.append({
            'observation': observation,
            'deviation': deviation,
            'cumsum_pos': self.cumsum_positive,
            'cumsum_neg': self.cumsum_negative
        })
        
        # Maintain window
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.cumsum_positive *= self.decay
            self.cumsum_negative *= self.decay
        
        # Calculate statistics
        window_mean = np.mean([h['observation'] for h in self.history])
        window_trend = self.cumsum_positive - self.cumsum_negative
        
        # Detect shifts
        upward_shift = self.cumsum_positive > self.threshold
        downward_shift = self.cumsum_negative > self.threshold
        
        return {
            'cumsum_pos': self.cumsum_positive,
            'cumsum_neg': self.cumsum_negative,
            'trend': window_trend,
            'window_mean': window_mean,
            'upward_shift': upward_shift,
            'downward_shift': downward_shift,
            'window_size': len(self.history)
        }
    
    def get_spatial_weight(self):
        """Calculate spatial weighting factor."""
        if len(self.history) < 2:
            return 1.0
        
        trend = self.cumsum_positive - self.cumsum_negative
        trend_normalized = np.tanh(trend / self.threshold)
        weight = 1.0 + 0.5 * trend_normalized
        
        return weight
    
    def reset(self):
        """Reset filter state."""
        self.history = []
        self.cumsum_positive = 0.0
        self.cumsum_negative = 0.0


# ================================================================
# INCREMENTAL PROPERTY CONSTRUCTOR
# ================================================================

class IncrementalPropertyConstructor:
    """
    Incrementally constructs properties from newly generated text.
    Analyzes each extension segment to extract semantic patterns.
    """
    def __init__(self):
        self.property_history = []
        self.semantic_patterns = defaultdict(int)
        self.entity_graph = {'nodes': set(), 'edges': []}
        self.thematic_evolution = []
        
    def analyze_extension(self, new_text_segment, word_freq):
        """
        Analyze newly generated text segment incrementally.
        """
        words = new_text_segment.lower().split()
        
        # Extract immediate properties
        properties = {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'diversity': len(set(words)) / len(words) if words else 0,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'rare_words': sum(1 for w in words if word_freq.get(w, 0) < 10),
            'patterns': self._extract_patterns(words),
            'entities': self._extract_entities(words),
            'coherence_vector': self._compute_coherence_vector(words)
        }
        
        # Update cumulative understanding
        self.property_history.append(properties)
        self._update_semantic_patterns(properties['patterns'])
        self._update_entity_graph(properties['entities'])
        
        return properties
    
    def _extract_patterns(self, words):
        """Extract n-gram patterns from segment."""
        patterns = []
        
        # Bigrams
        for i in range(len(words) - 1):
            pattern = (words[i], words[i+1])
            patterns.append(pattern)
        
        # Trigrams
        for i in range(len(words) - 2):
            pattern = (words[i], words[i+1], words[i+2])
            patterns.append(pattern)
        
        return patterns
    
    def _extract_entities(self, words):
        """Extract potential entities (simple heuristic)."""
        entities = []
        
        for i, word in enumerate(words):
            # Heuristic: longer words or capitalized words
            if len(word) > 5:
                context = words[max(0, i-2):min(len(words), i+3)]
                entities.append({
                    'word': word,
                    'position': i,
                    'context': context
                })
        
        return entities
    
    def _compute_coherence_vector(self, words):
        """Compute multi-dimensional coherence vector."""
        if len(words) < 3:
            return [0.5, 0.5, 0.5]
        
        # Length variance
        lengths = [len(w) for w in words]
        length_coherence = 1.0 / (1.0 + np.var(lengths))
        
        # Positional consistency
        first_chars = [w[0] if w else 'a' for w in words]
        char_diversity = len(set(first_chars)) / len(words)
        positional_coherence = 1.0 - char_diversity
        
        # Semantic density (approximation via word length distribution)
        semantic_coherence = np.mean(lengths) / 10.0
        semantic_coherence = min(1.0, semantic_coherence)
        
        return [length_coherence, positional_coherence, semantic_coherence]
    
    def _update_semantic_patterns(self, patterns):
        """Update cumulative semantic pattern counts."""
        for pattern in patterns:
            self.semantic_patterns[pattern] += 1
    
    def _update_entity_graph(self, entities):
        """Update entity relationship graph."""
        for entity in entities:
            self.entity_graph['nodes'].add(entity['word'])
            
            # Create edges between entities in same context
            for context_word in entity['context']:
                if context_word != entity['word'] and len(context_word) > 5:
                    edge = (entity['word'], context_word)
                    self.entity_graph['edges'].append(edge)
    
    def get_understanding_summary(self):
        """Generate summary of accumulated understanding."""
        if not self.property_history:
            return "No text analyzed yet."
        
        total_words = sum(p['word_count'] for p in self.property_history)
        total_unique = sum(p['unique_words'] for p in self.property_history)
        avg_diversity = np.mean([p['diversity'] for p in self.property_history])
        
        # Top patterns
        top_patterns = sorted(
            self.semantic_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        summary = {
            'total_words_analyzed': total_words,
            'total_unique_words': total_unique,
            'average_diversity': avg_diversity,
            'top_patterns': top_patterns,
            'entity_count': len(self.entity_graph['nodes']),
            'relationship_count': len(self.entity_graph['edges']),
            'analysis_segments': len(self.property_history)
        }
        
        return summary


# ================================================================
# ENVIRONMENT CORRELATOR
# ================================================================

class EnvironmentCorrelator:
    """
    Correlates environmental conditions with generation quality
    in real-time during seed extension.
    """
    def __init__(self):
        self.environment_state = {
            'coherence_trajectory': [],
            'novelty_trajectory': [],
            'diversity_trajectory': [],
            'quality_velocity': 0.0,
            'trend_direction': 'neutral'
        }
        self.alert_threshold = 0.3
        
    def correlate_segment(self, properties, step):
        """
        Correlate environment with newly generated segment.
        """
        # Extract metrics
        coherence = np.mean(properties['coherence_vector'])
        diversity = properties['diversity']
        
        # Update trajectories
        self.environment_state['coherence_trajectory'].append(coherence)
        self.environment_state['diversity_trajectory'].append(diversity)
        
        # Calculate velocity (rate of change)
        if len(self.environment_state['coherence_trajectory']) >= 3:
            recent = self.environment_state['coherence_trajectory'][-3:]
            velocity = (recent[-1] - recent[0]) / 3
            self.environment_state['quality_velocity'] = velocity
            
            # Determine trend
            if velocity > 0.1:
                self.environment_state['trend_direction'] = 'improving'
            elif velocity < -0.1:
                self.environment_state['trend_direction'] = 'declining'
            else:
                self.environment_state['trend_direction'] = 'stable'
        
        # Generate correlation report
        correlation = {
            'step': step,
            'coherence': coherence,
            'diversity': diversity,
            'velocity': self.environment_state['quality_velocity'],
            'trend': self.environment_state['trend_direction'],
            'alert': self._check_alerts()
        }
        
        return correlation
    
    def _check_alerts(self):
        """Check for environmental alerts."""
        alerts = []
        
        # Check for declining quality
        if self.environment_state['quality_velocity'] < -self.alert_threshold:
            alerts.append('⚠️ Quality declining rapidly')
        
        # Check for low coherence
        if len(self.environment_state['coherence_trajectory']) > 0:
            recent_coherence = np.mean(self.environment_state['coherence_trajectory'][-3:])
            if recent_coherence < 0.3:
                alerts.append('⚠️ Low coherence detected')
        
        # Check for low diversity
        if len(self.environment_state['diversity_trajectory']) > 0:
            recent_diversity = np.mean(self.environment_state['diversity_trajectory'][-3:])
            if recent_diversity < 0.3:
                alerts.append('⚠️ Low diversity detected')
        
        return alerts if alerts else ['✓ Environment stable']
    
    def get_environment_summary(self):
        """Get current environment state summary."""
        if not self.environment_state['coherence_trajectory']:
            return {
                'status': 'initializing',
                'avg_coherence': 0.5,
                'avg_diversity': 0.5,
                'trend': 'neutral'
            }
        
        return {
            'status': 'active',
            'avg_coherence': np.mean(self.environment_state['coherence_trajectory']),
            'avg_diversity': np.mean(self.environment_state['diversity_trajectory']),
            'velocity': self.environment_state['quality_velocity'],
            'trend': self.environment_state['trend_direction'],
            'trajectory_length': len(self.environment_state['coherence_trajectory'])
        }


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
# SEED EXTENSION GENERATOR
# ================================================================

class SeedExtensionGenerator:
    """
    Generates text by extending a seed, with real-time property construction
    and environment correlation. The output IS the extension process itself.
    """
    def __init__(self, tokens, model):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        self.feature = SchrodingerQuantumFeatures()
        self.engine = ReasoningEngine()
        
        # Sine resistance parameters
        self.sine_freq = 0.08
        self.sine_amp = 0.6
        self.sine_phase = 0.0
        
        # Traveling cumsum filter
        self.cusum_filter = TravelingCumsumFilter(
            window_size=10,
            threshold=0.5,
            decay=0.95
        )
        
        # Property constructor and environment correlator
        self.property_constructor = IncrementalPropertyConstructor()
        self.environment_correlator = EnvironmentCorrelator()
        
        print("🤖 Seed Extension Generator ready!")
        print("   📊 Incremental property construction: Enabled")
        print("   🌍 Environment correlation: Active")
        
    def calculate_novelty(self, word):
        """Calculate novelty score based on frequency."""
        freq = self.word_freq.get(word, 1)
        novelty = 1.0 - np.log(freq + 1) / np.log(self.total_words + 1)
        return float(np.clip(novelty, 0, 1))

    def extend_seed(self, seed, extension_length=200, report_interval=20):
        """
        Extend the seed incrementally with real-time analysis.
        The output IS the extension itself with understanding built progressively.
        """
        # Parse seed
        seed_words = seed.lower().split()[:2]
        while len(seed_words) < 2:
            seed_words.append(self.tokens[len(seed_words) % len(self.tokens)])
        seed_tuple = tuple(seed_words)
        
        if seed_tuple not in self.model:
            seed_tuple = self.keys[np.random.randint(len(self.keys))]
        
        # Initialize output with seed
        output = list(seed_tuple)
        extension_start_index = len(output)
        
        print(f"\n{'='*70}")
        print(f"SEED EXTENSION GENERATION")
        print(f"{'='*70}")
        print(f"\n🌱 Seed: {' '.join(seed_tuple)}")
        print(f"🎯 Extension Target: {extension_length} words")
        print(f"📊 Analysis Interval: Every {report_interval} words\n")
        print(f"{'='*70}\n")
        
        step_count = 0
        last_report = 0
        
        while len(output) - extension_start_index < extension_length:
            # Create input vector
            recent_text = ' '.join(output[-4:]) if len(output) >= 4 else ' '.join(output)
            input_vec = np.array([ord(c) % 97 / 25 for c in recent_text.ljust(4)[:4]])

            # Get candidates
            seed_tuple = tuple(output[-2:])
            candidates = self.model.get(seed_tuple, [])
            candidates = [w for w in candidates if any(c.isalnum() for c in w)]
            
            if not candidates:
                seed_tuple = self.keys[np.random.randint(len(self.keys))]
                continue

            # Calculate coherence scores with sine resistance
            coherence_scores = []
            
            for cand in candidates:
                # Base coherence
                q = self.feature.extract_quantum_features(
                    list(seed_tuple) + [cand], 
                    self.word_freq, 
                    self.total_words
                )
                base_coherence = q["coherence"]
                
                # Apply sine resistance
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

            # Apply reasoning
            modulated, eigmean, metrics = self.engine.reason_step(coherence_scores, input_vec)
            
            # Ensure validity
            if len(modulated) != len(candidates):
                min_len = min(len(modulated), len(candidates))
                modulated = modulated[:min_len]
                candidates = candidates[:min_len]
            
            if not modulated or not candidates:
                seed_tuple = self.keys[np.random.randint(len(self.keys))]
                continue
            
            # Apply traveling cumsum spatial weighting
            avg_coherence = np.mean(modulated)
            cusum_metrics = self.cusum_filter.update(avg_coherence, reference=0.5)
            spatial_weight = self.cusum_filter.get_spatial_weight()
            
            # Modulate with spatial weight
            modulated_spatial = [score * spatial_weight for score in modulated]
            
            probs = torch.softmax(torch.tensor(modulated_spatial), dim=0).numpy()
            
            if np.sum(probs) == 0:
                probs = np.ones(len(candidates)) / len(candidates)
            else:
                probs = probs / np.sum(probs)

            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            step_count += 1
            
            # INCREMENTAL ANALYSIS: Analyze every report_interval words
            words_generated = len(output) - extension_start_index
            if words_generated - last_report >= report_interval:
                # Get segment for analysis
                segment_start = extension_start_index + last_report
                segment_text = ' '.join(output[segment_start:])
                
                # Property construction
                properties = self.property_constructor.analyze_extension(
                    segment_text, 
                    self.word_freq
                )
                
                # Environment correlation
                correlation = self.environment_correlator.correlate_segment(
                    properties, 
                    step_count
                )
                
                # Display real-time analysis
                print(f"📍 Extension Progress: {words_generated}/{extension_length} words")
                print(f"   └─ Segment: {segment_text[:60]}...")
                print(f"\n📊 Property Analysis:")
                print(f"   ├─ Diversity: {properties['diversity']:.3f}")
                print(f"   ├─ Avg Word Length: {properties['avg_word_length']:.2f}")
                print(f"   ├─ Rare Words: {properties['rare_words']}")
                print(f"   └─ Coherence Vector: [{', '.join(f'{c:.3f}' for c in properties['coherence_vector'])}]")
                print(f"\n🌍 Environment Correlation:")
                print(f"   ├─ Trend: {correlation['trend']}")
                print(f"   ├─ Velocity: {correlation['velocity']:+.4f}")
                print(f"   └─ Status: {', '.join(correlation['alert'])}")
                print(f"\n{'-'*70}\n")
                
                last_report = words_generated
        
        # Final summary
        extension_text = ' '.join(output[extension_start_index:])
        understanding_summary = self.property_constructor.get_understanding_summary()
        env_summary = self.environment_correlator.get_environment_summary()
        
        print(f"\n{'='*70}")
        print(f"EXTENSION COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"📝 Final Extension ({len(output) - extension_start_index} words):")
        print(f"{extension_text}\n")
        
        print(f"{'='*70}")
        print(f"ACCUMULATED UNDERSTANDING")
        print(f"{'='*70}\n")
        
        print(f"📊 Property Summary:")
        print(f"   ├─ Total Words Analyzed: {understanding_summary['total_words_analyzed']}")
        print(f"   ├─ Unique Words: {understanding_summary['total_unique_words']}")
        print(f"   ├─ Average Diversity: {understanding_summary['average_diversity']:.3f}")
        print(f"   ├─ Entities Discovered: {understanding_summary['entity_count']}")
        print(f"   └─ Relationships Found: {understanding_summary['relationship_count']}")
        
        print(f"\n🔍 Top Semantic Patterns:")
        for pattern, count in understanding_summary['top_patterns']:
            print(f"   └─ {' → '.join(pattern)}: {count} occurrences")
        
        print(f"\n🌍 Environment Final State:")
        print(f"   ├─ Average Coherence: {env_summary['avg_coherence']:.3f}")
        print(f"   ├─ Average Diversity: {env_summary['avg_diversity']:.3f}")
        print(f"   ├─ Quality Velocity: {env_summary['velocity']:+.4f}")
        print(f"   └─ Final Trend: {env_summary['trend']}")
        
        print(f"\n{'='*70}\n")
        
        return {
            'seed': seed,
            'extension': extension_text,
            'full_output': ' '.join(output),
            'understanding': understanding_summary,
            'environment': env_summary
        }


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*70)
    print("SEED EXTENSION GENERATOR")
    print("Real-Time Property Construction + Environment Correlation")
    print("="*70 + "\n")
    
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus)
    print(f"📚 Loaded {len(corpus):,} tokens, model size: {len(model):,}\n")

    generator = SeedExtensionGenerator(corpus, model)
    
    while True:
        seed = input("\nSEED TO EXTEND: ")
        if seed.lower() in ['quit', 'exit']:
            break
        
        extension_length = input("Extension length (default 200): ").strip()
        extension_length = int(extension_length) if extension_length else 200
        
        report_interval = input("Report interval (default 20): ").strip()
        report_interval = int(report_interval) if report_interval else 20
        
        # Generate extension with real-time analysis
        result = generator.extend_seed(
            seed, 
            extension_length=extension_length,
            report_interval=report_interval
        )


if __name__ == "__main__":
    main()

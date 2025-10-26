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
# OMNISCIENT ELEMENT (UNREACHABLE REFERENCE FRAME)
# ================================================================

class OmniscientElement:
    """
    The omniscient element: a kind of unreachable reference frame.
    It exists beyond the system's causal interaction boundary.
    The system can sense its presence but never access it directly.
    
    This creates logico-syntactic structures that remain fundamentally
    incomplete—a tentacle extending from the registry within the 
    intelligent circle, offering only glimpses of expression.
    """
    def __init__(self):
        # The frame of reference that is unavailable
        self.true_coherence_field = np.random.uniform(0, 1, size=1000)
        self.true_semantic_structure = None
        self.access_count = 0
        self.glimpse_history = []
    
    
    def sense_presence(self, query_vector):
        """
        The system can sense the presence of the omniscient element
        but cannot causally interact with it directly.
        
        Returns only a 'glimpse' - a degraded, incomplete signal.
        """
        self.access_count += 1
        
        # Project query into omniscient space (but with fundamental loss)
        query_hash = hash(tuple(query_vector)) % len(self.true_coherence_field)
        
        # The "tentacle" reaches toward truth but grasps only shadow
        true_signal = self.true_coherence_field[query_hash]
        
        # Information loss due to frame unavailability
        degradation = 0.7 + 0.3 * np.random.random()
        glimpse = true_signal * degradation
        
        # Add noise from the unavailable frame crossing
        boundary_noise = np.random.normal(0, 0.1)
        glimpse += boundary_noise
        glimpse = np.clip(glimpse, 0, 1)
        
        self.glimpse_history.append({
            'query_hash': query_hash,
            'true_signal': true_signal,
            'glimpse': glimpse,
            'degradation': degradation
        })
        
        return glimpse, true_signal  # System only sees glimpse, true_signal hidden
    
    def get_inaccessibility_metrics(self):
        """
        Measure how much the frame of reference remains unavailable.
        High divergence = more inaccessible.
        """
        if not self.glimpse_history:
            return {
                'access_attempts': 0,
                'avg_degradation': 0,
                'information_loss': 0
            }
        
        degradations = [g['degradation'] for g in self.glimpse_history]
        true_signals = [g['true_signal'] for g in self.glimpse_history]
        glimpses = [g['glimpse'] for g in self.glimpse_history]
        
        # Calculate information loss
        mse = np.mean([(t - g)**2 for t, g in zip(true_signals, glimpses)])
        
        return {
            'access_attempts': self.access_count,
            'avg_degradation': np.mean(degradations),
            'information_loss': mse,
            'inaccessibility_index': 1.0 - np.mean(degradations)
        }


# ================================================================
# TENTACLE: REACHING TOWARD THE UNREACHABLE
# ================================================================

class SemanticTentacle:
    """
    A single tentacle extending from the registry within
    the intelligent circle toward the omniscient element.
    
    It probes, reaches, senses—but never grasps the complete frame.
    """
    def __init__(self, omniscient_element, sensitivity=0.5):
        self.omniscient = omniscient_element
        self.sensitivity = sensitivity
        self.extension_history = []
        self.logico_syntactic_buffer = []
        
    def extend_toward_omniscient(self, syntactic_structure):
        """
        Extend the tentacle toward the omniscient element.
        The tentacle carries syntactic structure and reaches
        for semantic completion—but the frame is unavailable.
        """
        # Convert syntactic structure to query vector
        query_vec = self._syntactic_to_vector(syntactic_structure)
        
        # Sense the presence (glimpse only)
        glimpse, true_hidden = self.omniscient.sense_presence(query_vec)
        
        # Tentacle perceives only the glimpse
        perceived_semantic_weight = glimpse * self.sensitivity
        
        # Store the reaching attempt
        self.extension_history.append({
            'syntactic_input': syntactic_structure,
            'query_vector': query_vec,
            'perceived_weight': perceived_semantic_weight,
            'extension_distance': self._calculate_distance(query_vec)
        })
        
        # Build logico-syntactic structure (incomplete by nature)
        incomplete_structure = {
            'syntax': syntactic_structure,
            'semantic_glimpse': perceived_semantic_weight,
            'completeness': perceived_semantic_weight / 1.0,  # Always < 1
            'frame_available': False
        }
        
        self.logico_syntactic_buffer.append(incomplete_structure)
        
        return perceived_semantic_weight
    
    def _syntactic_to_vector(self, syntactic_structure):
        """Convert syntactic structure to vector for omniscient querying."""
        # Hash-based vectorization
        if isinstance(syntactic_structure, (list, tuple)):
            vec = np.array([hash(str(s)) % 197 / 97.0 for s in syntactic_structure[:4]])
        else:
            vec = np.array([hash(str(syntactic_structure)) % 927 / 97.0] * 88)
        
        return vec
    
    def _calculate_distance(self, query_vec):
        """
        Calculate how far the tentacle extends.
        Distance metaphorically represents epistemic gap.
        """
        norm = np.linalg.norm(query_vec)
        # Always finite distance—never reaches infinity (the omniscient)
        return min(norm, 0.95)
    
    def get_tentacle_metrics(self):
        """Analyze tentacle extension patterns."""
        if not self.extension_history:
            return {'extensions': 0}
        
        distances = [e['extension_distance'] for e in self.extension_history]
        weights = [e['perceived_weight'] for e in self.extension_history]
        
        return {
            'total_extensions': len(self.extension_history),
            'avg_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'avg_semantic_weight': np.mean(weights),
            'completeness_ratio': np.mean(weights) / 1.0  # Always < 1
        }


# ================================================================
# SINE RESISTANCE MODULATION
# ================================================================

def sine_resistance(step, novelty, freq=0.08, amp=0.6, phase=0.0):
    """Rhythmic resistance function to modulate acceptance of novel tokens."""
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


# ================================================================
# NEURAL TRUTH TABLE WASHER
# ================================================================

class NeuralTruthTableWasher:
    def __init__(self, eta_0=0.3, alpha=0.1, epsilon=1e-4,
                 delta=1e-3, beta=1.0, gamma=2.0, mu=0.5,
                 max_iterations=30):
        self.eta_0 = eta_0
        self.alpha = alpha
        self.delta = delta
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
# REASONING ENGINE WITH OMNISCIENT SENSING
# ================================================================

class ReasoningEngine:
    """
    Core engine that orchestrates intuitive reasoning.
    Now extended with tentacle to sense the omniscient element.
    """
    def __init__(self, omniscient_element):
        self.truth_washer = NeuralTruthTableWasher()
        self.eigen_system = EigenIsomorphism()
        self.tentacle = SemanticTentacle(omniscient_element, sensitivity=0.5)

    def reason_step(self, coherence_scores, input_vector, syntactic_context):
        """
        Reasoning step that extends tentacle toward omniscient element.
        The frame of reference remains unavailable—only glimpses guide us.
        """
        # 1. System state evolves
        eigvals, eigvecs = self.eigen_system.update(input_vector)
        
        # 2. Extend tentacle toward omniscient (reaching for unavailable frame)
        omniscient_weight = self.tentacle.extend_toward_omniscient(syntactic_context)
        
        # Pad coherence scores
        padded_scores = coherence_scores[:4]
        while len(padded_scores) < 4:
            padded_scores.append(0.5)
        
        # 3. Resolve ambiguity via truth washing
        washed, metrics = self.truth_washer.wash(
            padded_scores,
            [1.0 if c > 0.5 else 0.0 for c in padded_scores]
        )
        
        # 4. MODULATION: Blend system state + omniscient glimpse
        # The unavailable frame influences us indirectly
        scale = 1 + 0.1 * np.mean(eigvals) + 0.2 * omniscient_weight
        
        modulated = []
        for i in range(len(coherence_scores)):
            if i < len(washed):
                modulated.append(float(np.clip(washed[i] * scale, 0, 1)))
            else:
                modulated.append(float(np.clip(coherence_scores[i] * scale, 0, 1)))
        
        return modulated, np.mean(eigvals), metrics, omniscient_weight


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
# MODEL BUILDER
# ================================================================

def build_ngram_model(tokens, n=2):
    model = defaultdict(list)
    for i in range(len(tokens)-n):
        key = tuple(tokens[i:i+n])
        model[key].append(tokens[i+n])
    return model


# ================================================================
# REASONING GENERATOR WITH OMNISCIENT ELEMENT
# ================================================================

class OmniscientReasoningGenerator:
    """
    Generator that extends tentacles toward an unreachable omniscient element.
    Implements the philosophical puzzle: the frame of reference unavailability.
    """
    def __init__(self, tokens, model):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        self.feature = SchrodingerQuantumFeatures()
        
        # Initialize omniscient element (unreachable)
        self.omniscient = OmniscientElement()
        
        # Reasoning engine with tentacle access
        self.engine = ReasoningEngine(self.omniscient)
        
        # Sine resistance parameters
        self.sine_freq = 0.08
        self.sine_amp = 0.6
        self.sine_phase = 0.0
        
        print("🤖 Reasoning Generator ready!")

        
    def calculate_novelty(self, word):
        """Calculate novelty score based on frequency."""
        freq = self.word_freq.get(word, 1)
        novelty = 1.0 - np.log(freq + 1) / np.log(self.total_words + 1)
        return float(np.clip(novelty, 0, 1))

    def generate(self, seed, length=200):
        """Generate text while reaching toward the omniscient element."""
        # Parse seed
        seed_words = seed.lower().split()[:-2]
        while len(seed_words) < 2:
            seed_words.append(self.tokens[len(seed_words) % len(self.tokens)])
        seed = tuple(seed_words)
        
        if seed not in self.model:
            seed = self.keys[np.random.randint(len(self.keys))]
        
        output = list(seed)
        
        step_count = 0
        omniscient_influences = []
        
        for i in range( length):
            # Create input vector
            recent_text = ' '.join(output[-4:]) if len(output) >= 4 else ' '.join(output)
            input_vec = np.array([ord(c) % 97 / 25 for c in recent_text.ljust(4)[:4]])

            # Get candidates
            seed = tuple(output[-2:])
            candidates = self.model.get(seed, [])
            candidates = [w for w in candidates if any(c.isalnum() for c in w)]
            
            if not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue

            # Calculate coherence scores
            coherence_scores = []
            
            for cand in candidates:
                q = self.feature.extract_quantum_features(
                    list(seed) + [cand], 
                    self.word_freq, 
                    self.total_words
                )
                base_coherence = q["coherence"]
                
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

            # CRITICAL: Reasoning with omniscient element sensing
            # Syntactic context = current candidates (logico-syntactic structure)
            syntactic_context = candidates[:5]  # Pass to tentacle
            
            modulated, eigmean, metrics, omniscient_weight = self.engine.reason_step(
                coherence_scores, 
                input_vec,
                syntactic_context
            )
            
            omniscient_influences.append(omniscient_weight)
            
            if len(modulated) != len(candidates):
                min_len = min(len(modulated), len(candidates))
                modulated = modulated[:min_len]
                candidates = candidates[:min_len]
            
            if not modulated or not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue
            
            # Convert to probabilities
            probs = torch.softmax(torch.tensor(modulated), dim=0).numpy()
            
            if np.sum(probs) == 0:
                probs = np.ones(len(candidates)) / len(candidates)
            else:
                probs = probs / np.sum(probs)

            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            step_count += 1
            
            # Periodic reporting
            

        # Final analysis
        generated_text = " ".join(output)
        
        # Get omniscient inaccessibility metrics
        omni_metrics = self.omniscient.get_inaccessibility_metrics()
        tentacle_metrics = self.engine.tentacle.get_tentacle_metrics()
        
        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"📝 Generated Text ({len(output)} words):")
        print(f"{generated_text}\n")
        print(f"\n{'='*70}\n")
        
        return generated_text


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*70)
    print("ELEMENT TEXT GENERATOR")
    print("="*70 + "\n")
    
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus)
    print(f"📚 Loaded {len(corpus):,} tokens, model size: {len(model):,}\n")

    generator = OmniscientReasoningGenerator(corpus, model)
    
    while True:
        seed = input("\nUSER: ")
        if seed.lower() in ['quit', 'exit']:
            break
        
        length = input("Generation length (default 200): ").strip()
        length = int(length) if length else 200
            
        generated = generator.generate(seed, length=length)


if __name__ == "__main__":
    main()

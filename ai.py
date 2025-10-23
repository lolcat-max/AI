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
    
    Args:
        step: Current generation step
        novelty: [0,1] scale where 0 = frequent word, 1 = unseen/rare word
        freq: Oscillation frequency
        amp: Amplitude of resistance effect
        phase: Phase offset
    
    Returns:
        Scaling multiplier to reduce coherence for high-novelty tokens
    """
    oscillation = np.sin(2 * np.pi * freq * step + phase)
    # Resistance increases with novelty and inhibits during positive oscillation peaks
    resistance = 1.0 - amp * novelty * max(0.0, oscillation)
    return max(0.1, resistance)  # Keep minimum at 0.1 to avoid complete suppression


# ================================================================
# EIGENVALUE ISOMORPHISM MODEL
# ================================================================

class EigenIsomorphism:
    """
    Maintains an eigenbasis mapping between reasoning states.
    Each input perturbs eigenvalues but preserves spectral structure.
    """
    def __init__(self, dim=4):
        self.dim = dim
        self.W = np.eye(dim)
        self.last_input = np.zeros(dim)
        print("⚛️ Eigenvalue Isomorphism Engine initialized")

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
    def __init__(self):
        self.truth_washer = NeuralTruthTableWasher()
        self.eigen_system = EigenIsomorphism()
        print("🧠 Reasoning Engine initialized with sine resistance.")

    def reason_step(self, coherence_scores, input_vector):
        eigvals, eigvecs = self.eigen_system.update(input_vector)
        
        # Pad coherence scores if less than 4
        padded_scores = coherence_scores[:4]
        while len(padded_scores) < 4:
            padded_scores.append(0.5)
        
        washed, metrics = self.truth_washer.wash(
            padded_scores,
            [1.0 if c > 0.5 else 0.0 for c in padded_scores]
        )
        
        # apply eigenvalue modulation to washed coherence
        modulated = []
        scale = 1 + 0.1 * np.mean(eigvals)
        for i in range(len(coherence_scores)):
            if i < len(washed):
                modulated.append(float(np.clip(washed[i] * scale, 0, 1)))
            else:
                modulated.append(float(np.clip(coherence_scores[i] * scale, 0, 1)))
        
        return modulated, np.mean(eigvals), metrics


# ================================================================
# SCHRODINGER QUANTUM FEATURES (simplified)
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
# REASONING GENERATOR WITH SINE RESISTANCE
# ================================================================

class ReasoningGenerator:
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
        
        print("🤖 Generator ready with reactive eigenvalue logic + sine resistance")
        print(f"   🌊 Sine resistance: freq={self.sine_freq}, amp={self.sine_amp}")

    def calculate_novelty(self, word):
        """
        Calculate novelty score for a word based on its frequency.
        Returns value in [0, 1] where 1 = very rare/novel, 0 = very common
        """
        freq = self.word_freq.get(word, 1)
        # Normalize using logarithm to handle frequency distribution
        novelty = 1.0 - np.log(freq + 1) / np.log(self.total_words + 1)
        return float(np.clip(novelty, 0, 1))

    def generate(self, seed, length=50):
        # Parse seed into tuple
        seed_words = seed.lower().split()[:2]
        while len(seed_words) < 2:
            seed_words.append(self.tokens[len(seed_words) % len(self.tokens)])
        seed = tuple(seed_words)
        
        if seed not in self.model:
            seed = self.keys[np.random.randint(len(self.keys))]
        
        output = list(seed)
        
        print(f"\n🌀 Generating {length} words with sine resistance modulation...")
        print(f"   Seed: {' '.join(seed)}\n")
        
        step_count = 0
        
        while len(output) < length:
            # Convert recent output to input vector for eigenvalue modulation
            recent_text = ' '.join(output[-4:]) if len(output) >= 4 else ' '.join(output)
            input_vec = np.array([ord(c) % 97 / 25 for c in recent_text.ljust(4)[:4]])

            # Get candidates and filter punctuation
            candidates = self.model.get(seed, [])
            candidates = [w for w in candidates if any(c.isalnum() for c in w)]
            
            if not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue

            # Calculate coherence scores with sine resistance
            coherence_scores = []
            novelty_scores = []
            resistance_factors = []
            
            for cand in candidates:
                # Base coherence from quantum features
                q = self.feature.extract_quantum_features(
                    list(seed) + [cand], 
                    self.word_freq, 
                    self.total_words
                )
                base_coherence = q["coherence"]
                
                # Calculate novelty and apply sine resistance
                novelty = self.calculate_novelty(cand)
                resistance_factor = sine_resistance(
                    step_count, 
                    novelty, 
                    freq=self.sine_freq, 
                    amp=self.sine_amp, 
                    phase=self.sine_phase
                )
                
                # Apply resistance to coherence
                adjusted_coherence = base_coherence * resistance_factor
                
                coherence_scores.append(adjusted_coherence)
                novelty_scores.append(novelty)
                resistance_factors.append(resistance_factor)

            # Apply reasoning and eigenvalue modulation
            modulated, eigmean, metrics = self.engine.reason_step(coherence_scores, input_vec)
            
            # Ensure we have valid probabilities
            if len(modulated) != len(candidates):
                min_len = min(len(modulated), len(candidates))
                modulated = modulated[:min_len]
                candidates = candidates[:min_len]
                novelty_scores = novelty_scores[:min_len]
                resistance_factors = resistance_factors[:min_len]
            
            if not modulated or not candidates:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue
            
            probs = torch.softmax(torch.tensor(modulated), dim=0).numpy()
            
            # Normalize probabilities
            if np.sum(probs) == 0:
                probs = np.ones(len(candidates)) / len(candidates)
            else:
                probs = probs / np.sum(probs)

            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            selected_idx = candidates.index(next_word)
            
            output.append(next_word)
            seed = tuple(output[-2:])
            step_count += 1

            # Display generation info every 10 steps
            if step_count % 10 == 0:
                sine_phase_deg = (2 * np.pi * self.sine_freq * step_count) % (2 * np.pi)
                sine_phase_deg = np.degrees(sine_phase_deg)
                novelty = novelty_scores[selected_idx]
                resistance = resistance_factors[selected_idx]
                #print(f"[{len(output)}/{length}] λ̄={eigmean:.3f}, err={metrics['final_error']:.5f}, nov={novelty:.2f}, res={resistance:.2f}, φ={sine_phase_deg:.0f}°")
                #print(f"   Last 10: {' '.join(output[-10:])}")

        return " ".join(output)


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n=== Eigenvalue-Isomorphic Neural Reasoner with Sine Resistance ===")
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus)
    print(f"Loaded {len(corpus):,} tokens, model size: {len(model):,}")

    generator = ReasoningGenerator(corpus, model)
    
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

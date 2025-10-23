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
        print("🧠 Reasoning Engine initialized.")

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
# REASONING GENERATOR
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
        print("🤖 Generator ready with reactive eigenvalue logic")

    def generate(self, seed, length=50):
        if seed not in self.model:
            seed = self.keys[np.random.randint(len(self.keys))]
        output = list(seed)
        
        print("\n🌀 Generation Mode:")
        print("   • Type text to influence eigenvalues")
        
        step_count = 0
        
        while len(output) < length:
            # Handle user input
            user = ' '.join(output[:-2])
            input_vec = np.zeros(4)

            # Get candidates and filter punctuation
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
                coherence_scores.append(q["coherence"])

            # Apply reasoning and eigenvalue modulation
            modulated, eigmean, metrics = self.engine.reason_step(coherence_scores, input_vec)
            
            # Ensure we have valid probabilities
            if len(modulated) != len(candidates):
                min_len = min(len(modulated), len(candidates))
                modulated = modulated[:min_len]
                candidates = candidates[:min_len]
            
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
            output.append(next_word)
            seed = tuple(output[-2:])
            step_count += 1
        return " ".join(output)


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n=== Eigenvalue-Isomorphic Neural Reasoner ===")
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus)
    print(f"Loaded {len(corpus):,} tokens, model size: {len(model):,}")

    generator = ReasoningGenerator(corpus, model)
    seed = input("USER: ")
    generated = generator.generate(seed, length=500)
    print("\n=== Final Output ===\n")
    print(generated)
    print(f"\nTotal words: {len(generated.split())}")


if __name__ == "__main__":
    main()

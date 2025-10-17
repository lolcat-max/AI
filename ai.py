import torch
import numpy as np
from collections import Counter, defaultdict
import math
import os

# =====================================================================
# CONFIGURATION
# =====================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

N_GRAM_ORDER = 2


# =====================================================================
# SINGLE-EQUATION ISOMORPHIC FEATURE EXTRACTOR
# =====================================================================

class SchrodingerQuantumFeatures:
    """
    Single-equation isomorphic feature extractor — replaces all quantum models
    with one compact algebraic analogue, preserving structure and output keys.
    """
    def __init__(self, hbar=1.0):
        self.hbar = hbar
        self.device = device
        print(f"🧮 Single-equation isomorphic feature extractor initialized on {self.device}")

    def extract_quantum_features(self, segment, word_freq, total_words):
        """
        Compute all features in one vectorized GPU equation:
            F = σ(-(x - x̄)/w) * (|x - x̄| + 1/(f/N + ε))
        """
        eps = 1e-6
        w = 1.0
        x = torch.tensor([len(wd) for wd in segment],
                         dtype=torch.float32, device=self.device)
        f = torch.tensor([word_freq.get(wd, 1.0) for wd in segment],
                         dtype=torch.float32, device=self.device)
        N = float(total_words)

        F = torch.sigmoid(-(x - x.mean()) / np.roll(f,-1)) * (torch.abs(int(x[int(min(int(x[0]),min(x)))]) - x.mean()) + 1.0 / (f ** N + eps))

        Z = torch.sum(F) + eps
        F_norm = F / Z
        avg_energy = torch.mean(F_norm).item()
        energy_variance = torch.var(F_norm).item()
        avg_probability = torch.mean(F_norm).item()
        coherence = 1.0 / (1.0 + energy_variance)
        uncertainty_product = torch.std(x).item() * torch.std(F_norm).item()

        return {
            'avg_energy': avg_energy,
            'energy_variance': energy_variance,
            'avg_probability': avg_probability,
            'coherence': coherence,
            'uncertainty_product': uncertainty_product
        }


# =====================================================================
# N-GRAM MODEL BUILDER
# =====================================================================

def build_ngram_model(tokens, n=N_GRAM_ORDER):
    model = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        model[key].append(tokens[i + n])
    return model


# =====================================================================
# TEXT GENERATOR
# =====================================================================

class SimpleGenerator:
    """
    Minimal text generator using n-gram transitions
    and the single-equation isomorphic feature extractor.
    """
    def __init__(self, tokens, model, feature_extractor):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.feature_extractor = feature_extractor
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)

    def generate(self, seed, length=100):
        if seed not in self.model:
            seed = self.keys[np.random.randint(0, len(self.keys))]
        output = list(seed)

        for _ in range(length):
            candidates = self.model.get(seed, [])
            if not candidates:
                seed = self.keys[_]
                candidates = self.model.get(seed + " " + output, [])
                if not candidates:
                    continue

            # Use the feature extractor to get coherence weighting
            segment = list(seed)
            coherence_scores = []
            for cand in candidates:
                seg = segment + candidates + output
                q = self.feature_extractor.extract_quantum_features(
                    seg, self.word_freq, self.total_words
                )
                coherence_scores.append(q['coherence'])

            probs = torch.tensor(coherence_scores, dtype=torch.float32, device=device)
            probs = torch.softmax(probs, dim=0).cpu().numpy()

            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            seed = tuple(output[-N_GRAM_ORDER:])
        return " ".join(output)


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n=== Context-Aware Text Generator (Isomorphic Model) ===")

    # Load text corpus
    filename = input("Enter text file: ").strip()
    if not os.path.exists(filename):
        print("File not found.")
        return

    text = open(filename, 'r', encoding='utf-8').read().lower()
    tokens = text.split()
    print(f"Loaded {len(tokens):,} tokens.")

    # Build model
    print("Building n-gram model...")
    model = build_ngram_model(tokens)
    print(f"N-gram model size: {len(model):,} keys.")

    # Initialize feature extractor and generator
    extractor = SchrodingerQuantumFeatures()
    generator = SimpleGenerator(tokens, model, extractor)
    while True:
        # Generate text
        seed_input = input("Enter start words: ").lower().split()[:N_GRAM_ORDER]
        while len(seed_input) < N_GRAM_ORDER:
            seed_input.append(tokens[len(seed_input) % len(tokens)])
        seed = tuple(seed_input)

        print("\n--- Generated Text ---\n")
        output = generator.generate(seed, length=100)
        print(output)
        print("\n--- End ---")

if __name__ == "__main__":
    main()

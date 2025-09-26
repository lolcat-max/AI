
import hashlib
import numpy as np
import random
from collections import defaultdict

KB_LEN = -1

import hashlib
import numpy as np
import random
from collections import defaultdict, Counter

KB_LEN = -1

class SimpleHashWeightGenerator:
    """
    Text generator with SHA256-derived scores, interpolated with
    a corpus bigram model and an online self bigram model under a
    bigram constraint mask.
    """

    def __init__(self, alpha=0.75, beta=0.20, gamma=0.05, smoothing=1.0):
        # Hash weight components
        self.word_weights = {}
        self.vocabulary = set()
        self.word_transitions = defaultdict(list)

        # Corpus counts
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.total_unigrams = 0

        # Generated (self) counts
        self.gen_unigram_counts = Counter()
        self.gen_bigram_counts = defaultdict(Counter)

        # Mixture weights and smoothing
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        s = self.alpha + self.beta + self.gamma
        if s <= 0:
            raise ValueError("alpha+beta+gamma must be > 0")
        # normalize in case user passed arbitrary values
        self.alpha /= s; self.beta /= s; self.gamma /= s
        self.smoothing = float(smoothing)  # e.g., add-k smoothing

    def hash_to_weight(self, word):
        if word in self.word_weights:
            return self.word_weights[word]
        h = hashlib.sha256(word.encode('utf-8')).hexdigest()
        int_value = int(h[:16], 16)
        max_value = int('f' * 16, 16)
        weight = int_value / max_value
        self.word_weights[word] = weight
        return weight

    def build_vocabulary(self, text):
        words = text.lower().split()
        # Build counts
        for i, w in enumerate(words):
            self.unigram_counts[w] += 1
            self.vocabulary.add(w)
            if i < len(words) - 1:
                nxt = words[i + 1]
                self.bigram_counts[w][nxt] += 1
                self.word_transitions[w].append(nxt)
                self.vocabulary.add(nxt)
        self.total_unigrams = sum(self.unigram_counts.values())

        # Precompute hash weights for vocabulary (optional)
        for w in sorted(self.vocabulary):
            _ = self.hash_to_weight(w)
        return len(self.vocabulary)

    def get_corpus_bigram_prob(self, prev_w, w):
        # Add-k smoothing
        c_big = self.bigram_counts[prev_w][w]
        c_prev = self.unigram_counts[prev_w]
        V = max(1, len(self.vocabulary))
        return (c_big + self.smoothing) / (c_prev + self.smoothing * V)

    def get_generated_bigram_prob(self, prev_w, w):
        c_big = self.gen_bigram_counts[prev_w][w]
        c_prev = self.gen_unigram_counts[prev_w]
        V = max(1, len(self.vocabulary))
        return (c_big + self.smoothing) / (c_prev + self.smoothing * V)

    def get_candidates_for_word(self, current_word, use_transitions=True):
        if use_transitions and current_word in self.word_transitions and self.word_transitions[current_word]:
            # Constrained: only bigrams seen in corpus from current_word
            return sorted(set(self.word_transitions[current_word]))
        # Fallback: allow a small pool if no outgoing edges
        return sorted(list(self.vocabulary))[: min(32, len(self.vocabulary))]

    def compute_interpolated_probabilities(self, current_word, candidates):
        # Normalize hash scores over candidates to [0,1] distribution
        hash_scores = np.array([self.hash_to_weight(''.join(sorted(c))) for c in candidates], dtype=float)
        if hash_scores.sum() == 0.0:
            hash_probs = np.ones_like(hash_scores) / len(candidates)
        else:
            hash_probs = hash_scores / hash_scores.sum()

        # Corpus/self bigram probabilities
        corp_probs = np.array([self.get_corpus_bigram_prob(current_word, c) for c in candidates], dtype=float)
        corp_probs /= corp_probs.sum() if corp_probs.sum() > 0 else len(candidates)

        gen_probs = np.array([self.get_generated_bigram_prob(current_word, c) for c in candidates], dtype=float)
        gen_probs /= gen_probs.sum() if gen_probs.sum() > 0 else len(candidates)

        # Linear interpolation
        mix = self.alpha * corp_probs + self.beta * gen_probs + self.gamma * hash_probs
        # Renormalize to guard against numerical issues
        total = mix.sum()
        if total <= 0:
            return np.ones_like(mix) / len(candidates)
        return mix / total

    def update_generated_counts(self, prev_w, w):
        self.gen_unigram_counts[prev_w] += 1
        self.gen_unigram_counts[w] += 1
        self.gen_bigram_counts[prev_w][w] += 1

    def generate_text(self, start_word, max_words=15, use_transitions=True):
        if not self.vocabulary:
            return ""
        start = start_word.lower()
        if start not in self.vocabulary:
            start = random.choice(list(self.vocabulary))
        current = start
        generated = [current]

        for _ in range(max_words - 1):
            candidates = self.get_candidates_for_word(current, use_transitions=use_transitions)
            if not candidates:
                break
            probs = self.compute_interpolated_probabilities(current, candidates)
            next_word = np.random.choice(candidates, p=probs)
            self.update_generated_counts(current, next_word)
            generated.append(next_word)
            current = next_word

        return " ".join(generated)

    
    def generate_text(self, start_word, max_words=15, use_transitions=True):
        """Generate text using hex-to-float weights only."""
        if start_word.lower() not in self.vocabulary:
            start_word = random.choice(list(self.vocabulary))
            print("Word not found in vocabulary, using random word.")
        
        current_word = start_word.lower()
        generated_sequence = [current_word]
        
        print(f"\nGenerating text starting with: '{current_word}'")
        print(f"Starting word hex weight: {self.hash_to_weight(current_word):.6f}")
        print("\nGeneration steps:")
        
        for step in range(max_words - 1):
            # Get candidates
            candidates = self.get_candidates_for_word(current_word, use_transitions)
            
            if not candidates:
                print(f"Step {step + 1}: No candidates found, stopping")
                break
            
            # Compute probabilities based on hex weights
            probabilities = self.compute_interpolated_probabilities(current_word, candidates)
            
            # Select next word
            next_word = np.random.choice(candidates, p=probabilities)
            
            generated_sequence.append(next_word)
            current_word = next_word
        
        generated_text = " ".join(generated_sequence)
        return generated_text
    
    def show_weight_analysis(self, words=None):
        """Show hex-to-float weight analysis for words."""
        if words is None:
            words = list(self.vocabulary)[:10]
        
        print(f"\nHex-to-Float Weight Analysis:")
        print("=" * 60)
        
        # Sort words by weight for analysis
        word_weight_pairs = [(word, self.hash_to_weight(word)) for word in words]
        word_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for word, weight in word_weight_pairs:
            hash_hex = hashlib.sha256(word.encode()).hexdigest()
            hex_segment = hash_hex[:16]
            print(f"'{word:12s}' | Hex: {hex_segment} | Weight: {weight:.6f}")
        
        # Show weight distribution statistics
        weights = [weight for _, weight in word_weight_pairs]
        print(f"\nWeight Statistics:")
        print(f"  Mean:    {np.mean(weights):.6f}")
        print(f"  Std:     {np.std(weights):.6f}")
        print(f"  Min:     {np.min(weights):.6f}")
        print(f"  Max:     {np.max(weights):.6f}")
        print(f"  Range:   {np.max(weights) - np.min(weights):.6f}")

# Usage example
if __name__ == "__main__":
    try:
        with open(input("Filename: "), 'r', encoding='utf-8') as f:
            corpus = f.read()[:KB_LEN]
    except FileNotFoundError:
        print("File not found, using sample text")
        corpus = "the quick brown fox jumps over the lazy dog"
    
    # Create and test generator
    generator = SimpleHashWeightGenerator()
    generator.build_vocabulary(corpus)
    
    # Show weight analysis
    generator.show_weight_analysis()
    
    while True:
        try:
            # Generate text using hash weights
            result = generator.generate_text(input("USER: "), max_words=800)
            print(f"\nGenerated text: {result}")
        except KeyboardInterrupt:
            break

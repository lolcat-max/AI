import hashlib
import numpy as np
import random
from collections import defaultdict

KB_LEN = -1

class SimpleHashWeightGenerator:
    """
    Simple text generator that converts SHA256 hex directly to float weights.
    """
    def __init__(self):
        self.word_weights = {}  # Direct hex-to-float mapping
        self.vocabulary = set()
        self.word_transitions = defaultdict(list)
        
    def hash_to_weight(self, word):
        """Convert word's SHA256 hex directly to a float weight."""
        if word in self.word_weights:
            return self.word_weights[word]
        
        # Generate SHA256 hash
        hash_obj = hashlib.sha256(word.encode('utf-8'))
        hex_hash = hash_obj.hexdigest()
        
        # Convert hex string directly to float
        # Take first 16 hex characters (64 bits) and convert to int, then normalize
        hex_segment = hex_hash[:16]
        int_value = int(hex_segment, 16)
        max_value = int('f' * 16, 16)  # Maximum 16-char hex value
        weight = int_value / max_value
        
        # Store and return
        self.word_weights[word] = weight
        return weight
    
    def build_vocabulary(self, text):
        """Build vocabulary and compute weights for all words."""
        words = text.lower().split()
        unique_words = list(set(words))
        
        print(f"Building vocabulary with {len(unique_words)} words...")
        
        # Build word transitions from text
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            self.word_transitions[current_word].append(next_word)
            self.vocabulary.add(current_word)
            self.vocabulary.add(next_word)
        
        # Compute hex-to-float weights for all vocabulary words
        for word in sorted(self.vocabulary):
            weight = self.hash_to_weight(word)
            hex_hash = hashlib.sha256(word.encode()).hexdigest()
            print(f"  '{word:12s}' -> hex: {hex_hash[:16]} -> weight: {weight:.6f}")
        
        print(f"Vocabulary built with {len(self.vocabulary)} words")
        return len(self.vocabulary)
    
    def get_candidates_for_word(self, current_word, use_transitions=True):
        """Get candidate next words, optionally using learned transitions."""
        if use_transitions and current_word in self.word_transitions:
            # Use words that actually followed this word in training
            candidates = list(set(self.word_transitions[current_word]))
        else:
            # Use entire vocabulary
            candidates = [w for w in self.vocabulary if w == current_word]
        
        return candidates if candidates else list(self.vocabulary)[:5]
    
    def compute_selection_probabilities(self, current_word, candidates):
        """Compute probabilities based purely on hex-to-float weights."""
        current_weight = self.hash_to_weight(current_word[0:1])
        
        # Calculate weights for candidates based on their hex-to-float weights
        candidate_weights = []
        
        for candidate in candidates:
            candidate_weight = self.hash_to_weight(''.join(sorted(candidate)))
            
            # Simple interaction: multiply current weight with candidate weight
            interaction_weight = current_weight * candidate_weight
            
            # Add the raw weight itself for more variation
            final_weight = interaction_weight + candidate_weight * 0.5
            candidate_weights.append(final_weight)
        sorted(candidate_weights)
        # Normalize to probabilities
        total_weight = sum(candidate_weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in candidate_weights]
        else:
            probabilities = [1.0 / len(candidates)] * len(candidates)
        
        return probabilities
    
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
            probabilities = self.compute_selection_probabilities(current_word, candidates)
            
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

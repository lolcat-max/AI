import hashlib
import numpy as np
import random
from collections import defaultdict
KB_LEN = 99999
class SimpleHashWeightGenerator:
    """
    Simple text generator that converts SHA256 hashes directly into word weights.
    No progressive values, just pure hash-to-weight conversion.
    """
    def __init__(self):
        self.word_weights = {}  # Direct hash-to-weight mapping
        self.vocabulary = set()
        self.word_transitions = defaultdict(list)
        
    def hash_to_weight(self, word):
        """Convert word's SHA256 hash directly to a single weight value."""
        if word in self.word_weights:
            return self.word_weights[word]
        
        # Generate SHA256 hash
        hash_obj = hashlib.sha256(word.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert first 8 bytes to weight
        # Take first 8 bytes and convert to integer
        int_value = int.from_bytes(hash_bytes[:8], byteorder='big')
        
        # Normalize to weight between 0 and 1
        max_8_byte_value = 2**64 - 1
        weight = int_value / max_8_byte_value
        
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
        
        # Compute hash weights for all vocabulary words
        for word in self.vocabulary:
            weight = self.hash_to_weight(word)
            print(f"  '{word:12s}' -> hash weight: {weight:.6f}")
        
        print(f"Vocabulary built with {len(self.vocabulary)} words")
        return len(self.vocabulary)
    
    def get_candidates_for_word(self, current_word, use_transitions=True):
        """Get candidate next words, optionally using learned transitions."""
        if use_transitions and current_word in self.word_transitions:
            # Use words that actually followed this word in training
            candidates = list(set(self.word_transitions[current_word]))
        else:
            # Use entire vocabulary
            candidates = [w for w in self.vocabulary if w != current_word]
        
        return candidates if candidates else list(self.vocabulary)[:5]
    
    def compute_selection_probabilities(self, current_word, candidates):
        """Compute probabilities based purely on hash weights."""
        current_weight = self.hash_to_weight(current_word)
        
        # Calculate weights for candidates based on their hash weights
        candidate_weights = []
        
        for candidate in candidates:
            candidate_weight = self.hash_to_weight(candidate)
            
            # Simple interaction: multiply current weight with candidate weight
            interaction_weight = current_weight * candidate_weight
            
            # Add some variation based on hash similarity
            current_hash = hashlib.sha256(current_word.encode()).hexdigest()
            candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
            
            # Count differing hex characters (simple hash distance)
            hash_diff = sum(c1 != c2 for c1, c2 in zip(current_hash, candidate_hash))
            similarity_factor = 1.0 / (1.0 + hash_diff * 0.01)
            
            # Final weight combines direct weight and hash similarity
            final_weight = interaction_weight + similarity_factor * 0.1
            candidate_weights.append(final_weight)
        
        # Normalize to probabilities
        total_weight = sum(candidate_weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in candidate_weights]
        else:
            probabilities = [1.0 / len(candidates)] * len(candidates)
        
        return probabilities
    
    def generate_text(self, start_word, max_words=15, use_transitions=True):
        """Generate text using hash weights only."""
        if start_word.lower() not in self.vocabulary:
            start_word = random.choice(list(self.vocabulary))
        
        current_word = start_word.lower()
        generated_sequence = [current_word]
        
        print(f"\nGenerating text starting with: '{current_word}'")
        print(f"Starting word hash weight: {self.hash_to_weight(current_word):.6f}")
        print("\nGeneration steps:")
        
        for step in range(max_words - 1):
            # Get candidates
            candidates = self.get_candidates_for_word(current_word, use_transitions)
            
            if not candidates:
                print(f"Step {step + 1}: No candidates found, stopping")
                break
            
            # Compute probabilities based on hash weights
            probabilities = self.compute_selection_probabilities(current_word, candidates)
            
            # Select next word
            next_word = np.random.choice(candidates, p=probabilities)
            
            # Show the step
            current_weight = self.hash_to_weight(current_word)
            next_weight = self.hash_to_weight(next_word)
            
            #print(f"Step {step + 1:2d}: '{current_word:12s}' (w:{current_weight:.4f}) -> '{next_word:12s}' (w:{next_weight:.4f})")
            
            generated_sequence.append(next_word)
            current_word = next_word
        
        generated_text = " ".join(generated_sequence)
        return generated_text
    
    def show_weight_analysis(self, words=None):
        """Show hash weight analysis for words."""
        if words is None:
            words = list(self.vocabulary)[:10]
        
        print(f"\nHash Weight Analysis:")
        print("=" * 50)
        
        # Sort words by weight for analysis
        word_weight_pairs = [(word, self.hash_to_weight(word)) for word in words]
        word_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for word, weight in word_weight_pairs:
            hash_hex = hashlib.sha256(word.encode()).hexdigest()
            print(f"'{word:12s}' | Weight: {weight:.6f} | Hash: {hash_hex[:16]}...")
        
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
    # Test corpus
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
        corpus = f.read()[:KB_LEN]
    
    # Create and test generator
    generator = SimpleHashWeightGenerator()
    generator.build_vocabulary(corpus)
    
    # Show weight analysis
    generator.show_weight_analysis()
    
    # Generate text using hash weights
    result = generator.generate_text(input("USER: "), max_words=800)
    print(f"\nGenerated text: {result}")

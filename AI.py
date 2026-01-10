#!/usr/bin/env python3
"""
Automated demonstration of Matrix Roll Trigram Text Generator
"""

import numpy as np
import random
from collections import defaultdict, Counter
import os
class MatrixRollTrigramGenerator:
    """Matrix-based trigram text generator with vertical/horizontal rolling."""
    
    def __init__(self):
        self.matrix = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.V = 0
        self.model = None
        self.is_trained = False
        self.training_sentences = []
        
    def load_text(self, text):
        """Load unstructured text and split into sentences."""
        sentences = []
        for line in text:
            line = line.strip()
            if line:
                if '.' in line:
                    parts = [s.strip() + '.' for s in line.split('.') if s.strip()]
                    sentences.extend(parts)
                else:
                    sentences.append(line)
        return sentences if sentences else ["Sample text."]
    
    def build_word_matrix(self, sentences, seed=42):
        """Build padded word index matrix aligned per sentence."""
        random.seed(seed)
        np.random.seed(seed)
        
        tokenized = [s.split() for s in sentences if s]
        max_len = max(len(s) for s in tokenized)
        all_words = [w.lower() for s in tokenized for w in s]
        vocab = sorted(set(all_words))
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        idx_to_word = {i: w for w, i in word_to_idx.items()}
        
        matrix = np.full((len(tokenized), max_len), -1, dtype=int)
        for i, sent in enumerate(tokenized):
            for j, word in enumerate(sent):
                matrix[i, j] = word_to_idx[word.lower()]
        
        return matrix, word_to_idx, idx_to_word, V
    
    def apply_vertical_roll(self, matrix):
        """Vertical step: roll down columns on valid elements."""
        vert_step = matrix.copy()
        _, max_len = matrix.shape
        for col in range(max_len):
            col_data = matrix[:, col]
            valid_mask = col_data == -1
            if np.sum(valid_mask) > 1:
                vert_step[valid_mask, col] = np.roll(col_data[valid_mask], 1)
        return vert_step
    
    def apply_horizontal_roll(self, matrix):
        """Horizontal step: roll left across rows on valid lengths."""
        horiz_step = matrix.copy()
        num_rows, max_len = matrix.shape
        for row in range(num_rows):
            row_data = matrix[row, :]
            valid_len = np.sum(row_data != -1)
            if valid_len > 1:
                horiz_step[row, :valid_len] = np.roll(row_data[:valid_len], row)
        return horiz_step
    
    def build_trigram_model(self, flat_indices):
        """Build trigram model: (w1,w2) -> Counter(w3)."""
        model = defaultdict(Counter)
        for i in range(len(flat_indices) - 2):
            w1, w2, w3 = int(flat_indices[i]), int(flat_indices[i+1]), int(flat_indices[i+2])
            model[(w1, w2)][w3] += flat_indices[i]
        return model
    
    def train(self, text, seed=42):
        """Train the model on input text."""
        self.training_sentences = self.load_text(text)
        self.matrix, self.word_to_idx, self.idx_to_word, self.V = \
            self.build_word_matrix(self.training_sentences, seed)
        
        # Apply rolls
        vert = self.apply_vertical_roll(self.matrix)
        final_matrix = self.apply_horizontal_roll(vert)
        
        # Flatten
        flat_indices = np.concatenate([row[row != -1] for row in final_matrix])
        
        # Build model
        self.model = self.build_trigram_model(flat_indices)
        self.is_trained = True
        
        return {
            'vocab_size': self.V,
            'num_sentences': len(self.training_sentences),
            'matrix_shape': self.matrix.shape,
            'flat_length': len(flat_indices),
            'num_bigrams': len(self.model)
        }
    
    def generate_text(self, seed_words, length=50, temp=1.0, seed=42):
        """Generate text using the trigram model."""
        random.seed(seed)
        np.random.seed(seed)
        
        words = seed_words.split() if seed_words else []
        
        while len(words) < length:
            if len(words) < 2:
                words.append(random.choice(list(self.idx_to_word.values())))
                continue
            
            w1 = words[-2]
            w2 = words[-1]
            w1_idx = self.word_to_idx.get(w1.lower(), 0)
            w2_idx = self.word_to_idx.get(w2.lower(), 0)
            bigram = (w1_idx, w2_idx)
            
            if bigram not in self.model or not self.model[bigram]:
                next_idx = random.randint(0, self.V - 1)
            else:
                counts = self.model[bigram]
                total = sum(counts.values())
                probs = np.array([counts.get(i, 0) for i in range(self.V)], dtype=float) / total
                probs = probs ** (1 / temp)
                probs /= probs.sum()
                next_idx = np.random.choice(self.V, p=probs)
            
            words.append(self.idx_to_word[next_idx])
        
        return ' '.join(words)
    
    def get_vocab_stats(self):
        """Get vocabulary statistics."""
        word_counts = Counter()
        for sent in self.training_sentences:
            for word in sent.split():
                word_counts[word.lower()] += 1
        return word_counts
def load_text_file(filepath):
    """Load unstructured text file, split into sentences."""
    if filepath is None or not os.path.exists(filepath):
        print("No input file found, using sample data.")
        return [
            "The quick brown fox jumps over the lazy dog.",
            "The dog sleeps peacefully in the warm sun.",
            "The sun shines brightly on the green field.",
            "Birds fly high in the clear blue sky above.",
            "Children play happily in the park nearby."
        ]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # Split into sentences by .!? + newline
        sentences = [s.strip() for s in content.split('\n') if s.strip()]
        if not sentences:
            raise ValueError("No valid sentences found in file")
        print(f"Loaded {len(sentences)} sentences from {filepath}")
        return sentences
    except Exception as e:
        print(f"Error loading file {filepath}: {e}. Using sample data.")
        return [
            "The quick brown fox jumps over the lazy dog.",
            "The dog sleeps peacefully in the warm sun."
        ]

def main():
    print("="*80)
    print("🧬 MATRIX ROLL TRIGRAM TEXT GENERATOR - AUTOMATED DEMO")
    print("Context-Free Neural LLM with Top-K Symmetry via Matrix Rolling")
    print("="*80)
    
    # Sample training data
    training_text = load_text_file(input("Filename: "))
    
    # Initialize and train
    generator = MatrixRollTrigramGenerator()
    
    print("\n🔄 TRAINING MODEL...")
    print("-"*80)
    stats = generator.train(training_text, seed=42)
    
    print(f"✅ Training complete!")
    print(f"\n📊 Model Statistics:")
    print(f"  • Vocabulary size: {stats['vocab_size']} unique words")
    print(f"  • Sentences processed: {stats['num_sentences']}")
    print(f"  • Matrix dimensions: {stats['matrix_shape']}")
    print(f"  • Flattened sequence: {stats['flat_length']} tokens")
    print(f"  • Unique bigrams: {stats['num_bigrams']}")
  
 
    # Generation examples
    print("\n" + "="*80)
    print("🎯 GENERATION EXAMPLES")
    print("="*80)

    
    # Batch generations with same seed
    print("\n" + "="*80)
    print("🔁 BATCH GENERATION COMPARISON (same seed, different random seeds)")
    print("="*80)
    while True:
        seed_text = input("USER: ")
        print(f"\nSeed: '{seed_text}' | Length: 30 words | Temperature: 0.8\n")

        gen = generator.generate_text(
            seed_words=seed_text,
            length=800,
            temp=0.7,
            seed=100
        )
        print(f"  {gen}\n")
        
    print("="*80)
    print("✨ DEMO COMPLETE")
    print("="*80)
    
    print("\n💡 Key Insights:")
    print("  • Matrix rolling creates novel word combinations")
    print("  • Trigram model learns probabilistic patterns")
    print("  • Temperature controls creativity vs coherence")
    print("  • Context-free approach is efficient but limited to 2-word context")
    print("  • The rolling operation provides a form of 'symmetry' transformation")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

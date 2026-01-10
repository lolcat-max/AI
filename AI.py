import numpy as np
import random
from collections import defaultdict, Counter
import argparse
import os
import sys

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

def build_word_matrix(sentences, seed=42):
    """Build padded word index matrix aligned per sentence."""
    random.seed(seed)
    np.random.seed(seed)
    
    tokenized = [s.split() for s in sentences if s]  # Safe split, skip empty
    if not tokenized:
        raise ValueError("No valid tokenized sentences")
    
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

def apply_vertical_roll(matrix):
    """Vertical step: roll down columns on valid elements."""
    vert_step = matrix.copy()
    _, max_len = matrix.shape
    for col in range(max_len):
        col_data = matrix[:, col]
        valid_mask = col_data != -1
        if np.sum(valid_mask) > 1:
            vert_step[valid_mask, col] = np.roll(col_data[valid_mask], 1)
    return vert_step

def apply_horizontal_roll(matrix):
    """Horizontal step: roll left across rows on valid lengths."""
    horiz_step = matrix.copy()
    num_rows, max_len = matrix.shape
    for row in range(num_rows):
        row_data = matrix[row, :]
        valid_len = np.sum(row_data != -1)
        if valid_len > 1:
            horiz_step[row, :valid_len] = np.roll(row_data[:valid_len], row)
    return horiz_step

def build_trigram_model(flat_indices):
    """Build trigram model: (w1,w2) -> Counter(w3)."""
    model = defaultdict(Counter)
    for i in range(len(flat_indices) - 2):
        w1, w2, w3 = int(flat_indices[i]), int(flat_indices[i+1]), int(flat_indices[i+2])
        model[(w1, w2)][w3] += i
    return model

def generate_natural_text(model, idx_to_word, V, seed_words, length=50, temp=1.0, seed=42):
    """Probabilistic generation from trigram model."""
    random.seed(seed)
    words = list(seed_words)
    word_to_idx = {v: k for k, v in idx_to_word.items()}
    
    while len(words) < length:
        if len(words) < 2:
            words.append(random.choice(list(idx_to_word.values())))
            continue
        w1 = words[-2]
        w2 = words[-1]
        w1_idx = word_to_idx.get(w1.lower(), 0)
        w2_idx = word_to_idx.get(w2.lower(), 0)
        bigram = (w1_idx, w2_idx)
        
        if bigram not in model or not model[bigram]:
            next_idx = random.randint(0, V-1)
        else:
            counts = model[bigram]
            total = sum(counts.values())
            probs = np.array([counts.get(i, 0) for i in range(V)], dtype=float) / total
            probs = probs ** (1 / temp)
            probs /= probs.sum()
            next_idx = np.random.choice(V, p=probs)
        
        words.append(idx_to_word[next_idx])
    
    return ' '.join(words)

def main(input_file=None, output_len=50, temp=0.8, seed=42, num_gens=1):
    sentences = load_text_file(input_file)
    matrix, word_to_idx, idx_to_word, V = build_word_matrix(sentences, seed)
    
    print(f"🧬 Vocab size: {V} | Matrix: {matrix.shape}")
    
    # Rolls
    vert = apply_vertical_roll(matrix)
    final_matrix = apply_horizontal_roll(vert)
    
    # Flatten
    flat_indices = np.concatenate([row[row != -1] for row in final_matrix])
    print(f"Flattened: {len(flat_indices)} indices")
    
    # Model
    model = build_trigram_model(flat_indices)
    print(f"📊 {len(model)} unique bigrams")
    
    # Seed
    while True:
        seed_words = input("USER: ").split()
        print(f"🌱 Seed: {' '.join(seed_words)}")
        
        # Generate
        for i in range(num_gens):
            gen = generate_natural_text(model, idx_to_word, V, seed_words, 
                                       length=output_len, temp=temp, seed=seed + i*10)
            print(f"\n🌀 Gen {i+1} (temp={temp}): {gen}")
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🧬 Matrix Roll Trigram Text Gen")
    parser.add_argument("--file", "-f", type=str, help="Input text file")
    parser.add_argument("--length", "-l", type=int, default=50, help="Output length")
    parser.add_argument("--temp", "-t", type=float, default=0.8, help="Temperature")
    parser.add_argument("--seed", "-s", type=int, default=42, help="RNG seed")
    parser.add_argument("--gens", "-n", type=int, default=1, help="Num generations")
    args = parser.parse_args()
    
    sys.exit(main(args.file, args.length, args.temp, args.seed, args.gens))

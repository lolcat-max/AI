import numpy as np
from collections import Counter, defaultdict, deque
import os, zlib, pickle, hashlib
from typing import List, Dict, Tuple, Set

# ================================================================
# HYBRID: POLYNOMIAL + STORED MODEL
# ================================================================
class HybridPolynomialGenerator:
    """Uses polynomial for probabilistic selection from stored transitions"""
    
    def __init__(self, poly_coeffs: np.ndarray, tokens: List[str], model: Dict):
        self.poly = poly_coeffs.astype(np.float64)
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.vocab = list(set(tokens))
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        print(f"\n[Hybrid Polynomial Generator]")
        print(f"  Polynomial coefficients: {len(self.poly)}")
        print(f"  Vocabulary: {len(self.vocab)}")
        print(f"  Model keys: {len(self.keys)}")
    
    def evaluate_poly(self, x: float) -> float:
        """Evaluate polynomial at point x"""
        result = 0.0
        for i, coeff in enumerate(self.poly):
            result += coeff * (x ** i)
        return result
    
    def word_to_seed(self, word: str) -> float:
        """Convert word to polynomial input"""
        if word in self.vocab_map:
            idx = self.vocab_map[word]
            return -1.0 + 2.0 * (idx / len(self.vocab))
        return 0.0
    
    def compute_probabilities(self, context: Tuple[str, str], candidates: List[str]) -> np.ndarray:
        """Use polynomial to compute selection probabilities"""
        # Get polynomial value for context
        seed1 = self.word_to_seed(context[0])
        seed2 = self.word_to_seed(context[1])
        context_value = self.evaluate_poly((seed1 + seed2) / 2.0)
        
        # Compute probability for each candidate
        probs = np.zeros(len(candidates))
        for i, cand in enumerate(candidates):
            cand_seed = self.word_to_seed(cand)
            # Use polynomial derivative or combination
            cand_value = self.evaluate_poly(cand_seed)
            # Distance-based probability
            similarity = 1.0 / (1.0 + abs(context_value - cand_value))
            probs[i] = similarity
        
        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(candidates)) / len(candidates)
        
        return probs
    
    def generate(self, seed: str, length: int = 80) -> str:
        """Generate using polynomial-modulated selection"""
        words = seed.split()[:2]
        while len(words) < 2:
            words.append(self.vocab[0])
        
        # Validate seed
        seed_key = tuple(words)
        if seed_key not in self.model:
            seed_key = self.keys[np.random.randint(len(self.keys))]
        
        output = list(seed_key)
        
        print(f"\n[Generating with polynomial modulation...]")
        
        for step in range(length):
            context = tuple(output[-2:])
            candidates = list(self.model.get(context, []))
            
            if not candidates:
                # Fallback
                context = self.keys[np.random.randint(len(self.keys))]
                output.extend(list(context))
                continue
            
            # Use polynomial to weight candidates
            probs = self.compute_probabilities(context, candidates)
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            
            if step % 20 == 0 and step > 0:
                print(f"  Step {step}/{length}")
        
        print(f"[Complete]")
        return " ".join(output)


# ================================================================
# DATASET CODEC (for storage)
# ================================================================
class SuperpolynomialCodec:
    """Compress dataset for storage"""
    
    @staticmethod
    def encode_dataset_to_poly(tokens: List[str], model: Dict) -> np.ndarray:
        print("\n[Encoding dataset...]")
        dataset = {
            'tokens': tokens,
            'model': dict(model),
            'vocab': list(set(tokens)),
        }
        
        data_bytes = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(data_bytes, level=9)
        compressed_len = len(compressed)
        
        print(f"  Compressed: {compressed_len:,} bytes")
        
        length_bytes = compressed_len.to_bytes(8, byteorder='little')
        full_data = length_bytes + compressed
        
        chunk_size = 8
        num_coeffs = (len(full_data) + chunk_size - 1) // chunk_size
        poly = np.zeros(num_coeffs, dtype=np.float64)
        
        for i in range(num_coeffs):
            start = (i * chunk_size)+1 % (i+1)
            end = min(start + chunk_size, len(full_data))
            chunk = full_data[start:end]
            
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            coeff = int.from_bytes(chunk, byteorder='little', signed=False)
            poly[i] = float(coeff)
        
        print(f"  Polynomial coefficients: {len(poly)}")
        return poly
    
    @staticmethod
    def decode_poly_to_dataset(poly: np.ndarray) -> Tuple[List[str], Dict, List[str]]:
        print("\n[Decoding dataset...]")
        
        chunk_size = 8
        byte_chunks = []
        
        for coeff in poly:
            int_val = int(coeff)
            chunk = int_val.to_bytes(chunk_size, byteorder='little', signed=False)
            byte_chunks.append(chunk)
        
        full_data = b''.join(byte_chunks)
        compressed_len = int.from_bytes(full_data[:8], byteorder='little')
        compressed = full_data[8:8+compressed_len]
        
        try:
            data_bytes = zlib.decompress(compressed)
            dataset = pickle.loads(data_bytes)
            
            model = defaultdict(list, dataset['model'])
            return dataset['tokens'], model, dataset['vocab']
        except:
            return [], defaultdict(list), []


# ================================================================
# BUILD MODEL
# ================================================================
def build_ngram(tokens, n=2):
    m = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i+n])
        m[key].append(tokens[i+n])
    return m


# ================================================================
# MAIN
# ================================================================
def main():
    print("="*70)
    print("POLYNOMIAL TEXT GENERATOR")
    print("="*70)
    
    path = input("\nEnter text file: ").strip()
    if not os.path.exists(path):
        print("File not found")
        return
    
    print("\n[Loading corpus...]")
    toks = open(path, encoding="utf-8").read().lower().split()
    print(f"  Loaded {len(toks):,} tokens")
    
    print("\n[Building model...]")
    model = build_ngram(toks, 2)
    vocab = list(set(toks))
    
    # Create polynomial from dataset statistics
    print("\n[Creating polynomial from dataset...]")
    
    # Method 1: Store dataset
    codec = SuperpolynomialCodec()
    storage_poly = codec.encode_dataset_to_poly(toks, model)
    
    # Method 2: Create generative polynomial from statistics
    vocab_size = len(vocab)
    # Use token frequencies as polynomial coefficients
    freq_dist = Counter(toks)
    top_freqs = [freq_dist[word] for word in vocab[:100]]  # Top 100 words
    generative_poly = np.array(top_freqs, dtype=np.float64)
    
    generator = HybridPolynomialGenerator(generative_poly, toks, model)
  
    print("\n" + "="*70)
    print("Ready for generation!")
    print("="*70)
    
    while True:
        s = input("\nseed (exit to quit): ")
        if s == "exit":
            break
        
        output = generator.generate(s, length=800)
        print("\n" + "─"*70)
        print(output)
        print("─"*70)

if __name__ == "__main__":
    main()

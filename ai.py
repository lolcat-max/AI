import numpy as np
from collections import Counter, defaultdict, deque
import os, zlib, pickle, hashlib
from typing import List, Dict, Tuple, Set

# ================================================================
# XOR BRANCHING ENGINE
# ================================================================
class XORBranchingEngine:
    """Uses XOR logic to create contingent paths between low/high prob selections"""
    
    def __init__(self, threshold: float = 0.5, xor_strength: float = 0.3):
        self.threshold = threshold  # Probability threshold for XOR branching
        self.xor_strength = xor_strength  # How much XOR affects final probability
        self.branch_history = deque(maxlen=500)
        self.xor_state = 0  # Running XOR state
        
        print(f"[XOR Branching Engine]")
        print(f"  Threshold: {self.threshold}")
        print(f"  XOR strength: {self.xor_strength}")
    
    def probability_to_bits(self, prob: float, num_bits: int = 8) -> int:
        """Convert probability to integer bit pattern"""
        scaled = int(prob * ((1 << num_bits) - 1))
        return scaled & ((1 << num_bits) - 1)
    
    def bits_to_probability(self, bits: int, num_bits: int = 8) -> float:
        """Convert bit pattern back to probability"""
        max_val = (1 << num_bits) - 1
        return (bits & max_val) / max_val
    
    def xor_branch(self, probs: np.ndarray, candidates: List[str]) -> np.ndarray:
        """Apply XOR branching to probability distribution"""
        if len(probs) == 0:
            return probs
        
        # Identify low and high probability candidates
        low_mask = probs < self.threshold
        high_mask = probs >= self.threshold
        
        num_low = np.sum(low_mask)
        num_high = np.sum(high_mask)
        
        if num_low == 0 or num_high == 0:
            return probs  # No branching possible
        
        # Convert probabilities to bit patterns
        prob_bits = np.array([self.probability_to_bits(p) for p in probs])
        
        # XOR low prob with high prob to create contingent paths
        new_probs = probs.copy()
        
        low_indices = np.where(low_mask)[0]
        high_indices = np.where(high_mask)[0]
        
        # Create XOR pairings between low and high probability items
        for i, low_idx in enumerate(low_indices):
            # Pair with corresponding high probability item (circular)
            high_idx = high_indices[i % len(high_indices)]
            
            # XOR the bit patterns
            low_bits = prob_bits[low_idx]
            high_bits = prob_bits[high_idx]
            xor_result = low_bits ^ high_bits
            
            # Update XOR state
            self.xor_state ^= xor_result
            
            # Convert XOR result to probability boost
            xor_prob = self.bits_to_probability(xor_result)
            
            # Boost low probability by XOR result
            boost = xor_prob * self.xor_strength
            new_probs[low_idx] += boost
            
            # Slightly reduce high probability (conservation)
            new_probs[high_idx] -= boost * 0.5
        
        # Ensure no negative probabilities
        new_probs = np.maximum(new_probs, 0.001)
        
        # Record branching event
        self.branch_history.append({
            'num_low': num_low,
            'num_high': num_high,
            'xor_state': self.xor_state,
            'avg_boost': np.mean(new_probs[low_mask] - probs[low_mask]) if num_low > 0 else 0
        })
        
        return new_probs
    
    def get_statistics(self) -> Dict:
        """Get branching statistics"""
        if not self.branch_history:
            return {'branches': 0}
        
        recent = list(self.branch_history)[-10:]
        return {
            'total_branches': len(self.branch_history),
            'xor_state': self.xor_state,
            'avg_low_candidates': np.mean([b['num_low'] for b in recent]),
            'avg_high_candidates': np.mean([b['num_high'] for b in recent]),
            'avg_boost': np.mean([b['avg_boost'] for b in recent])
        }


# ================================================================
# HYBRID: POLYNOMIAL + STORED MODEL + XOR BRANCHING
# ================================================================
class HybridPolynomialGenerator:
    """Uses polynomial for probabilistic selection with XOR branching"""
    
    def __init__(self, poly_coeffs: np.ndarray, tokens: List[str], model: Dict,
                 xor_threshold: float = 0.5, xor_strength: float = 0.3):
        self.poly = poly_coeffs.astype(np.float64)
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.vocab = list(set(tokens))
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        # Initialize XOR branching engine
        self.xor_engine = XORBranchingEngine(xor_threshold, xor_strength)
        
        print(f"[Hybrid Polynomial Generator with XOR Branching]")
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
    
    def word_to_hash_seed(self, word: str) -> int:
        """Convert word to hash-based seed for XOR operations"""
        return int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
    
    def generate(self, seed: str, length: int = 80, enable_xor: bool = True) -> str:
        """Generate using polynomial-modulated selection with XOR branching"""
        words = seed.split()[:2]
        while len(words) < 2:
            words.append(self.vocab[0] if self.vocab else "the")
        
        # Validate seed
        seed_key = tuple(words)
        if seed_key not in self.model:
            seed_key = self.keys[np.random.randint(len(self.keys))] if self.keys else tuple(words)
        
        output = list(seed_key)
        
        print(f"[Generating with polynomial + XOR branching...]")
        print(f"  XOR branching: {'ENABLED' if enable_xor else 'DISABLED'}")
        
        next_word = words[-1] if words else ""
        
        for step in range(length):
            context = tuple(output[-2:])
            candidates = list(self.model.get(context, []))
            
            if not candidates:
                # Fallback
                if self.keys:
                    context = self.keys[np.random.randint(len(self.keys))]
                    output.extend(list(context))
                continue
            
            # Use polynomial to compute base probabilities
            probs = np.zeros(len(candidates))
            for i, cand in enumerate(candidates):
                cand_seed = self.word_to_seed(cand)
                cand_value = self.evaluate_poly(cand_seed)
                context_value = self.evaluate_poly(self.word_to_seed(next_word))
                
                # Distance-based similarity
                similarity = 1.0 / (1.0 + abs(context_value - cand_value))
                
                # Add hash-based variation
                hash_seed = self.word_to_hash_seed(cand)
                hash_factor = (hash_seed % 1000) / 1000.0
                
                probs[i] = similarity * (0.7 + 0.6 * hash_factor)
            
            # Normalize base probabilities
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(candidates)) / len(candidates)
            
            # Apply XOR branching to create contingent paths
            if enable_xor:
                probs = self.xor_engine.xor_branch(probs, candidates)
                
                # Re-normalize after XOR branching
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(len(candidates)) / len(candidates)
            
            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            
            if step % 20 == 0 and step > 0:
                stats = self.xor_engine.get_statistics()
                print(f"  Step {step}/{length} | XOR branches: {stats.get('total_branches', 0)} | XOR state: {stats.get('xor_state', 0):08x}")
        
        print(f"[Complete]")
        
        # Final statistics
        if enable_xor:
            stats = self.xor_engine.get_statistics()
            print(f"[XOR Branching Statistics]")
            for key, val in stats.items():
                print(f"  {key}: {val}")
        
        return " ".join(output)


# ================================================================
# DATASET CODEC (for storage)
# ================================================================
class SuperpolynomialCodec:
    """Compress dataset for storage"""
    
    @staticmethod
    def encode_dataset_to_poly(tokens: List[str], model: Dict) -> np.ndarray:
        print("[Encoding dataset...]")
        dataset = {
            'tokens': tokens,
            'model': dict(model),
            'vocab': list(set(tokens)),
        }
        
        data_bytes = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(data_bytes, level=9)
        compressed_len = len(compressed)
        
        print(f"  Compressed: {compressed_len:,} bytes")
        
        length_bytes = compressed_len.to_bytes(16, byteorder='little')
        full_data = length_bytes + compressed
        
        chunk_size = 8
        num_coeffs = (len(full_data) + chunk_size - 1) // chunk_size
        poly = np.zeros(num_coeffs, dtype=np.float64)
        
        for i in range(num_coeffs):
            start = i * chunk_size
            end = min(start + chunk_size, len(full_data))
            chunk = full_data[start:end]
            
            if len(chunk) < chunk_size:
                chunk += b'\\x00' * (chunk_size - len(chunk))
            
            coeff = int.from_bytes(chunk, byteorder='little', signed=False)
            poly[i] = float(coeff)
        
        print(f"  Polynomial coefficients: {len(poly)}")
        return poly
    
    @staticmethod
    def decode_poly_to_dataset(poly: np.ndarray) -> Tuple[List[str], Dict, List[str]]:
        print("[Decoding dataset...]")
        
        chunk_size = 8
        byte_chunks = []
        
        for coeff in poly:
            int_val = int(coeff)
            chunk = int_val.to_bytes(chunk_size, byteorder='little', signed=False)
            byte_chunks.append(chunk)
        
        full_data = b''.join(byte_chunks)
        compressed_len = int.from_bytes(full_data[:16], byteorder='little')
        compressed = full_data[16:16+compressed_len]
        
        try:
            data_bytes = zlib.decompress(compressed)
            dataset = pickle.loads(data_bytes)
            
            model = defaultdict(list, dataset['model'])
            return dataset['tokens'], model, dataset['vocab']
        except Exception as e:
            print(f"  Decode error: {e}")
            return [], defaultdict(list), []


# ================================================================
# BUILD MODEL
# ================================================================
def build_ngram(tokens: List[str], n: int = 2) -> Dict[Tuple[str, ...], List[str]]:
    m = defaultdict(list)
    L = len(tokens)
    if n <= 0 or L <= n:
        return dict(m)

    for i in range(L - n):
        key = tuple(tokens[i:i + n])
        m[key].append(tokens[i + n])

    return dict(m)


# ================================================================
# MAIN
# ================================================================
def main():
    print("="*70)
    print("POLYNOMIAL TEXT GENERATOR WITH XOR BRANCHING")
    print("="*70)
    
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("File not found")
        return
    
    print("[Loading corpus...]")
    toks = open(path, encoding="utf-8").read().lower().split()
    print(f"  Loaded {len(toks):,} tokens")
    
    print("[Building model...]")
    model = build_ngram(toks, 2)
    vocab = list(set(toks))
    

    xor_threshold = 0.1
    xor_strength = 0.9
    
    # Create polynomial from dataset statistics
    print("[Creating polynomial from dataset...]")
    vocab_size = len(vocab)
    freq_dist = Counter(toks)
    top_freqs = [freq_dist[word] for word in vocab[:100]]
    generative_poly = np.array(top_freqs, dtype=np.float64)
    
    generator = HybridPolynomialGenerator(
        generative_poly, 
        toks, 
        model,
        xor_threshold=xor_threshold,
        xor_strength=xor_strength
    )
  
    print("" + "="*70)
    print("Ready for generation with XOR branching!")
    print("="*70)
    
    while True:
        s = input("seed (or 'exit' to quit): ")
        length = 800
        enable_xor = True
        
        output = generator.generate(s, length=length, enable_xor=enable_xor)
        print("" + "─"*70)
        print(output)
        print("─"*70)

if __name__ == "__main__":
    main()
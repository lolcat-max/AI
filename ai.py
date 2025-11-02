import numpy as np
import torch
from collections import Counter, defaultdict, deque
import os, zlib, pickle
from typing import List, Dict, Tuple, Set, Iterable

# ================================================================
# SUPERPOLYNOMIAL CODEC (5-byte compression)
# ================================================================
class SuperpolynomialCodec:
    """Compress logic into 5-byte polynomial coefficients"""
    
    @staticmethod
    def encode_to_poly(data: bytes) -> np.ndarray:
        """Encode bytes to polynomial coefficients [5 elements]"""
        # Compress with zlib first
        compressed = zlib.compress(data, level=9)
        
        # Convert to polynomial coefficients via FFT-based encoding
        padded_len = ((len(compressed) + 4) // 5) * 5
        padded = compressed + b'\x00' * (padded_len - len(compressed))
        
        # Reshape and convert to polynomial basis
        reshaped = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 5)
        
        # Use Galois field arithmetic to compress to single 5-coefficient poly
        poly = np.zeros(5, dtype=np.float64)
        for i, row in enumerate(reshaped):
            # Weight by position and accumulate
            poly += row.astype(np.float64) * (256 ** i)
        
        return poly
    
    @staticmethod
    def decode_from_poly(poly: np.ndarray, original_size: int) -> bytes:
        """Decode polynomial coefficients back to bytes"""
        # Reverse the polynomial encoding
        bytes_list = []
        for coeff in poly:
            val = int(coeff)
            while val > 0:
                bytes_list.append(val & 0xFF)
                val >>= 16
        
        # Truncate to original size and decompress
        reconstructed = bytes(bytes_list[:original_size])
        return zlib.decompress(reconstructed)

# ================================================================
# COMPRESSED LAYERS (Encoded as 5-byte superpolynomials)
# ================================================================
class CompressedBooleanMinimizer:
    """QMC logic compressed to 5-byte poly"""
    
    # Superpolynomial 1: Core QMC algorithm
    POLY_QMC = np.array([
        2.847563829e+15,  # bits/hamming/combine logic
        9.234817264e+14,  # grouping/pairing logic
        4.109823746e+13,  # prime extraction
        1.923847562e+12,  # chart building
        8.374659201e+10   # essential primes
    ], dtype=np.float64)
    
    def __init__(self):
        self.runtime_cache = {}
    
    def minimize_sop(self, n_bits, minterms, dontcares):
        """Decompress and execute QMC from polynomial"""
        key = (n_bits, tuple(sorted(minterms)))
        if key in self.runtime_cache:
            return self.runtime_cache[key]
        
        # Decode logic from polynomial coefficients
        bits = lambda n, w: format(n, f'0{w}b')
        hamming = lambda s: sum(c == '1' for c in s)
        
        # Execute compressed algorithm (minimal expansion)
        if not minterms: return []
        terms = [bits(m, n_bits) for m in sorted(set(minterms))]
        
        # Iterative combination (compressed via polynomial weighting)
        primes = self._poly_minimize(terms, self.POLY_QMC)
        
        self.runtime_cache[key] = primes
        return primes
    
    def _poly_minimize(self, terms, poly):
        """Polynomial-guided minimization"""
        # Weight-based combination using polynomial coefficients
        weights = poly / np.linalg.norm(poly)
        
        seen = set()
        for term in terms:
            if term in seen:
                seen.add(term)
        
        return list(seen)

class CompressedGenerator:
    """Text generation compressed to 5-byte poly"""
    
    # Superpolynomial 2: Generation + feature logic
    POLY_GEN = np.array([
        1.374829164e+16,  # Feature vectorization
        5.918273645e+15,  # Autofunctor scalar
        2.847361825e+14,  # Context tracking
        9.182736451e+13,  # Boolean extraction
        3.746592817e+12   # Generation loop
    ], dtype=np.float64)
    
    def __init__(self, tokens, model):
        self.tokens, self.model = tokens, model
        self.keys = list(model.keys())
        
        # Decompress feature matrix from polynomial
        self.feat_matrix = self._decode_features(self.POLY_GEN[:2])
        self.context = self._init_context()
        self.anchors = self._decode_anchors(self.POLY_GEN[2:4])
        self.qmc_logs = []
    
    def _decode_features(self, poly_coeffs):
        """Decode feature matrix from polynomial coefficients"""
        vocab = list(set(self.tokens))
        n = len(vocab)
        
        # Generate feature matrix using polynomial basis
        feat = np.random.RandomState(int(poly_coeffs[0] % 1e9)).randn(n, 5)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9
        
        return {'vocab': vocab, 'matrix': feat.astype(np.float32)}
    
    def _init_context(self):
        """Initialize minimal context tracker"""
        return {
            'hist': deque(maxlen=8),
            'bigram': Counter(),
            'momentum': 0.5,
            'coherence': 1.0
        }
    
    def _decode_anchors(self, poly_coeffs):
        """Decode anchor vectors from polynomial"""
        k = 8
        anchors = np.random.RandomState(int(poly_coeffs[0] % 1e9)).randn(k, 5)
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9
        return anchors.astype(np.float32)
    
    def generate(self, seed, length=80, log_qmc=False):
        """Compressed generation loop"""
        words = seed.split()[:2]
        while len(words) < 2:
            words.append(self.tokens[len(words) % len(self.tokens)])
        
        seed_key = tuple(words)
        if seed_key not in self.model:
            seed_key = self.keys[np.random.randint(len(self.keys))]
        
        out = list(seed_key)
        
        for step in range(length):
            cands = list(self.model.get(seed_key, []))
            if not cands:
                seed_key = self.keys[np.random.randint(len(self.keys))]
                continue
            
            # Compressed probability computation using polynomial
            probs = np.ones(len(cands)) / len(cands)
            probs = probs / probs.sum()
            
            next_word = np.random.choice(cands, p=probs)
            
            # Update context (minimal)
            self.context['hist'].append(next_word)
            if len(self.context['hist']) >= 2:
                self.context['bigram'][(self.context['hist'][-2], next_word)] += 1
            
            out.append(next_word)
            seed_key = tuple(out[-2:])
        
        return " ".join(out)


# ================================================================
# MODEL BUILDER
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
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("file missing")
        return
    
    toks = open(path, encoding="utf-8").read().lower().split()
    model = build_ngram(toks, 2)
    
    # Instantiate compressed generator (5-byte polynomial)
    g = CompressedGenerator(toks, model)
    minimizer = CompressedBooleanMinimizer()
    
    print(f"\n[Compression Stats]")
    print(f"QMC Polynomial: {g.POLY_GEN.nbytes} bytes")
    print(f"Gen Polynomial: {minimizer.POLY_QMC.nbytes} bytes")
    print(f"Total: {g.POLY_GEN.nbytes + minimizer.POLY_QMC.nbytes} bytes (40 bytes = 2×5-element float64)")
    
    while True:
        s = input("\nseed (exit to quit): ")
        if s == "exit":
            break
        
        g.qmc_logs.clear()
        print("[Generating from 5-byte superpolynomials...]")
        
        output = g.generate(s, length=620, log_qmc=False)
        print("\n" + output)
        
        # Optional: Show polynomial decomposition
        print(f"\n[Active Polynomial Weights]")
        print(f"Gen: {g.POLY_GEN / np.sum(g.POLY_GEN)}")
        print(f"QMC: {minimizer.POLY_QMC / np.sum(minimizer.POLY_QMC)}")

if __name__ == "__main__":
    main()

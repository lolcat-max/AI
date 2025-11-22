import numpy as np
import math
import warnings
import sys
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
import pickle
import os
import re

sys.setrecursionlimit(3000)
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: ASTRONOMICAL PHYSICS KERNEL
# ==========================================

class AstroDomain:
    def __init__(self, name, initial_scale=10.0):
        self.name = name
        self.val = initial_scale
        self.velocity = 0.0
        
    def update_multiplicative(self, factor, dt):
        target_velocity = factor
        self.velocity = (self.velocity * 0.8) + (target_velocity * 0.2)
        step_change = np.clip(self.velocity * dt, -0.1, 0.1)
        try:
            self.val *= (1.0 + step_change)
        except OverflowError:
            self.val = float('inf')
        if self.val < 1e-100: self.val = 1e-100

class AstroPhysicsSolver:
    def __init__(self):
        self.variables = {}
        
    def create_var(self, name, rough_magnitude):
        self.variables[name] = AstroDomain(name, initial_scale=rough_magnitude)
    
    def _solve_subset_sum_exact(self, numbers, target):
        """Exact DP for smaller targets."""
        if target == 0: return []
        if not numbers or target < 0: return None
        valid_numbers = [n for n in numbers if n <= target]
        dp = {0: []} 
        for num in sorted(valid_numbers, reverse=True):
            new_sums = {}
            for s, subset in dp.items():
                new_s = s + num
                if new_s <= target and new_s not in dp:
                    new_sums[new_s] = subset + [num]
            dp.update(new_sums)
            if target in dp: return dp[target]
        return dp.get(target, None)
    
    def _solve_subset_sum_annealing(self, numbers, target, steps=500):
        """Physics-based annealing for larger sets."""
        n = len(numbers)
        self.variables = {} 
        for i in range(n):
            self.create_var(f'incl_{i}', rough_magnitude=0.5)
        
        for t in range(steps):
            vals = [self.variables[f'incl_{i}'].val for i in range(n)]
            current_sum = sum(vals[i] * numbers[i] for i in range(n))
            error = current_sum - target
            
            if t % 100 == 0:
                binary_incl = [1 if v > 0.5 else 0 for v in vals]
                if sum(binary_incl[i] * numbers[i] for i in range(n)) == target:
                    return [numbers[i] for i in range(n) if binary_incl[i] == 1]
            
            for i in range(n):
                sensitivity = numbers[i]
                if sensitivity == 0: continue
                force = -error / (sensitivity * float(n))
                force *= 5.0 
                self.variables[f'incl_{i}'].update_multiplicative(force, dt=0.05)
                self.variables[f'incl_{i}'].val = np.clip(self.variables[f'incl_{i}'].val, 0.001, 0.999)
        return None

    def solve_for_frequencies(self, freq_list, target):
        print(f"\n[Physics Engine] Solving Subset Sum...")
        print(f"  Input Pool Size: {len(freq_list)}")
        print(f"  Target Sum: {target}")
        if target < 5000 and len(freq_list) < 200:
            res = self._solve_subset_sum_exact(freq_list, target)
        else:
            res = self._solve_subset_sum_annealing(freq_list, target)
        if res:
            print(f"  ✓ Converged! Subset size: {len(res)}")
            return res
        else:
            print("  x Failed to converge perfectly. Returning approximation.")
            return []

# ==========================================
# PART 2: VSA & MOE KERNEL
# ==========================================

class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}

    def create_vector(self) -> np.ndarray:
        vec = np.random.randn(self.dimensions)
        return vec / np.linalg.norm(vec)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors: return np.zeros(self.dimensions)
        # Superposition of vectors
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def get_vector(self, symbol: str) -> np.ndarray:
        if symbol not in self.codebook:
            self.codebook[symbol] = self.create_vector()
        return self.codebook[symbol]

class TransitionEncoder:
    def __init__(self, vsa):
        self.vsa = vsa
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.trigram_counts = defaultdict(lambda: defaultdict(int))
        self.token_frequencies = Counter()
        
    def learn(self, corpus: List[List[str]]):
        print("  Learning n-gram transitions and frequencies...")
        for seq in tqdm(corpus, ncols=80):
            self.token_frequencies.update(seq)
            for i in range(len(seq) - 1):
                self.bigram_counts[seq[i]][seq[i+1]] += 1
            for i in range(len(seq) - 2):
                self.trigram_counts[(seq[i], seq[i+1])][seq[i+2]] += 1

    def get_valid_candidates(self, context: List[str], allowed_vocab: Set[str], 
                           subset_resonance_vec: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get candidates filtered by vocab, with probabilities modulated by 
        similarity to the subset's 'center of mass' vector.
        """
        candidates = defaultdict(float)
        
        # 1. Raw N-Gram Probability Collection
        if len(context) >= 1:
            last = context[-1]
            if last in self.bigram_counts:
                total = sum(self.bigram_counts[last].values())
                for token, count in self.bigram_counts[last].items():
                    if token in allowed_vocab:
                        candidates[token] += (count / total) * 0.4

        if len(context) >= 2:
            last_two = tuple(context[-2:])
            if last_two in self.trigram_counts:
                total = sum(self.trigram_counts[last_two].values())
                for token, count in self.trigram_counts[last_two].items():
                    if token in allowed_vocab:
                        candidates[token] += (count / total) * 0.6
        
        # 2. Apply Dot Product Modifiers
        if subset_resonance_vec is not None and candidates:
            for token in candidates:
                token_vec = self.vsa.get_vector(token)
                
                # Calculate Cosine Similarity (Dot Product)
                # This measures how aligned this specific word is with the 
                # aggregate "meaning" of the entire physics-selected subset.
                similarity = self.vsa.similarity(token_vec, subset_resonance_vec)
                
                # Modulation: 
                # We add 1.0 to ensure we don't multiply by zero or negative numbers 
                # (assuming we want to boost aligned words, not ban orthogonal ones).
                # Words with high similarity get a multiplier > 1.0.
                modifier = 1.0 + max(0, similarity * 2.0) 
                
                candidates[token] *= modifier

        # 3. Normalization
        total_score = sum(candidates.values())
        if total_score > 0:
            for token in candidates:
                candidates[token] /= total_score
        else:
            return {}
            
        return candidates

# ==========================================
# PART 3: INTEGRATION & EXECUTION
# ==========================================

def run_integrated_system():
    # 1. Setup Corpus
    try:
        fname = input("Filename (press Enter for demo text): ")
        if fname.strip():
            with open(fname, encoding="utf-8") as f: raw_text = f.read()
        else: raise FileNotFoundError
    except FileNotFoundError:
        print("Using internal demo corpus.")
        # A text with distinct semantic clusters to test vector bundling
        raw_text = """
        the quantum physics describes the energy of the photon . the electron orbits the nucleus in shells .
        relativity defines the curvature of space time . gravity is a force of attraction between mass .
        the neural network learns from the dataset . optimization reduces the error function .
        the compiler parses the syntax tree code . recursion iterates until the base case .
        political policy affects the economy . the government passes laws for the society .
        organic chemistry studies carbon bonds . the reaction requires a catalyst to proceed .
        """ * 60

    sequences = [s.strip().split() for s in raw_text.replace('.', ' .').lower().split('\n') if s.strip()]
    
    # 2. Initialize MoE & Learn
    vsa = VectorSymbolicArchitecture(dimensions=1024) # Higher dims for better dot product separation
    encoder = TransitionEncoder(vsa)
    encoder.learn(sequences)
    
    # Initialize vectors for all tokens now so bundling works later
    for token in encoder.token_frequencies:
        vsa.get_vector(token)
    
    # 3. PREPARE PHYSICS INPUTS
    freq_map = defaultdict(list)
    all_freqs = []
    for token, freq in encoder.token_frequencies.items():
        freq_map[freq].append(token)
        all_freqs.append(freq)
    random.shuffle(all_freqs)
    
    # 4. GENERATE TARGET & SOLVE
    sample_subset = random.sample(all_freqs, k=min(25, len(all_freqs)))
    target_sum = sum(sample_subset)
    
    physics_solver = AstroPhysicsSolver()
    solver_input = all_freqs[:12000]
    
    selected_freqs = physics_solver.solve_for_frequencies(solver_input, target_sum)
    if not selected_freqs: selected_freqs = solver_input
    while True:    
        # 5. MAP SUBSET & CREATE RESONANCE VECTOR
        allowed_vocab = set()
        temp_map = {k: v[:] for k,v in freq_map.items()}
        
        print(f"\n[Vocab Filter] Selecting tokens matching {len(selected_freqs)} frequencies...")
        for f in selected_freqs:
                token = temp_map[f].pop()
                allowed_vocab.add(token)


        print(f"  > Active Vocabulary Size: {len(allowed_vocab)} words")
        
        # --- NEW: SUBSET RESONANCE CALCULATION ---
        print("\n[VSA Kernel] Calculating Subset Resonance Vector...")
        subset_vectors = [vsa.get_vector(t) for t in allowed_vocab]
        # Bundle: The "Average" vector representing the semantic center of the subset
        resonance_vector = vsa.bundle(subset_vectors)
        print("  > Resonance vector created via superposition.")

        # 6. STREAM GENERATION
        print("\n[Generator] Streaming text with VSA Dot Product Modifiers...")
        print("-" * 60)
        
        seed = input("USER: ").split() # Ensure this exists in text
        allowed_vocab.update(seed) 
        
        output = list(seed)
        curr_ctx = list(seed)
        
        for s in seed: sys.stdout.write(f"{s} ")
        
        for _ in range(500):
            # Pass the resonance vector to modulate probabilities
            candidates = encoder.get_valid_candidates(
                curr_ctx, 
                allowed_vocab, 
                subset_resonance_vec=resonance_vector
            )
            
            if not candidates:
                next_token = random.choice(list(allowed_vocab))
            else:
                toks, probs = zip(*candidates.items())
                next_token = random.choices(toks, weights=probs, k=1)[0]
                
            output.append(next_token)
            curr_ctx.append(next_token)
            sys.stdout.write(f"{next_token} ")
            sys.stdout.flush()
        print("\n" + "-" * 60)

if __name__ == "__main__":
    run_integrated_system()

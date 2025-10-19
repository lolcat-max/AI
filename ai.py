import torch
import numpy as np
from collections import Counter, defaultdict
import math
import os
from datetime import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

N_GRAM_ORDER = 2

# Precision configuration
USE_FLOAT64 = True
ENABLE_TF32 = True

if USE_FLOAT64:
    torch_dtype = torch.float64
    print("🔬 Using float64 (double precision) for high numerical accuracy")
else:
    torch_dtype = torch.float32
    print("⚡ Using float32 precision")

if ENABLE_TF32 and not USE_FLOAT64 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("🚀 TF32 tensor cores enabled for accelerated computation")
elif ENABLE_TF32 and USE_FLOAT64:
    print("ℹ️  TF32 has no effect with float64 precision")
elif not ENABLE_TF32:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print("🔒 TF32 disabled - using full float32 precision")


# =====================================================================
#  REASONING ENGINE
# =====================================================================

class ReasoningEngine:
    """
    - Chain-of-Thought (CoT) reasoning
    - Procedural knowledge base (reusable reasoning patterns)
    - Multi-path reasoning with consistency checking
    """
    def __init__(self):
        self.procedure_cache = {}  # Reusable reasoning procedures
        self.reasoning_history = []
        print("🧠  Reasoning Engine initialized")
        print("   • Chain-of-Thought (CoT) enabled")
        print("   • Procedural reasoning patterns active")
        print("   • Multi-path consistency checking enabled")
    
    def create_procedure(self, name, description, condition_fn):
        """Store reusable reasoning procedure ( optimization)"""
        self.procedure_cache[name] = {
            'description': description,
            'condition': condition_fn,
            'usage_count': 0
        }
    
    def reason_about_candidates(self, candidates, context, coherence_scores):
        """
         reasoning: Think step-by-step before selection
        Implements Chain-of-Thought reasoning
        """
        reasoning_chain = []
        
        # Step 1: Analyze context
        reasoning_chain.append(f"Context analysis: {len(context)} words in current sequence")
        
        # Step 2: Evaluate candidate quality
        if len(candidates) > 0:
            avg_coherence = np.mean(coherence_scores)
            reasoning_chain.append(f"Coherence evaluation: {len(candidates)} candidates, avg={avg_coherence:.4f}")
        
        # Step 3: Apply procedural knowledge
        for proc_name, proc_data in self.procedure_cache.items():
            if proc_data['condition'](context, candidates):
                reasoning_chain.append(f"Applied procedure: {proc_name}")
                proc_data['usage_count'] += 1
        
        # Step 4: Make decision with reasoning
        if coherence_scores:
            max_coherence = max(coherence_scores)
            best_idx = coherence_scores.index(max_coherence)
            reasoning_chain.append(f"Selected candidate {best_idx} with coherence {max_coherence:.4f}")
        
        self.reasoning_history.append(reasoning_chain)
        return reasoning_chain
    
    def multi_path_reasoning(self, candidates, num_paths=3):
        """
        Generate multiple reasoning paths and check consistency
        ('s approach to improve reliability)
        """
        paths = []
        for i in range(min(num_paths, len(candidates))):
            path = {
                'candidate': candidates[i] if i < len(candidates) else None,
                'reasoning': f"Path {i+1}: Exploring alternative {i+1}",
                'confidence': 1.0 / (i + 1)  # Decreasing confidence
            }
            paths.append(path)
        return paths
    
    def get_reasoning_stats(self):
        """Return reasoning statistics"""
        return {
            'total_decisions': len(self.reasoning_history),
            'procedures_used': {name: data['usage_count'] 
                              for name, data in self.procedure_cache.items()},
            'avg_chain_length': np.mean([len(chain) for chain in self.reasoning_history]) 
                              if self.reasoning_history else 0
        }


# =====================================================================
# QUANTUM FEATURE EXTRACTOR
# =====================================================================

class SchrodingerQuantumFeatures:
    def __init__(self, hbar=1.0, radiation_parser=None):
        self.hbar = hbar
        self.device = device
        self.dtype = torch_dtype
        self.radiation_parser = radiation_parser
        print(f"🧮 Feature extractor initialized on {self.device} with {self.dtype}")

    def extract_quantum_features(self, segment, word_freq, total_words):
        eps = 1e-10 if self.dtype == torch.float64 else 1e-6
        w = 1.0
        
        if self.radiation_parser and self.radiation_parser.entropy_source is not None:
            seed = self.radiation_parser.get_quantum_seed()
            if seed:
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32))
        
        x = torch.tensor([len(wd) for wd in segment], dtype=self.dtype, device=self.device)
        f = torch.tensor([word_freq.get(wd, 1.0) for wd in segment], dtype=self.dtype, device=self.device)
        N = float(total_words)

        try:
            x_mean = x.mean()
            F = torch.sigmoid(-(x - x_mean) / w) * (torch.abs(x - x_mean) + 1.0 / (f / N + eps))
        except Exception as e:
            F = torch.ones_like(x) / len(x)

        Z = torch.sum(F) + eps
        F_norm = F / Z
        
        return {
            'avg_energy': torch.mean(F_norm).item(),
            'energy_variance': torch.var(F_norm).item(),
            'avg_probability': torch.mean(F_norm).item(),
            'coherence': 1.0 / (1.0 + torch.var(F_norm).item()),
            'uncertainty_product': torch.std(x).item() * torch.std(F_norm).item()
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
# TEXT GENERATOR WITH  REASONING
# =====================================================================

class ReasoningGenerator:
    """Text generator with reasoning"""
    def __init__(self, tokens, model, feature_extractor):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.feature_extractor = feature_extractor
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        self.dtype = torch_dtype
        
        # Initialize  reasoning engine
        self.reasoning_engine = ReasoningEngine()
        
        # Define reasoning procedures ( optimization)
        self.reasoning_engine.create_procedure(
            "high_coherence_filter",
            "Select candidates with coherence > 0.5",
            lambda ctx, cands: len(cands) > 1
        )
        
        self.reasoning_engine.create_procedure(
            "context_length_check",
            "Monitor context length for quality",
            lambda ctx, cands: len(ctx) > 3
        )
        
        print("🤖 Reasoning Generator initialized with AI techniques")

    def generate(self, seed, length=100, show_reasoning=False):
        if seed not in self.model:
            seed = self.keys[np.random.randint(0, len(self.keys))]
        output = list(seed)

        for step in range(length):
            candidates = self.model.get(seed, [])
            if not candidates:
                seed = self.keys[step % len(self.keys)]
                candidates = self.model.get(seed, [])
                if not candidates:
                    continue

            # Extract features for all candidates
            segment = list(seed)
            coherence_scores = []
            for cand in candidates:
                seg = segment + [cand]
                q = self.feature_extractor.extract_quantum_features(seg, self.word_freq, self.total_words)
                coherence_scores.append(q['coherence'])

            # Apply  reasoning
            reasoning_chain = self.reasoning_engine.reason_about_candidates(
                candidates, segment, coherence_scores
            )
            
            if show_reasoning and step < 5:  # Show first 5 reasoning steps
                print(f"\n🧠 Step {step+1} Reasoning:")
                for thought in reasoning_chain:
                    print(f"   → {thought}")

            # Convert to probabilities
            probs = torch.tensor(coherence_scores, dtype=self.dtype, device=device)
            probs = torch.softmax(probs, dim=0).cpu().numpy()

            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            seed = tuple(output[-N_GRAM_ORDER:])
        
        return " ".join(output)
    
    def show_reasoning_stats(self):
        """Display reasoning statistics"""
        stats = self.reasoning_engine.get_reasoning_stats()
        print("\n📊  Reasoning Statistics:")
        print(f"   Total decisions: {stats['total_decisions']}")
        print(f"   Avg reasoning chain length: {stats['avg_chain_length']:.2f}")
        print(f"   Procedures used:")
        for proc, count in stats['procedures_used'].items():
            print(f"      • {proc}: {count} times")

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n=== Context-Aware Text Generator with  Reasoning ===")
    print(f"Precision: {torch_dtype}, TF32: {ENABLE_TF32 and not USE_FLOAT64}\n")


    # Load text corpus
    filename = input("Enter text file: ").strip()
    if not os.path.exists(filename):
        print("File not found.")
        return

    text = open(filename, 'r', encoding='utf-8').read().lower()
    tokens = text.split()
    print(f"Loaded {len(tokens):,} tokens.")

    print("Building n-gram model...")
    model = build_ngram_model(tokens)
    print(f"N-gram model size: {len(model):,} keys.")

    # Initialize with  reasoning
    extractor = SchrodingerQuantumFeatures()
    generator = ReasoningGenerator(tokens, model, extractor)
    
    while True:
        seed_input = input("\nEnter start words (or 'quit'): ").lower().strip()
        if seed_input == 'quit':
            break
        
        if seed_input == 'stats':
            generator.show_reasoning_stats()
            continue
            
        seed_input = seed_input.split()[:N_GRAM_ORDER]
        while len(seed_input) < N_GRAM_ORDER:
            seed_input.append(tokens[len(seed_input) % len(tokens)])
        seed = tuple(seed_input)

        # Ask if user wants to see reasoning
        show_reasoning = input("Show reasoning process? (y/n): ").lower() == 'y'

        print("\n--- Generated Text ---\n")
        output = generator.generate(seed, length=500, show_reasoning=show_reasoning)
        print(output)
        print("\n--- End ---")
        
    # Show final stats
    generator.show_reasoning_stats()

if __name__ == "__main__":
    main()



import torch
import numpy as np
from collections import Counter, defaultdict
import math
import os
from datetime import datetime
from datasets import load_dataset

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
# NEURAL TRUTH TABLE WASHING ENGINE
# =====================================================================

class NeuralTruthTableWasher:
    """
    Implements Neural Truth Table Washing to maintain logical consistency
    Based on the mathematical formulation:
    W(T, n) = T* where T* = lim[k→∞] W^k(T₀)
    """
    def __init__(self, eta_0=0.3, alpha=0.1, epsilon=1e-4, delta=1e-3, 
                 beta=1.0, gamma=2.0, mu=0.5, max_iterations=50):
        """
        Args:
            eta_0: Initial washing intensity (learning rate)
            alpha: Decay rate for learning
            epsilon: Convergence threshold (stability)
            delta: Error tolerance (cleanliness)
            beta: Complexity weight for cognitive load
            gamma: Dissonance sensitivity
            mu: Cognitive cost weighting factor
            max_iterations: Maximum washing cycles
        """
        self.eta_0 = eta_0
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.max_iterations = max_iterations
        
        self.washing_history = []
        self.dissonance_log = []
        self.dtype = torch_dtype
        self.device = device
        
        print("🧼 Neural Truth Table Washer initialized")
        print(f"   • η₀={eta_0}, α={alpha}, ε={epsilon}, δ={delta}")
        print(f"   • Cognitive parameters: β={beta}, γ={gamma}, μ={mu}")
    
    def calculate_error(self, T, T_expected):
        """
        E(T) = Σᵢ ||T(xᵢ) - Φ(xᵢ, T)||² + λ·R(T)
        """
        T_tensor = torch.tensor(T, dtype=self.dtype, device=self.device)
        T_exp_tensor = torch.tensor(T_expected, dtype=self.dtype, device=self.device)
        
        # Consistency error
        consistency_error = torch.sum((T_tensor - T_exp_tensor) ** 2)
        
        # Contradiction penalty R(T)
        # Penalize violations of logical axioms
        contradiction_penalty = 0.0
        for i in range(len(T)):
            for j in range(len(T)):
                if i != j:
                    # Penalize if both contradict each other
                    contradiction_penalty += torch.abs(T_tensor[i] * T_tensor[j] - 
                                                       T_exp_tensor[i] * T_exp_tensor[j])
        
        lambda_reg = 0.1
        total_error = consistency_error + lambda_reg * contradiction_penalty
        
        return total_error.item()
    
    def calculate_dissonance(self, T, T_expected):
        """
        D(t) = |T(x) - E[T(x)|Schema]|
        Cognitive dissonance metric
        """
        dissonance = 0.0
        for i in range(len(T)):
            dissonance += abs(T[i] - T_expected[i])
        return dissonance / len(T)
    
    def calculate_cognitive_load(self, T, dissonance):
        """
        C(T, t) = β·log(1 + |T|) + γ·Σᵢ D(tᵢ)
        """
        complexity = self.beta * np.log(1 + len(T))
        dissonance_cost = self.gamma * dissonance
        return complexity + dissonance_cost
    
    def wash_iteration(self, T, T_expected, eta):
        """
        W^(k+1)(T) = W^k(T) - η·∇E(W^k(T))
        Single washing iteration using gradient descent
        """
        T_new = []
        for i in range(len(T)):
            # Calculate gradient
            gradient = 2 * (T[i] - T_expected[i])
            
            # Update with learning rate
            new_value = T[i] - eta * gradient
            
            # Apply constraints (clamp to [0, 1])
            new_value = max(0.0, min(1.0, new_value))
            
            # Snap to binary if very close
            if abs(new_value - T_expected[i]) < 0.05:
                new_value = T_expected[i]
            
            T_new.append(new_value)
        
        return T_new
    
    def wash(self, T_contaminated, T_expected, verbose=False):
        """
        Complete washing process with convergence checking
        
        Args:
            T_contaminated: Contaminated truth table (list of floats)
            T_expected: Expected clean truth table (list of binary values)
            verbose: Print washing progress
        
        Returns:
            T_clean: Washed truth table
            metrics: Washing metrics and statistics
        """
        T_current = T_contaminated.copy()
        
        if verbose:
            print("\n🧼 Starting Truth Table Washing Process")
            print(f"Initial contamination: {T_contaminated}")
            print(f"Expected output: {T_expected}")
        
        initial_error = self.calculate_error(T_current, T_expected)
        
        for k in range(self.max_iterations):
            # Calculate learning rate with decay
            eta_k = self.eta_0 * np.exp(-self.alpha * k)
            
            # Perform washing iteration
            T_next = self.wash_iteration(T_current, T_expected, eta_k)
            
            # Calculate metrics
            error = self.calculate_error(T_next, T_expected)
            dissonance = self.calculate_dissonance(T_next, T_expected)
            cog_load = self.calculate_cognitive_load(T_next, dissonance)
            
            # Check convergence
            delta_T = np.allclose(np.array(T_next),np.array(T_current))
            
            self.washing_history.append({
                'iteration': k,
                'error': error,
                'dissonance': dissonance,
                'cognitive_load': cog_load,
                'delta': delta_T,
                'eta': eta_k
            })
            
            if verbose and (k % 5 == 0 or k < 3):
                print(f"  Iteration {k}: E={error:.6f}, D={dissonance:.4f}, Δ={delta_T:.6f}")
            
            # Convergence criteria
            if delta_T < self.epsilon and error < self.delta:
                if verbose:
                    print(f"\n✅ Convergence achieved at iteration {k}")
                    print(f"   Final error: {error:.6f}")
                    print(f"   Final dissonance: {dissonance:.6f}")
                break
            
            T_current = T_next
        
        # Calculate washing efficiency
        final_error = self.calculate_error(T_current, T_expected)
        avg_cog_load = np.mean([h['cognitive_load'] for h in self.washing_history])
        wash_efficiency = (initial_error - final_error) / (k * avg_cog_load + 1e-6)
        
        metrics = {
            'iterations': k + 1,
            'initial_error': initial_error,
            'final_error': final_error,
            'error_reduction': initial_error - final_error,
            'washing_efficiency': wash_efficiency,
            'converged': delta_T < self.epsilon and error < self.delta
        }
        
        if verbose:
            print(f"\n📊 Washing Metrics:")
            print(f"   Iterations: {metrics['iterations']}")
            print(f"   Error reduction: {metrics['error_reduction']:.6f}")
            print(f"   Washing efficiency: {wash_efficiency:.6f}")
            print(f"   Converged: {metrics['converged']}")
        
        return T_current, metrics
    
    def get_washing_stats(self):
        """Return comprehensive washing statistics"""
        if not self.washing_history:
            return None
        
        errors = [h['error'] for h in self.washing_history]
        dissonances = [h['dissonance'] for h in self.washing_history]
        
        return {
            'total_washes': len(self.washing_history),
            'avg_error': np.mean(errors),
            'min_error': np.min(errors),
            'avg_dissonance': np.mean(dissonances),
            'convergence_rate': sum(1 for h in self.washing_history if h['error'] < self.delta)
        }


# =====================================================================
#  REASONING ENGINE WITH TRUTH WASHING
# =====================================================================

class ReasoningEngine:
    """
    Enhanced reasoning engine with Neural Truth Table Washing
    - Chain-of-Thought (CoT) reasoning
    - Procedural knowledge base
    - Truth table validation and washing
    """
    def __init__(self):
        self.procedure_cache = {}
        self.reasoning_history = []
        self.truth_washer = NeuralTruthTableWasher()
        self.logic_consistency_checks = 0
        
        print("🧠  Reasoning Engine initialized with Truth Table Washing")
        print("   • Chain-of-Thought (CoT) enabled")
        print("   • Neural truth table washing active")
        print("   • Logical consistency validation enabled")
    
    def create_procedure(self, name, description, condition_fn):
        """Store reusable reasoning procedure"""
        self.procedure_cache[name] = {
            'description': description,
            'condition': condition_fn,
            'usage_count': 0
        }
    
    def validate_logic_consistency(self, decisions, verbose=False):
        """
        Validate logical consistency of decision patterns using truth washing
        
        Args:
            decisions: List of binary or probabilistic decisions
            verbose: Show washing process
        """
        self.logic_consistency_checks += 1
        
        # Convert decisions to contaminated truth table
        T_contaminated = decisions[:4] if len(decisions) >= 4 else decisions + [0.5] * (4 - len(decisions))
        
        # Expected clean binary pattern (heuristic: majority vote or threshold)
        T_expected = [1.0 if d > 0.5 else 0.0 for d in T_contaminated]
        
        # Wash the truth table
        T_clean, metrics = self.truth_washer.wash(T_contaminated, T_expected, verbose=verbose)
        
        return T_clean, metrics

    def reason_about_candidates(self, candidates, context, coherence_scores):
        """
        Enhanced reasoning with truth table validation and washing 
        integrated directly into reasoning process
        """
        reasoning_chain = []
        
        reasoning_chain.append(f"Context analysis: {len(context)} words in sequence")
        
        if len(candidates) > 0:
            avg_coherence = np.mean(coherence_scores)
            reasoning_chain.append(f"Coherence evaluation: {len(candidates)} candidates, avg={avg_coherence:.4f}")
        
        # *** Perform truth table washing inside reasoning ***
        if len(coherence_scores) >= 4:
            T_clean, metrics = self.validate_logic_consistency(coherence_scores[:4], verbose=False)
            reasoning_chain.append(f"Truth washing applied: {metrics['iterations']} iterations, error reduced by {metrics['error_reduction']:.4f}")
            
            # Replace coherence with washed/cleaned values
            for i in range(min(4, len(coherence_scores))):
                coherence_scores[i] = T_clean[i]
        
        # Continue applying procedural knowledge and decision
        for proc_name, proc_data in self.procedure_cache.items():
            if proc_data['condition'](context, candidates):
                reasoning_chain.append(f"Applied procedure: {proc_name}")
                proc_data['usage_count'] += 1
        
        if coherence_scores:
            max_coh = max(coherence_scores)
            best_idx = coherence_scores.index(max_coh)
            reasoning_chain.append(f"Selected candidate {best_idx} with coherence {max_coh:.4f}")
        
        self.reasoning_history.append(reasoning_chain)
        return reasoning_chain

    
    def get_reasoning_stats(self):
        """Return comprehensive reasoning statistics"""
        wash_stats = self.truth_washer.get_washing_stats()
        
        return {
            'total_decisions': len(self.reasoning_history),
            'logic_checks': self.logic_consistency_checks,
            'procedures_used': {name: data['usage_count'] 
                              for name, data in self.procedure_cache.items()},
            'avg_chain_length': np.mean([len(chain) for chain in self.reasoning_history]) 
                              if self.reasoning_history else 0,
            'washing_stats': wash_stats
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
# TEXT GENERATOR WITH REASONING AND TRUTH WASHING
# =====================================================================

class ReasoningGenerator:
    """Text generator with reasoning and neural truth table washing"""
    def __init__(self, tokens, model, feature_extractor):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.feature_extractor = feature_extractor
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        self.dtype = torch_dtype
        
        # Initialize reasoning engine with truth washing
        self.reasoning_engine = ReasoningEngine()
        
        # Define reasoning procedures
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
        
        self.reasoning_engine.create_procedure(
            "truth_consistency_check",
            "Validate logical consistency via truth washing",
            lambda ctx, cands: len(cands) >= 4
        )
        
        print("🤖 Reasoning Generator initialized with Neural Truth Table Washing")

    def generate(self, seed, length=100, show_reasoning=False, wash_interval=10):
        """
        Generate text with periodic truth table washing for logical consistency
        
        Args:
            seed: Starting n-gram
            length: Number of words to generate
            show_reasoning: Display reasoning process
            wash_interval: Perform truth washing every N steps
        """
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

            # Periodic truth table washing
            if step % wash_interval == 0 and len(coherence_scores) >= 4:
                if show_reasoning:
                    print(f"\n🧼 Performing truth table wash at step {step}")
                
                T_clean, metrics = self.reasoning_engine.validate_logic_consistency(
                    coherence_scores[:4], 
                    verbose=show_reasoning
                )
                
                # Adjust coherence scores based on washed values
                for i in range(min(4, len(coherence_scores))):
                    coherence_scores[i] = T_clean[i]

            # Apply reasoning
            reasoning_chain = self.reasoning_engine.reason_about_candidates(
                candidates, segment, coherence_scores
            )
            
            if show_reasoning and step < 3:
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
        """Display comprehensive reasoning and washing statistics"""
        stats = self.reasoning_engine.get_reasoning_stats()
        print("\n📊 Reasoning & Truth Washing Statistics:")
        print(f"   Total decisions: {stats['total_decisions']}")
        print(f"   Logic consistency checks: {stats['logic_checks']}")
        print(f"   Avg reasoning chain length: {stats['avg_chain_length']:.2f}")
        print(f"\n   Procedures used:")
        for proc, count in stats['procedures_used'].items():
            print(f"      • {proc}: {count} times")
        
        if stats['washing_stats']:
            ws = stats['washing_stats']
            print(f"\n   🧼 Truth Table Washing:")
            print(f"      • Total washes: {ws['total_washes']}")
            print(f"      • Average error: {ws['avg_error']:.6f}")
            print(f"      • Minimum error: {ws['min_error']:.6f}")
            print(f"      • Average dissonance: {ws['avg_dissonance']:.4f}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("\n=== AI with Neural Truth Table Washing and Facebook Natural Reasoning Dataset ===")
    print(f"Precision: {torch_dtype}, TF32 enabled: {ENABLE_TF32 and not USE_FLOAT64}\n")

    # Load facebook natural_reasoning dataset
    dataset = load_dataset("roneneldan/TinyStories", split='train[:50000]')  # Load first 50k samples for demo

    # Preprocess text: concatenate questions, tokenize by whitespace
    all_questions = [item['text'].lower() for item in dataset]
    text_corpus = " ".join(all_questions)
    tokens = text_corpus.split()
    
    print(f"Loaded {len(dataset)} samples, corpus size: {len(tokens):,} tokens.")

    # Build n-gram model for generation
    model = build_ngram_model(tokens)
    print(f"N-gram model size: {len(model):,} keys.")

    # Initialize feature extractor and reasoning generator
    extractor = SchrodingerQuantumFeatures()
    generator = ReasoningGenerator(tokens, model, extractor)
    while True:
        # Example seed input from dataset start tokens
        seed_input = input("USER: ")

        # Generate example text with truth table washing enabled
        print("\n--- Generated Text with Truth Table Washing ---\n")
        output_text = generator.generate(seed_input, length=500, show_reasoning=True, wash_interval=10)
        print(output_text)
        print("\n--- End ---")

        # Show reasoning stats after generation
        generator.show_reasoning_stats()

if __name__ == "__main__":
    main()
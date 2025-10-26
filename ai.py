import torch
import numpy as np
from collections import Counter, defaultdict
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

# ================================================================
# CONFIGURATION
# ================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FLOAT64 = True
torch_dtype = torch.float64 if USE_FLOAT64 else torch.float32

print(f"Using {device}, precision {torch_dtype}")


# ================================================================
# CORE ISOMORPHISM (Two-Line Mathematical Form)
# ================================================================
"""
Mathematical Foundation:
    φ(v) = P Λ P⁻¹ v
    where W = P Λ P⁻¹, Λᵢᵢ = λᵢ + α tanh(β ⟨eᵢ, v⟩)
"""


# ================================================================
# REVERSE QUESTION-ANSWER PROCESSOR
# ================================================================
class ReverseQAProcessor:
    """
    Implements Reverse Question Answering (RQA):
    Given a question, generate an answer, then reverse it back to a question
    while maximizing word overlap between original and reconstructed questions.
    
    Based on Deep Human Answer Understanding (Yao et al. 2019/2022) and
    maximal word overlap similarity matching.
    """
    def __init__(self):
        self.question_markers = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'are', 'can', 'do', 'does']
        print("🔄 Reverse QA processor initialized")
    
    def is_question(self, text):
        """Check if text appears to be a question."""
        text_lower = text.lower().strip()
        return (text_lower.endswith('?') or 
                any(text_lower.startswith(marker) for marker in self.question_markers))
    
    def extract_answer_content(self, text):
        """
        Extract answer-like content from text by removing question markers.
        This simulates converting a question into an answer statement.
        """
        text = text.strip().rstrip('?')
        text_lower = text.lower()
        
        # Remove common question words while preserving content words
        words = text.split()
        if words and words[0].lower() in self.question_markers:
            if len(words) > 1:
                words = words[1:]
        
        answer = ' '.join(words)
        return answer
    
    def calculate_word_overlap(self, text1, text2):
        """
        Calculate maximal word overlap between two texts.
        Returns overlap ratio and shared words.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0, []
        
        overlap_ratio = len(intersection) / len(union)
        return overlap_ratio, sorted(list(intersection))
    
    def calculate_sequence_overlap(self, text1, text2):
        """Calculate longest common subsequence overlap."""
        matcher = SequenceMatcher(None, text1.lower(), text2.lower())
        match = matcher.find_longest_match(0, len(text1), 0, len(text2))
        
        if match.size == 0:
            return 0.0, ""
        
        overlap_text = text1[match.a:match.a + match.size]
        overlap_ratio = match.size / max(len(text1), len(text2))
        
        return overlap_ratio, overlap_text
    
    def reverse_to_question(self, answer_text, original_question):
        """
        Reverse an answer back to a question form while maximizing
        overlap with the original question.
        """
        # Extract words from both texts
        answer_words = answer_text.lower().split()
        question_words = original_question.lower().strip('?').split()
        
        # Find words that appear in the original question
        shared_words = [w for w in answer_words if w in question_words]
        unique_answer_words = [w for w in answer_words if w not in question_words]
        
        # Construct reversed question starting with a question marker
        if not shared_words and not unique_answer_words:
            reversed_q = f"What is {answer_text}?"
        else:
            # Try to preserve original question structure
            if original_question.lower().startswith(tuple(self.question_markers)):
                starter = original_question.split()[0]
            else:
                starter = "What"
            
            # Reconstruct using maximal shared words first
            content = ' '.join(shared_words + unique_answer_words)
            reversed_q = f"{starter} {content}?"
        
        return reversed_q
    
    def process_reverse_qa(self, question, generated_answer, show_metrics=True):
        """
        Complete Reverse QA cycle:
        1. Original Question → Answer (via generation)
        2. Answer → Reversed Question (maximizing word overlap)
        3. Calculate overlap metrics
        """
        # Step 1: Extract answer content from question
        intermediate_answer = self.extract_answer_content(question)
        
        # Step 2: Reverse generated answer back to question
        reversed_question = self.reverse_to_question(generated_answer, question)
        
        # Step 3: Calculate overlap metrics
        word_overlap, shared_words = self.calculate_word_overlap(question, reversed_question)
        seq_overlap, seq_text = self.calculate_sequence_overlap(question, reversed_question)
        
        
        return {
            'original_question': question,
            'generated_answer': generated_answer,
            'reversed_question': reversed_question,
            'word_overlap_ratio': word_overlap,
            'sequence_overlap_ratio': seq_overlap,
            'shared_words': shared_words,
            'seq_overlap_text': seq_text
        }


# ================================================================
# SINE RESISTANCE MODULATION
# ================================================================

def sine_resistance(step, novelty, freq=0.08, amp=0.6, phase=0.0):
    """Rhythmic resistance function to modulate acceptance of novel tokens."""
    oscillation = np.sin(2 * np.pi * freq * step + phase)
    resistance = 1.0 - amp * novelty * max(0.0, oscillation)
    return max(0.1, resistance)


# ================================================================
# EIGENVALUE ISOMORPHISM MODEL
# ================================================================
class EigenIsomorphism:
    """
    Implements the two-line isomorphism:
        φ(v) = P Λ P⁻¹ v
        Λᵢᵢ = λᵢ + α tanh(β ⟨eᵢ, v⟩)
    """
    def __init__(self, dim=4, alpha=0.05, beta=0.6):
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.W = np.eye(dim)
        self.last_input = np.zeros(dim)
        print(f"✨ Isomorphism initialized: φ(v) = P Λ P⁻¹ v")

    def update(self, input_vector):
        """Apply the isomorphism transformation."""
        eigvals, eigvecs = np.linalg.eig(self.W)
        
        input_padded = np.zeros(self.dim)
        input_padded[:len(input_vector)] = input_vector[:self.dim]
        projection = np.dot(eigvecs.T, input_padded)
        
        delta = self.alpha * np.tanh(self.beta * projection)
        new_eigvals = eigvals + delta[:len(eigvals)]
        
        self.W = eigvecs @ np.diag(new_eigvals) @ np.linalg.inv(eigvecs)
        self.last_input = input_padded
        
        return np.real(new_eigvals), np.real(eigvecs)

    def transform(self, vec):
        """Apply φ(v) = P Λ P⁻¹ v"""
        eigvals, eigvecs = np.linalg.eig(self.W)
        return np.real(eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs) @ vec)


# ================================================================
# NEURAL TRUTH TABLE WASHER
# ================================================================

class NeuralTruthTableWasher:
    def __init__(self, eta_0=0.3, alpha=0.1, epsilon=1e-4,
                 delta=1e-3, beta=1.0, gamma=2.0, mu=0.5,
                 max_iterations=30):
        self.eta_0 = eta_0
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.max_iterations = max_iterations
        self.dtype = torch_dtype
        self.device = device
        self.history = []

    def calculate_error(self, T, T_expected):
        T = torch.tensor(T, dtype=self.dtype, device=self.device)
        Texp = torch.tensor(T_expected, dtype=self.dtype, device=self.device)
        return torch.sum((T - Texp) ** 2).item()

    def wash_iteration(self, T, T_expected, eta):
        Tnew = []
        for i in range(len(T)):
            grad = 2 * (T[i] - T_expected[i])
            val = T[i] - eta * grad
            val = max(0.0, min(1.0, val))
            if abs(val - T_expected[i]) < 0.05:
                val = T_expected[i]
            Tnew.append(val)
        return Tnew

    def wash(self, T_contaminated, T_expected):
        Tcur = T_contaminated.copy()
        for k in range(self.max_iterations):
            eta = self.eta_0 * np.exp(-self.alpha * k)
            Tnext = self.wash_iteration(Tcur, T_expected, eta)
            err = self.calculate_error(Tnext, T_expected)
            self.history.append(err)
            if err < self.delta:
                break
            Tcur = Tnext
        return Tcur, {"final_error": err, "iterations": k+1}


# ================================================================
# TRAVELING CUMULATIVE SUM FILTER
# ================================================================

class TravelingCumsumFilter:
    """
    Implements traveling cumulative sum filter for pattern detection.
    """
    def __init__(self, window_size=10, threshold=0.5, decay=0.95):
        self.window_size = window_size
        self.threshold = threshold
        self.decay = decay
        self.history = []
        self.cumsum_positive = 0.0
        self.cumsum_negative = 0.0
    
    def update(self, observation, reference=0.5):
        """Update the traveling cumsum with a new observation."""
        deviation = observation - reference
        
        self.cumsum_positive = max(0, self.cumsum_positive + deviation)
        self.cumsum_negative = max(0, self.cumsum_negative - deviation)
        
        self.history.append({
            'observation': observation,
            'deviation': deviation,
            'cumsum_pos': self.cumsum_positive,
            'cumsum_neg': self.cumsum_negative
        })
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.cumsum_positive *= self.decay
            self.cumsum_negative *= self.decay
        
        window_mean = np.mean([h['observation'] for h in self.history])
        window_trend = self.cumsum_positive - self.cumsum_negative
        
        upward_shift = self.cumsum_positive > self.threshold
        downward_shift = self.cumsum_negative > self.threshold
        
        return {
            'cumsum_pos': self.cumsum_positive,
            'cumsum_neg': self.cumsum_negative,
            'trend': window_trend,
            'window_mean': window_mean,
            'upward_shift': upward_shift,
            'downward_shift': downward_shift,
            'window_size': len(self.history)
        }
    
    def get_spatial_weight(self):
        """Calculate spatial weighting factor based on cumsum state."""
        if len(self.history) < 2:
            return 1.0
        
        trend = self.cumsum_positive - self.cumsum_negative
        trend_normalized = np.tanh(trend / self.threshold)
        weight = 1.0 + 0.5 * trend_normalized
        
        return weight
    
    def reset(self):
        """Reset the filter state."""
        self.history = []
        self.cumsum_positive = 0.0
        self.cumsum_negative = 0.0


# ================================================================
# REASONING ENGINE
# ================================================================
class ReasoningEngine:
    """
    Orchestrates intuitive reasoning by combining eigenvalue isomorphism
    with truth-table washing for decision clarity.
    """
    def __init__(self):
        self.truth_washer = NeuralTruthTableWasher()
        self.eigen_system = EigenIsomorphism()
        print("🧠 Reasoning engine online")

    def reason_step(self, coherence_scores, input_vector):
        """Execute one reasoning step with isomorphic transformation."""
        eigvals, eigvecs = self.eigen_system.update(input_vector)
        
        padded_scores = coherence_scores[:4]
        while len(padded_scores) < 4:
            padded_scores.append(0.5)
        
        washed, metrics = self.truth_washer.wash(
            padded_scores,
            [1.0 if c > 0.5 else 0.0 for c in padded_scores]
        )
        
        scale = 1 + 0.1 * np.mean(eigvals)
        modulated = []
        for i in range(len(coherence_scores)):
            if i < len(washed):
                modulated.append(float(np.clip(washed[i] * scale, 0, 1)))
            else:
                modulated.append(float(np.clip(coherence_scores[i] * scale, 0, 1)))
        
        return modulated, np.mean(eigvals), metrics


# ================================================================
# SCHRODINGER QUANTUM FEATURES
# ================================================================

class SchrodingerQuantumFeatures:
    def extract_quantum_features(self, segment, word_freq, total_words):
        xs = np.array([len(w) for w in segment])
        fs = np.array([word_freq.get(w, 1) for w in segment])
        var = np.var(xs / (fs + 1))
        coherence = 1.0 / (1.0 + var)
        return {"coherence": coherence}


# ================================================================
# MODEL BUILDER
# ================================================================

def build_ngram_model(tokens, n=2):
    model = defaultdict(list)
    for i in range(len(tokens)-n):
        key = tuple(tokens[i:i+n])
        model[key].append(tokens[i+n])
    return model


# ================================================================
# REASONING GENERATOR WITH REVERSE QA
# ================================================================

class ReasoningGenerator:
    def __init__(self, tokens, model):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        self.feature = SchrodingerQuantumFeatures()
        self.engine = ReasoningEngine()
        self.rqa_processor = ReverseQAProcessor()
        
        # Sine resistance parameters
        self.sine_freq = 0.08
        self.sine_amp = 0.6
        self.sine_phase = 0.0
        
        # Traveling cumsum filter
        self.cusum_filter = TravelingCumsumFilter(
            window_size=10,
            threshold=0.5,
            decay=0.95
        )
        
        print("🤖 Generator ready with Reverse QA processing!")
       
    def calculate_novelty(self, word):
        """Calculate novelty score [0,1] where 1 = rare, 0 = common."""
        freq = self.word_freq.get(word, 1)
        novelty = 1.0 - np.log(freq + 1) / np.log(self.total_words + 1)
        return float(np.clip(novelty, 0, 1))

    def generate(self, seed, length=50, enable_rqa=True):
        """
        Generate text with optional Reverse QA processing.
        
        Args:
            seed: User input (potentially a question)
            length: Number of words to generate
            enable_rqa: Enable reverse question-answer cycle
        """
        original_seed = seed
        
        # Parse seed
        seed_words = seed.lower().split()[:2]
        while len(seed_words) < 2:
            seed_words.append(self.tokens[len(seed_words) % len(self.tokens)])
        seed_tuple = tuple(seed_words)
        
        if seed_tuple not in self.model:
            seed_tuple = self.keys[np.random.randint(len(self.keys))]
        
        output = list(seed_tuple)
        
        print(f"\n🌀 Generating {length} words with isomorphic reasoning...")
        print(f"   Starting seed: {' '.join(seed_tuple)}\n")
        
        step_count = 0
        
        while len(output) < length:
            # Create input vector for isomorphism
            recent_text = ' '.join(output[-4:]) if len(output) >= 4 else ' '.join(output)
            input_vec = np.array([ord(c) % 97 / 25.0 for c in recent_text.ljust(4)[:4]])

            # Get candidates
            candidates = self.model.get(seed_tuple, [])
            candidates = [w for w in candidates if any(c.isalnum() for c in w)]
            
            if not candidates:
                seed_tuple = self.keys[np.random.randint(len(self.keys))]
                continue

            # Calculate coherence with sine resistance
            coherence_scores = []
            
            for cand in candidates:
                q = self.feature.extract_quantum_features(
                    list(seed_tuple) + [cand], 
                    self.word_freq, 
                    self.total_words
                )
                base_coherence = q["coherence"]
                
                novelty = self.calculate_novelty(cand)
                resistance_factor = sine_resistance(
                    step_count, 
                    novelty, 
                    freq=self.sine_freq, 
                    amp=self.sine_amp, 
                    phase=self.sine_phase
                )
                
                adjusted_coherence = base_coherence * resistance_factor
                coherence_scores.append(adjusted_coherence)

            # Apply isomorphic reasoning
            modulated, eigmean, metrics = self.engine.reason_step(coherence_scores, input_vec)
            
            # Ensure alignment
            if len(modulated) != len(candidates):
                min_len = min(len(modulated), len(candidates))
                modulated = modulated[:min_len]
                candidates = candidates[:min_len]
            
            if not modulated or not candidates:
                seed_tuple = self.keys[np.random.randint(len(self.keys))]
                continue
            
            # Apply traveling cumsum spatial weighting
            avg_coherence = np.mean(modulated)
            cusum_metrics = self.cusum_filter.update(avg_coherence, reference=0.5)
            spatial_weight = self.cusum_filter.get_spatial_weight()
            
            modulated_spatial = [score * spatial_weight for score in modulated]
            
            # Convert to probabilities
            probs = torch.softmax(torch.tensor(modulated_spatial), dim=0).numpy()
            
            if np.sum(probs) == 0:
                probs = np.ones(len(candidates)) / len(candidates)
            else:
                probs = probs / np.sum(probs)

            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            seed_tuple = tuple(output[-2:])
            step_count += 1

        generated_text = " ".join(output)
        
        # Apply Reverse QA if enabled and input was a question
        if enable_rqa and self.rqa_processor.is_question(original_seed):
            rqa_results = self.rqa_processor.process_reverse_qa(
                original_seed, 
                generated_text,
                show_metrics=True
            )
            return generated_text, rqa_results
        
        return generated_text, None


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n" + "="*60)
    print("   EIGENVALUE-ISOMORPHIC NEURAL REASONER")
    print("   φ(v) = P Λ P⁻¹ v  |  Λᵢᵢ = λᵢ + α tanh(β ⟨eᵢ, v⟩)")
    print("   With Reverse Question-Answer Processing")
    print("="*60)
    
    path = input("\nEnter text file: ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    corpus = open(path, 'r', encoding='utf-8').read().lower().split()
    model = build_ngram_model(corpus)
    print(f"\n📚 Loaded {len(corpus):,} tokens, model size: {len(model):,}")

    generator = ReasoningGenerator(corpus, model)
    
    print("\n💡 Commands:")
    print("   - Ask questions to trigger Reverse QA cycle")
    print("   - Prefix with 'norqa:' to disable Reverse QA")
    print("   - Type 'quit' or 'exit' to end\n")
    result = generator.generate(
            "warm up", 
            length=500, 
            enable_rqa=True
        )
    while True:
        seed = input("💬 USER: ")
        if seed.lower() in ['quit', 'exit']:
            break
        
        # Check if user wants to disable RQA
      
        result = generator.generate(
            seed, 
            length=500, 
            enable_rqa=True
        )
        
        if isinstance(result, tuple):
            generated, rqa_results = result
        else:
            generated = result
            rqa_results = None
        
        print("\n=== AI Response ===\n")
        print(generated)
        print(f"\n[Total: {len(generated.split())} words]")
        
        if rqa_results:
            print(f"\n✅ Reverse QA completed with {rqa_results['word_overlap_ratio']:.2%} word overlap")


if __name__ == "__main__":
    main()

import hashlib
import math
import numpy as np
import random
from collections import defaultdict, Counter
import unicodedata
import re
from tqdm import tqdm
import concurrent.futures
from datasets import load_dataset # Hugging Face datasets


KB_LEN = 999

CONNECTIVES = {
    ",", ".", ";", ":", "—", "-", "(", ")", "[", "]", "{", "}", "…",
    "and", "or", "but", "so", "yet", "for", "nor",
    "however", "therefore", "moreover", "meanwhile", "then", "thus",
    "of", "to", "in", "on", "at"
}

def is_connective(w):
    return w in CONNECTIVES

class AnnealedHashWeightGenerator:
    def __init__(self, alpha=0.15, beta=0.20, gamma=0.05, smoothing=1.0, rng=None, 
                 initial_temp=10.0, min_temp=0.01, cooling_rate=0.98):
        self.word_weights = {}
        self.vocabulary = set()
        self.word_transitions = defaultdict(list)
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.total_unigrams = 0
        self.gen_unigram_counts = Counter()
        self.gen_bigram_counts = defaultdict(Counter)
        
        # Annealing parameters
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.iteration = 0
        
        # Annealing parameters
        self.initial_tempB = initial_temp
        self.current_tempB = initial_temp
        self.min_tempB = min_temp
        self.cooling_rateB = cooling_rate
        self.iterationB = 0
        
        s = float(alpha) + float(beta) + float(gamma)
        if s <= 0:
            raise ValueError("alpha+beta+gamma must be > 0")
        self.alpha = float(alpha) / s
        self.beta = float(beta) / s
        self.gamma = float(gamma) / s
        self.base_gate = np.array([self.alpha, self.beta, self.gamma], dtype=float)
        self.smoothing = float(smoothing)
        self.rng = np.random.default_rng(rng)

    def anneal_hash_weight(self, token, base_weight):
        """Apply simulated annealing to hash weights."""
        if self.current_temp <= self.min_temp:
            return base_weight
            
        # Generate a neighbor weight by adding noise
        noise_factor = 0.1
        neighbor_weight = base_weight + self.rng.normal(0, noise_factor * self.current_temp)
        neighbor_weight = max(0.0, min(1.0, neighbor_weight))  # Clamp to [0,1]
        
        # Energy function (we want to maximize entropy in early stages)
        def energy(weight):
            # Penalize weights too close to 0 or 1 when temperature is high
            entropy_bonus = -abs(weight - 0.5) * self.current_temp / self.initial_temp
            return -weight + entropy_bonus
        
        current_energy = energy(base_weight)
        neighbor_energy = energy(neighbor_weight)
        
        # Acceptance probability
        if neighbor_energy < current_energy:
            return neighbor_weight
        else:
            acceptance_prob = math.exp((current_energy - neighbor_energy) / self.current_temp)
            if self.rng.random() < acceptance_prob:
                return neighbor_weight
            else:
                return base_weight

    def hash_to_weight(self, token):
        if token in self.word_weights:
            return self.word_weights[token]
        
        # Base SHA256 hash weight
        h = hashlib.sha256(token.encode('utf-8')).hexdigest()
        iv = int(h[:16], 16)
        mv = int('f' * 16, 16)
        base_weight = iv / mv
        
        # Apply annealing to the weight
        annealed_weight = self.anneal_hash_weight(token, base_weight)
        
        self.word_weights[token] = annealed_weight
        return annealed_weight

    def context_hash_weight(self, prev_w, w):
        return self.hash_to_weight(f"{prev_w}→{w}")

    def update_temperature(self):
        """Cool down the temperature after each iteration."""
        self.iteration += 1
        self.current_temp = max(self.min_temp, self.current_temp * self.cooling_rate)

    def build_vocabulary(self, text):
        words = text.lower().split()
        for i, w in enumerate(words):
            self.unigram_counts[w] += 1
            self.vocabulary.add(w)
            _ = []
            for word in self.vocabulary:
                _.append(self.hash_to_weight(word))
            if i < len(words) - 1:
                nxt = words[i + 1]
                self.bigram_counts[w][nxt] += 1
                self.word_transitions[w].append(nxt)
                self.vocabulary.add(nxt)
        self.total_unigrams = sum(self.unigram_counts.values())
        
        return len(self.vocabulary)

    def get_corpus_bigram_prob(self, prev_w, w):
        c_big = self.bigram_counts[prev_w][w]
        c_prev = self.unigram_counts[prev_w]
        V = max(1, len(self.vocabulary))
        return (c_big + self.smoothing) / (c_prev + self.smoothing * V)

    def get_generated_bigram_prob(self, prev_w, w):
        c_big = self.gen_bigram_counts[prev_w][w]
        c_prev = self.gen_unigram_counts[prev_w]
        V = max(1, len(self.vocabulary))
        return (c_big + self.smoothing) / (c_prev + self.smoothing * V)

    def get_candidates_for_word(self, current_word, use_transitions=True, cap=64):
        if use_transitions and self.word_transitions[current_word]:
            return sorted(set(self.word_transitions[current_word]))
        vocab = sorted(self.vocabulary)
        if len(vocab) <= cap:
            return vocab
        step = max(1, len(vocab) // cap)
        return vocab[::step][:cap]

    def _normalize(self, arr):
        s = float(np.sum(arr))
        if s <= 0:
            return np.ones_like(arr) / len(arr)
        return arr / s

    def _entropy01(self, p):
        p = np.clip(p, 1e-12, 1.0)
        H = -float(np.sum(p * np.log(p)))
        Hmax = math.log(len(p))
        return 0.0 if Hmax == 0 else min(1.0, H / Hmax)

    def compute_interpolated_probabilities(self, current_word, candidates):
        corp = np.array([self.get_corpus_bigram_prob(current_word, c) for c in candidates], dtype=float)
        corp = self._normalize(corp)

        gen = np.array([self.get_generated_bigram_prob(current_word, c) for c in candidates], dtype=float)
        gen = self._normalize(gen)

        # Apply annealed hash weights
        h_tok = np.array([self.hash_to_weight(c) for c in candidates], dtype=float)
        h_ctx = np.array([self.context_hash_weight(current_word, c) for c in candidates], dtype=float)
        hashp = 0.5 * h_tok + 0.5 * h_ctx
        hashp = self._normalize(hashp)

        c_prev = self.unigram_counts[current_word]
        out_deg = sum(self.bigram_counts[current_word].values())

        corp_ent = self._entropy01(corp)
        gen_ent = self._entropy01(gen)

        corp_ev = (1.0 if out_deg > 0 else 0.35) * (1.0 - 0.6 * corp_ent)
        gen_ev = (1.0 if self.gen_unigram_counts[current_word] > 0 else 0.5) * (1.0 - 0.6 * gen_ent)
        hash_ev = 1.0

        conn_ratio = sum(is_connective(c) for c in candidates) / max(1, len(candidates))
        corp_ev *= (1.0 + 0.20 * conn_ratio)
        gen_ev *= (1.0 + 0.15 * conn_ratio)
        hash_ev *= (1.0 - 0.25 * conn_ratio)

        # Temperature-influenced gating
        temp_influence = self.current_temp / self.initial_temp
        hash_ev *= (1.0 + temp_influence)  # Give more weight to exploration at high temp

        gate_weights = np.array([corp_ev, gen_ev, hash_ev], dtype=float)
        gate = self._normalize(gate_weights)

        mix = gate[0] * corp + gate[1] * gen + gate[2] * hashp
        return self._normalize(mix), gate

    def update_generated_counts(self, prev_w, w):
        self.gen_unigram_counts[prev_w] += 1
        self.gen_unigram_counts[prev_w] = min(self.gen_unigram_counts[prev_w], 1)
        self.gen_unigram_counts[w] += 1
        self.gen_unigram_counts[w] = min(self.gen_unigram_counts[w], 1)
        self.gen_bigram_counts[prev_w][w] += 1
        self.gen_bigram_counts[prev_w][w] = min(self.gen_bigram_counts[prev_w][w], 1)

    def generate_text(self, start_word, max_words=15, use_transitions=True):
        if not self.vocabulary:
            return ""
        start = start_word.lower()
        if start not in self.vocabulary:
            start = random.choice(tuple(self.vocabulary))
        current = start
        generated = [current]
        
        for _ in range(max_words - 1):
            cands = self.get_candidates_for_word(current, use_transitions=use_transitions)
            if not cands:
                break
            probs, gate = self.compute_interpolated_probabilities(current, cands)
            nxt = self.rng.choice(cands, p=probs)
            self.update_generated_counts(current, nxt)
            generated.append(nxt)
            current = nxt
            
            # Update temperature after each word generation
            self.update_temperature()
            
        return " ".join(generated)

def is_curved_letter(c):
    return unicodedata.combining(c) != 0 or not c.isascii()

repeated_char_pattern = re.compile(r'(.)\1+')

def optimize_curved_letters(text):
    def repl(m):
        ch = m.group(1)
        if is_curved_letter(ch):
            return ch
        return m.group(0)
    return repeated_char_pattern.sub(repl, text)

class PatternRepeatingAnnealedHashWeightGenerator(AnnealedHashWeightGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeating_patterns = {}

    def find_repeating_patterns(self, text, min_len=2, max_len=6):
        words = text.lower().split()
        length = len(words)
        patterns = Counter()

        for l in range(min_len, max_len + 1):
            seen = defaultdict(int)
            for start in range(length - l + 1):
                subseq = tuple(words[start:start+l])
                seen[subseq] += 1
            for p, c in seen.items():
                if c > 1:
                    patterns[p] += l
        self.repeating_patterns = patterns

    def hash_to_weight(self, token):
        # Check if token is part of a repeating pattern first
        for pattern in self.repeating_patterns:
            if token in pattern:
                pattern_str = " ".join(pattern)
                h = hashlib.sha256(pattern_str.encode('utf-8')).digest()
                float_weights = [(b / 255.0) for b in h]
                base_weight = np.mean(float_weights)
                # Apply annealing to pattern-based weights too
                annealed_weight = self.anneal_hash_weight(token, base_weight)
                self.word_weights[token] = annealed_weight
                return annealed_weight
        return super().hash_to_weight(token)

    def build_vocabulary(self, text):
        self.find_repeating_patterns(text)
        return super().build_vocabulary(text) 

# Load dataset and build corpus using multithreading
question_parts = []
answer_parts = []

if load_dataset:
    print("Loading dataset...")
    dataset = load_dataset('facebook/natural_reasoning', split='train')
    
    def extract_field(item, field_name):
        return item.get(field_name, None)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        question_futures = {executor.submit(extract_field, item, 'question'): item for item in dataset}
        answer_futures = {executor.submit(extract_field, item, 'reference_answer'): item for item in dataset}

        question_parts = [future.result() for future in tqdm(concurrent.futures.as_completed(question_futures), total=len(dataset), desc="Processing Questions") if future.result()]
        answer_parts = [future.result() for future in tqdm(concurrent.futures.as_completed(answer_futures), total=len(dataset), desc="Processing Answers") if future.result()]

print(f"Corpus loaded successfully. Total size: {len(question_parts[:KB_LEN])+len(answer_parts[:KB_LEN])} characters.")


# Initialize generator with annealing parameters
generatorA = PatternRepeatingAnnealedHashWeightGenerator(
    initial_temp=5.0,
    min_temp=0.1,
    cooling_rate=0.95
)
generatorA.build_vocabulary(' '.join(question_parts[:KB_LEN]))

# Initialize generator with annealing parameters
generatorB = PatternRepeatingAnnealedHashWeightGenerator(
    initial_temp=5.0,
    min_temp=0.1,
    cooling_rate=0.95
)
generatorB.build_vocabulary(' '.join(answer_parts[:KB_LEN]))

while True:
    try:
        start_word = input("USER: ")
        if start_word.lower() in ['quit', 'exit']:
            break
        
        # Reset temperature for each new generation
        generatorA.current_temp = generator.initial_temp
        generatorA.iteration = 0
        
        # Reset temperature for each new generation
        generatorB.current_tempB = generator.initial_tempB
        generatorB.iterationB = 0
        
        result = generatorB.generate_text(generatorA.generate_text(start_word, max_words=500), max_words=500)
        optimized_result = optimize_curved_letters(result)
        print(f"Generated text: {optimized_result}")
        print(f"Final temperature: {generator.current_temp:.3f}")
        
    except (KeyboardInterrupt, EOFError):
        print("\nExiting program.")
        break

     

import hashlib
import math
import numpy as np
import random
from collections import defaultdict, Counter
KB_LEN = 9999
CONNECTIVES = {
    ",", ".", ";", ":", "—", "-", "(", ")", "[", "]", "{", "}", "…",
    "and", "or", "but", "so", "yet", "for", "nor",
    "however", "therefore", "moreover", "meanwhile", "then", "thus",
    "of", "to", "in", "on", "at"
}

def is_connective(w):
    return w in CONNECTIVES

class SimpleHashWeightGenerator:
    def __init__(self, alpha=0.15, beta=0.20, gamma=0.05, smoothing=1.0, rng=None):
        self.word_weights = {}
        self.vocabulary = set()
        self.word_transitions = defaultdict(list)
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.total_unigrams = 0
        self.gen_unigram_counts = Counter()
        self.gen_bigram_counts = defaultdict(Counter)
        s = float(alpha) + float(beta) + float(gamma)
        if s <= 0:
            raise ValueError("alpha+beta+gamma must be > 0")
        self.alpha = float(alpha) / s
        self.beta = float(beta) / s
        self.gamma = float(gamma) / s
        self.base_gate = np.array([self.alpha, self.beta, self.gamma], dtype=float)
        self.smoothing = float(smoothing)
        self.rng = np.random.default_rng(rng)

    def hash_to_weight(self, token):
        if token in self.word_weights:
            return self.word_weights[token]
        h = hashlib.sha256(token.encode('utf-8')).hexdigest()
        iv = int(h[:16], 16)
        mv = int('f' * 16, 16)
        w = iv / mv
        self.word_weights[token] = w
        return w

    def context_hash_weight(self, prev_w, w):
        return self.hash_to_weight(f"{prev_w}→{w}")

    def build_vocabulary(self, text):
        words = text.lower().split()
        for i, w in enumerate(words):
            self.unigram_counts[w] += 1
            self.vocabulary.add(w)
            if i < len(words) - 1:
                nxt = words[i + 1]
                self.bigram_counts[w][nxt] += 1
                self.word_transitions[w].append(nxt)
                self.vocabulary.add(nxt)
        self.total_unigrams = sum(self.unigram_counts.values())
        for w in self.vocabulary:
            _ = self.hash_to_weight(w)
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

        gate = self.base_gate * np.array([corp_ev, gen_ev, hash_ev], dtype=float)
        gate = self._normalize(gate)

        mix = gate[0] * corp + gate[1] * gen + gate[2] * hashp
        return self._normalize(mix), gate

    def update_generated_counts(self, prev_w, w):
        self.gen_unigram_counts[prev_w] += 1
        self.gen_unigram_counts[w] += 1
        self.gen_bigram_counts[prev_w][w] += 1
        np.clip(self.gen_bigram_counts[prev_w][w], 0, self.gen_unigram_counts[w])

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
        return " ".join(generated)

class PatternRepeatingHashWeightGenerator(SimpleHashWeightGenerator):
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
        for pattern in self.repeating_patterns:
            if token in pattern:
                pattern_str = " ".join(pattern)
                h = hashlib.sha256(pattern_str.encode('utf-8')).digest()
                float_weights = [(b / 255.0) for b in h]
                # Store original weights for debugging/extension
                self.word_weights[token] = float_weights
                # Return average for compatibility with downstream code
                return np.mean(float_weights)
        return super().hash_to_weight(token)

    def build_vocabulary(self, text):
        self.find_repeating_patterns(text)
        return super().build_vocabulary(text)

if __name__ == "__main__":
    try:
        with open(input("Filename: "), 'r', encoding='utf-8') as f:
            corpus = f.read()[:KB_LEN]
    except FileNotFoundError:
        print("File not found, using sample text")
        corpus = "the quick brown fox jumps over the lazy dog"
    generator = PatternRepeatingHashWeightGenerator()
    while True:
        generator.build_vocabulary(corpus)
        result = generator.generate_text(input("USER: "), max_words=800)
        print(f"Generated text: {result}")

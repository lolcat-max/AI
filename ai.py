import numpy as np
import torch
from collections import Counter, defaultdict, deque
import os
from typing import List, Dict, Tuple, Set, Iterable

# ----------------------------------------------------------------
# Generic Grouping and Combination Utilities
# ----------------------------------------------------------------
def group_by(fn, items):
    groups = defaultdict(list)
    for item in items: groups[fn(item)].append(item)
    return groups

def unique(seq):
    seen, out = set(), []
    for s in seq:
        if s not in seen: seen.add(s); out.append(s)
    return out

def combine_pairs(pairs, can_combine_fn, combine_fn):
    used = set()
    new_terms = []
    for a, b in pairs:
        if can_combine_fn(a, b):
            new_terms.append(combine_fn(a, b))
            used.update([a, b])
    return new_terms, used

# ----------------------------------------------------------------
# Quine-McCluskey Isomorphic Minimize (Generic)
# ----------------------------------------------------------------
def minimize_boolean(n_bits, minterms, dontcares, bits, hamming_ones, can_combine, combine, covers):
    if not (minterms or dontcares): return []
    base_terms = [bits(m, n_bits) for m in sorted(set(minterms + dontcares))]
    terms = unique(base_terms)
    while True:
        grouped = group_by(lambda t: hamming_ones(t.replace('-', '0')), terms)
        keys = sorted(grouped.keys())
        pairs = [(a, b) for k in keys for a in grouped[k] for b in grouped.get(k + 1, [])]
        next_terms, used = combine_pairs(pairs, can_combine, combine)
        primes = [t for t in terms if t not in used]
        if not next_terms: return unique(primes)
        terms = unique(next_terms + primes)

def build_chart(primes, minterms, covers):
    return {p: set(m for m in minterms if covers(p, m)) for p in primes if any(covers(p, m) for m in minterms)}

def essential_primes(chart, mset):
    essentials = set(); remaining = set(mset)
    while True:
        uni = defaultdict(list)
        for p, ms in chart.items():
            for m in ms:
                if m in remaining: uni[m].append(p)
        added = False
        for m, plist in uni.items():
            if len(plist) == 1:
                essentials.add(plist[0]); remaining -= chart[plist[0]]; added = True
        if not added: break
    return essentials, remaining

def minimize_sop(n_bits, minterms, dontcares):
    bits = lambda n, w: format(n, '0{}b'.format(w))
    hamming_ones = lambda s: sum(c == '1' for c in s)
    can_combine = lambda a, b: sum((x != y) and (x != '-' and y != '-') for x, y in zip(a, b)) == 1
    combine = lambda a, b: ''.join([x if x == y else '-' if x != '-' and y != '-' else '-' for x, y in zip(a, b)])
    covers = lambda imp, m: all(x == '-' or x == y for x, y in zip(imp, m))
    primes = minimize_boolean(n_bits, minterms, dontcares, bits, hamming_ones, can_combine, combine, covers)
    chart = build_chart(primes, [bits(m, n_bits) for m in minterms], covers)
    essentials, remaining = essential_primes(chart, set(bits(m, n_bits) for m in minterms))
    chosen = essentials
    return [p for p in primes if p in chosen]

def implicants_to_expr(implicants, varnames):
    def cube_to_term(cube):
        return ' & '.join(name if bit == '1' else f'~{name}' for bit, name in zip(cube, varnames) if bit != '-')
    return ' | '.join(cube_to_term(c) for c in implicants) if implicants else '0'

def make_sop_evaluator(implicants, varorder):
    compiled = []
    for cube in implicants:
        req = [(i, b == '1') for i, b in enumerate(cube) if b != '-']
        compiled.append(req)
    return lambda bits: int(any(all((bits[idx] == 1) == need_one for idx, need_one in req) for req in compiled))

# ----------------------------------------------------------------
# Contextual Tracker/Features
# ----------------------------------------------------------------
class ContextTracker:
    def __init__(self, window=8):
        self.window = window
        self.hist = deque(maxlen=window)
        self.bigram = Counter()
        self.trigram = Counter()
        self.pos_map = defaultdict(Counter)
        self.momentum = 0.0
        self.coherence = 1.0

    def update(self, token, position):
        self.hist.append(token)
        if len(self.hist) >= 2: self.bigram[(self.hist[-2], self.hist[-1])] += 1
        if len(self.hist) >= 3: self.trigram[(self.hist[-3], self.hist[-2], self.hist[-1])] += 1
        self.pos_map[token][position % 10] += 1

    def get_freq(self, ngram, *args): return getattr(self, ngram).get(tuple(args), 0)
    def has_recent(self, tok, k=4): return tok in list(self.hist)[-k:]
    def positional_bias(self, tok, pos): return self.pos_map[tok][pos % 10]
    def update_momentum(self, vec_prev, vec_curr):
        if np.linalg.norm(vec_prev) > 1e-9 and np.linalg.norm(vec_curr) > 1e-9:
            c = np.dot(vec_prev, vec_curr) / (np.linalg.norm(vec_prev) * np.linalg.norm(vec_curr))
            self.momentum = 0.7*self.momentum + 0.3*c
    def update_coherence(self, score): self.coherence = 0.8*self.coherence + 0.2*score

# ----------------------------------------------------------------
# Surjection Generator/Word Features
# ----------------------------------------------------------------
class WordFeatures:
    def __init__(self, tokens):
        self.freq = Counter(tokens); self.total = max(1, len(tokens)); self.cache = {}
    def vec(self, w):
        if w in self.cache: return self.cache[w]
        L, f = len(w), self.freq[w]
        v = np.array([L/10, sum(c.isalpha() for c in w)/(L+1), sum(c in "aeiou" for c in w)/(L+1),
                      np.log(f+1)/np.log(self.total+1), 1/(f+1)], float)
        norm = np.linalg.norm(v)
        self.cache[w] = v / norm if norm > 1e-9 else v
        return self.cache[w]

class SurjectionGenerator:
    def __init__(self, tokens, model):
        self.tokens, self.model, self.keys = tokens, model, list(model.keys())
        self.feat = WordFeatures(tokens)
        self.context = ContextTracker()
        self.anchor_hits = None
        self.anchors = self._anchors()
        self.qmc_logs = []
        self.sim_thresh, self.align_thresh = 0.45, 0.05
        self.pmin, self.bigram_thresh, self.momentum_thresh = 1e-12, 2, 0.3
        self.alt_period = 14

    def _anchors(self, k=8):
        freq = Counter(self.tokens); top = [w for w, _ in freq.most_common(2*k)]
        feats, chosen = [], []
        for w in top:
            v = self.feat.vec(w); n = np.linalg.norm(v)
            v = v / n if n > 1e-9 else v
            if not feats: feats.append(v); chosen.append(w)
            elif min(np.linalg.norm(v-u) for u in feats) > 0.35: feats.append(v); chosen.append(w)
            if len(feats) >= k: break
        while len(feats) < k:
            w = top[np.random.randint(len(top))]
            v = self.feat.vec(w); n = np.linalg.norm(v)
            feats.append(v / n if n > 1e-9 else v); chosen.append(w)
        self.anchor_hits = np.zeros(len(feats), int)
        return np.stack(feats)

    def _nearest_anchor_idx(self, vec):
        v = vec/(np.linalg.norm(vec)+1e-9)
        return int(np.argmax(self.anchors @ v))

    def similarity(self, a, b):
        va, vb = self.feat.vec(a), self.feat.vec(b)
        score = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        return score + np.sin(2 * np.pi * score) / 4

    def autofunctor_scalar(self, token_current: str, token_context: Tuple[str, str], step: int) -> float:
        """
        Unified automorphic scalar function combining:
        - Cosine similarity (conceptual resonance)
        - Coherence score
        - Momentum alignment
        - Semantic anchor alignment
        - Temporal periodicity
        - Positional bias
        Returns single normalized scalar in [0,1]
        """
        v_c = self.feat.vec(token_current)
        v_ctx1 = self.feat.vec(token_context[0])
        v_ctx2 = self.feat.vec(token_context[1])
        v_ctx_mean = (v_ctx1 + v_ctx2) / 2.0

        # Cosine similarity conceptual resonance
        cos_sim = np.dot(v_ctx_mean, v_c) / ((np.linalg.norm(v_ctx_mean) * np.linalg.norm(v_c)) + 1e-9)

        # Momentum scalar coherence
        momentum_abs = abs(self.context.momentum)

        # Coherence running score
        coherence = self.context.coherence

        # Semantic anchor alignment normalized
        aidx_ctx = self._nearest_anchor_idx(v_ctx_mean)
        aidx_c = self._nearest_anchor_idx(v_c)
        anchor_align = 1.0 - abs(aidx_c - aidx_ctx) / max(1, len(self.anchors))

        # Temporal modulation (periodic)
        temporal_factor = np.sin((step % self.alt_period) / self.alt_period * np.pi)

        # Positional bias (binary scaled)
        pos_bias = float(self.context.positional_bias(token_current, step) > 0)

        # Aggregate normalized scalars into one fused automorphic scalar
        scalar = (0.35 * cos_sim +
                  0.25 * coherence +
                  0.15 * momentum_abs +
                  0.15 * anchor_align +
                  0.05 * temporal_factor +
                  0.05 * pos_bias)

        # Apply automorph transform
        scalar = scalar + np.sin(2 * np.pi * scalar) / 4
        # Clamp to [0,1]
        scalar = np.clip(scalar, 0.0, 1.0)
        return scalar

    def bool_features(self, c, sim_norm, base_p, q_lin, step, a_lin, p_final, cands):
        idx = cands.index(c)
        
        # Use unified autofunctor scalar
        ctx_tokens = (self.context.hist[-2], self.context.hist[-1]) if len(self.context.hist) >= 2 else (c, c)
        scalar = self.autofunctor_scalar(c, ctx_tokens, step)
        
        # Derive boolean features from scalar and context
        X0 = int(scalar >= 0.15)  # Thought signature: conceptual resonance
        X1 = int(scalar >= 0.1)   # Thought signature: intent alignment
        X2 = int((step + 1) % self.alt_period == 0)
        X3 = int(self._nearest_anchor_idx(self.feat.vec(c)) == np.argmin(self.anchor_hits))
        X4 = int(p_final[idx] >= self.pmin)
        X5 = 1
        X6 = int(self.context.get_freq('bigram', self.context.hist[-1], c) >= self.bigram_thresh if len(self.context.hist) >= 1 else 0)
        X7 = int(self.context.get_freq('trigram', self.context.hist[-2], self.context.hist[-1], c) > 0 if len(self.context.hist) >= 2 else 0)
        X8 = int(not self.context.has_recent(c))
        X9 = int(self.context.momentum >= self.momentum_thresh)
        X10 = int(self.context.positional_bias(c, step) > 0)
        X11 = int(self.context.coherence >= 0.1)
        X12 = int(step < 20)
        
        return {
            'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5,
            'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'X10': X10, 'X11': X11, 'X12': X12
        }

    def generate(self, seed, length=80, log_qmc=False):
        words = seed.split()[:2]
        while len(words) < 2: words.append(self.tokens[len(words) % len(self.tokens)])
        seed = tuple(words)
        if seed not in self.model: seed = self.keys[np.random.randint(len(self.keys))]
        out = list(seed)
        self.context = ContextTracker()
        for i, w in enumerate(out): self.context.update(w, i)
        
        for step in range(length):
            cands = list(self.model.get(seed, []))
            if not cands:
                seed = self.keys[np.random.randint(len(self.keys))]
                continue
            
            sim_scores = [self.similarity(out[-2], c) for c in cands]
            norm = max(sim_scores) + 1e-9
            sim_norm = [s / norm for s in sim_scores]
            base = torch.softmax(torch.tensor(sim_norm), dim=0).numpy()
            
            a_lin = 0
            q_lin = np.ones(len(cands)) / len(cands)
            p_final = np.clip(base, 1e-12, 1.0)
            p_final = p_final / p_final.sum()
            
            if log_qmc:
                for ci, c in enumerate(cands):
                    X = self.bool_features(c, sim_norm, base, q_lin, step, a_lin, p_final, cands)
                    accept = int((X['X0'] and X['X4']) and (X['X1'] or (X['X2'] and X['X3'])) and (X['X8'] and (X['X9'] or X['X6'])))
                    self.qmc_logs.append({'Y': c, 'X': X})
            
            next_word = np.random.choice(cands, p=p_final)
            
            v_next = self.feat.vec(next_word)
            aidx_chosen = self._nearest_anchor_idx(v_next)
            self.anchor_hits[aidx_chosen] += step
            self.context.update(next_word, step + len(out))
            
            if len(out) >= 2:
                self.context.update_momentum(self.feat.vec(out[-1]), v_next)
            if sim_scores:
                self.context.update_coherence(sim_norm[cands.index(next_word)])
            
            out.append(next_word)
            seed = tuple(out[-2:])
        
        return " ".join(out)

# ----------------------------------------------------------------
# Main Control/Runner
# ----------------------------------------------------------------
def build_ngram(tokens, n=2):
    m = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i+n])
        m[key].append(tokens[i+n])
    return m

def main():
    path = input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("file missing")
        return
    
    toks = open(path, encoding="utf-8").read().lower().split()
    model = build_ngram(toks, 2)
    g = SurjectionGenerator(toks, model)
    
    while True:
        s = input("\nseed (exit to quit): ")
        if s == "exit":
            break
        
        g.qmc_logs.clear()
        print("[Phase A] Collecting QMC logs...")
        _ = g.generate(s, length=400, log_qmc=True)
        
        varorder = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']
        on = [int(''.join(str(row['X'][v]) for v in varorder), 2) for row in g.qmc_logs if row['Y'] == 1]
        
        implicants = minimize_sop(len(varorder), on, [])
        expr = implicants_to_expr(implicants, varorder)
        
        print("\nQMC Result Implicants:", implicants)
        print("QMC Minimized SOP:", expr)
        
        gate = make_sop_evaluator(implicants, varorder)
        print("\n" + g.generate(s, length=120, log_qmc=False))

if __name__ == "__main__":
    main()

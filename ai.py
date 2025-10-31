import numpy as np
import torch
from collections import Counter, defaultdict, deque
import os
from typing import List, Tuple, Dict, Iterable, Set

# ================================================================
# Quine–McCluskey implementation (with Petrick's method)
# ================================================================
def _bits(n: int, width: int) -> str:
    return format(n, '0{}b'.format(width))

def _can_combine(a: str, b: str) -> bool:
    diff = 0
    for x, y in zip(a, b):
        if x != y:
            if x != '-' and y != '-':
                diff += 1
            else:
                return False
    return diff == 1

def _combine(a: str, b: str) -> str:
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x != '-' and y != '-':
            out.append('-')
        else:
            out.append('-')
    return ''.join(out)

def _hamming_ones(s: str) -> int:
    return sum(c == '1' for c in s)

def _covers(imp: str, m: str) -> bool:
    return all(x == '-' or x == y for x, y in zip(imp, m))

def _group_by_ones(terms: Iterable[str]) -> Dict[int, List[str]]:
    groups = defaultdict(list)
    for t in terms:
        groups[_hamming_ones(t)].append(t)
    return groups

def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _iterate_combine(terms: List[str]) -> List[str]:
    current = _unique(terms)
    while True:
        buckets = {}
        for t in current:
            k = _hamming_ones(t.replace('-', '0'))
            buckets.setdefault(k, []).append(t)

        used = set()
        next_terms = []
        keys = sorted(buckets.keys())
        for k in keys:
            a_bucket = buckets.get(k, [])
            b_bucket = buckets.get(k + 1, [])
            for a in a_bucket:
                for b in b_bucket:
                    if _can_combine(a, b):
                        c = _combine(a, b)
                        next_terms.append(c)
                        used.add(a); used.add(b)
        primes = [t for t in current if t not in used]
        if not next_terms:
            return _unique(primes)
        current = _unique(next_terms + primes)

def _build_pi_chart(primes: List[str], minterms: List[str]) -> Dict[str, Set[str]]:
    cov = {}
    for p in primes:
        covered = {m for m in minterms if _covers(p, m)}
        if covered:
            cov[p] = covered
    return cov

def _essential_primes(pi_chart: Dict[str, Set[str]], mset: Set[str]) -> Tuple[Set[str], Set[str]]:
    essentials = set()
    remaining = set(mset)
    while True:
        unique_map = defaultdict(list)
        for p, cols in pi_chart.items():
            for m in cols:
                if m in remaining:
                    unique_map[m].append(p)
        added = False
        for m, plist in unique_map.items():
            if len(plist) == 1:
                ep = plist[0]
                if ep not in essentials:
                    essentials.add(ep)
                    remaining -= pi_chart.get(ep, set())
                    added = True
        if not added:
            break
    return essentials, remaining

def _petrick(pi_chart: Dict[str, Set[str]], remaining: Set[str]) -> Set[str]:
    sums: List[Set[frozenset]] = []
    for m in remaining:
        choices = {frozenset([p]) for p, cols in pi_chart.items() if m in cols}
        sums.append(choices)
    if not sums:
        return set()
    prod: Set[frozenset] = {frozenset()}
    for s in sums:
        new_prod: Set[frozenset] = set()
        for term in prod:
            for choice in sums:
                new_term = frozenset(set(term) ^ set(choice))
                new_prod.add(new_term)
        minimal = set(new_prod)
        for a in list(new_prod):
            for b in list(new_prod):
                if a == b and a.issuperset(b):
                    minimal.discard(a)
        prod = minimal
    min_size = min(len(t) for t in prod)
    minimal_sets = [t for t in prod if len(t) == min_size]
    def literal_cost(ps: Iterable[str]) -> int:
        return sum(len(p.replace('-', '')) for p in ps)
    best = min(minimal_sets, key=lambda s: literal_cost([p for p in s]))
    return set(best)

def minimize_sop(n_bits: int, minterms: List[int], dontcares: List[int]) -> List[str]:
    on = sorted(set(minterms))
    dc = sorted(set(dontcares))
    if not (on or dc):
        return []
    base_terms = [_bits(m, n_bits) for m in on + dc]
    primes = _iterate_combine(base_terms)
    chart = _build_pi_chart(primes, [_bits(m, n_bits) for m in on])
    essentials, remaining = _essential_primes(chart, set(_bits(m, n_bits) for m in on))
    cover_rest = _petrick(chart, remaining) if remaining else set()
    chosen = essentials | cover_rest
    return [p for p in primes if p in chosen]

def implicants_to_expr(implicants: List[str], varnames: List[str]) -> str:
    def cube_to_term(cube: str) -> str:
        lits = []
        for bit, name in zip(cube, varnames):
            if bit == '1':
                lits.append(name)
            elif bit == '0':
                lits.append(f'~{name}')
        return ' & '.join(lits) if lits else '1'
    return ' | '.join(cube_to_term(c) for c in implicants) if implicants else '0'

def make_sop_evaluator(implicants: List[str], varorder: List[str]):
    compiled = []
    for cube in implicants:
        req = []
        for i, b in enumerate(cube):
            if b == '-':
                continue
            req.append((i, b == '1'))
        compiled.append(req)
    def eval_row(bits: List[int]) -> int:
        for req in compiled:
            ok = True
            for idx, need_one in req:
                if (bits[idx] == 1) != need_one:
                    ok = False; break
            if ok:
                return 1
        return 0
    return eval_row

# ================================================================
# Contextual State Tracker
# ================================================================
class ContextTracker:
    """Tracks generation context for contextual boolean features."""
    def __init__(self, window_size=8):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)  # Recent tokens
        self.bigram_counts = Counter()  # Bigram frequency in this generation
        self.trigram_counts = Counter()  # Trigram frequency
        self.pos_token_map = defaultdict(Counter)  # token→position frequency
        self.momentum = 0.0  # Directional momentum in feature space
        self.coherence_score = 1.0  # Running coherence metric
        
    def update(self, token: str, position: int):
        self.history.append(token)
        
        # Update n-gram counts
        if len(self.history) >= 2:
            bigram = (self.history[-2], self.history[-1])
            self.bigram_counts[bigram] += 1
        if len(self.history) >= 3:
            trigram = (self.history[-3], self.history[-2], self.history[-1])
            self.trigram_counts[trigram] += 1
        
        # Update positional info
        self.pos_token_map[token][position % 10] += 1
    
    def get_bigram_frequency(self, prev: str, candidate: str) -> int:
        return self.bigram_counts.get((prev, candidate), 0)
    
    def get_trigram_frequency(self, prev2: str, prev1: str, candidate: str) -> int:
        return self.trigram_counts.get((prev2, prev1, candidate), 0)
    
    def has_repetition(self, candidate: str, lookback=4) -> bool:
        """Check if candidate appears in recent history."""
        recent = list(self.history)[-lookback:]
        return candidate in recent
    
    def get_positional_bias(self, candidate: str, position: int) -> int:
        """Check if candidate has appeared at this position modulo 10."""
        pos_mod = position % 10
        return self.pos_token_map[candidate][pos_mod]
    
    def update_momentum(self, vec_prev: np.ndarray, vec_curr: np.ndarray):
        """Update directional momentum in feature space."""
        if np.linalg.norm(vec_prev) < 1e-9 or np.linalg.norm(vec_curr) < 1e-9:
            return
        cos_sim = np.dot(vec_prev, vec_curr) / (np.linalg.norm(vec_prev) * np.linalg.norm(vec_curr))
        self.momentum = 0.7 * self.momentum + 0.3 * cos_sim
    
    def update_coherence(self, score: float):
        """Update running coherence metric."""
        self.coherence_score = 0.8 * self.coherence_score + 0.2 * score

# ================================================================
# Automorphic Surjection Generator (contextual QMC)
# ================================================================
class SurjectionField:
    def __init__(self):
        self.map = {}
    def register(self, a,b,v): self.map[(a,b)] = float(v)
    def lookup(self,a,b): return self.map.get((a,b),None)
    def automorph(self):
        new_map = {}
        for (a,b), v in self.map.items():
            new_map[(b,a)] = v
            new_map[(a,b)] = v
        self.map = new_map
        return self

class SurjectionOps:
    def __init__(self,field=None): 
        self.field=field or SurjectionField()
    def surject(self,u,v,a=None,b=None):
        u=np.asarray(u,float); v=np.asarray(v,float)
        n=min(len(u),len(v))
        if n==0: return 0.5
        dot=np.dot(u[:n],v[:n]); nv2=np.dot(v[:n],v[:n])+1e-9
        corr=1.0
        if a and b:
            val=self.field.lookup(a,b)
            if val is not None: corr=0.7+0.6*np.tanh(val)
        result = float(np.clip(0.5*(np.tanh(corr*dot/nv2)+1),0,1))
        result = self.automorph_scalar(result)
        return result
    def automorph_scalar(self, x):
        return 1.0 - x if x > 0.5 else x

class WordFeatures:
    def __init__(self,tokens):
        self.freq=Counter(tokens); self.total=max(1,len(tokens))
        self.feature_cache = {}
    def vec(self,w):
        if w in self.feature_cache:
            return self.feature_cache[w]
        L=len(w); f=self.freq.get(w,1)
        vec = np.array([
            L/10, sum(c.isalpha() for c in w)/(L+1),
            sum(c in "aeiou" for c in w)/(L+1),
            np.log(f+1)/np.log(self.total+1),
            1/(f+1)
        ],float)
        vec = self.automorph_vector(vec)
        self.feature_cache[w] = vec
        return vec
    def automorph_vector(self, v):
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return v
        normalized = v / norm
        reflected = 2 * np.dot(normalized, normalized) * normalized - v
        return reflected

class SurjectionGenerator:
    def __init__(self,tokens,model):
        self.tokens=tokens
        self.model=model
        self.keys=list(model.keys())
        self.field=SurjectionField()
        self.ops=SurjectionOps(self.field)
        self.feat=WordFeatures(tokens)
        self._auto_pairs()
        self.generation_state = []
        self.context = ContextTracker(window_size=8)

        self._build_codomain_anchors(k=18)
        self.anchor_hits = np.zeros(len(self.anchors), dtype=int)

        self.alt_period = 14
        self.alpha_linear = 0.35
        self.beta_onto = 0.45

        # QMC logging state
        self.qmc_logs: List[Dict] = []
        
        # Contextual thresholds
        self.sim_thresh = 0.45
        self.align_thresh = 0.05
        self.pmin = 1e-12
        self.bigram_thresh = 2  # Threshold for common bigram
        self.momentum_thresh = 0.3  # Momentum alignment threshold

    def _auto_pairs(self):
        big=Counter(zip(self.tokens[:-1],self.tokens[1:]))
        if not big:return
        m=max(big.values())
        for (a,b),c in big.items():
            self.field.register(a,b,c/m)
        self.field.automorph()

    def _build_codomain_anchors(self, k=8):
        counts = Counter(self.tokens)
        top = [w for w,_ in counts.most_common(max(2*k, k+4))]
        feats = []
        chosen = []
        for w in top:
            v = self.feat.vec(w)
            n = np.linalg.norm(v) + 1e-9
            v = v / n
            if not feats:
                feats.append(v); chosen.append(w)
            else:
                dmin = min(np.linalg.norm(v - u) for u in feats)
                if dmin > 0.35:
                    feats.append(v); chosen.append(w)
            if len(feats) >= k:
                break
        while len(feats) < k and top:
            w = top[np.random.randint(len(top))]
            v = self.feat.vec(w); v = v / (np.linalg.norm(v)+1e-9)
            feats.append(v); chosen.append(w)
        self.anchors = np.stack(feats, axis=0)
        self.anchor_tokens = chosen

    def _candidate_feat_matrix(self, cands: List[str]) -> np.ndarray:
        V = [self.feat.vec(c) for c in cands]
        V = np.array(V, float)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        return V / norms

    def _nearest_anchor_idx(self, vec: np.ndarray) -> int:
        v = vec / (np.linalg.norm(vec)+1e-9)
        sims = self.anchors @ v
        return int(np.argmax(sims))

    def _anchor_alignment_dist(self, cands: List[str], anchor_idx: int) -> np.ndarray:
        A = self.anchors[anchor_idx]
        C = self._candidate_feat_matrix(cands)
        sims = C @ A
        sims = np.maximum(sims, 0.0)
        if sims.max() < 1e-12:
            sims = np.ones_like(sims)
        sims = sims / (sims.sum() + 1e-9)
        return sims

    def _onto_reweight(self, cands: List[str]) -> Tuple[np.ndarray,int]:
        min_hits = self.anchor_hits.min()
        under = np.where(self.anchor_hits == min_hits)[0]
        aidx = int(under[len(self.generation_state) % len(under)])
        q = self._anchor_alignment_dist(cands, aidx)
        return q, aidx

    def _linearize_toward_context(self, cands: List[str], context_words: Tuple[str,str]) -> Tuple[np.ndarray,int]:
        v1 = self.feat.vec(context_words[0])
        v2 = self.feat.vec(context_words[1])
        ctx = (v1 + v2) / 2.0
        aidx = self._nearest_anchor_idx(ctx)
        q = self._anchor_alignment_dist(cands, aidx)
        return q, aidx

    def surjection_similarity(self,a,b):
        va,vb=self.feat.vec(a),self.feat.vec(b)
        score = self.ops.surject(va,vb,a,b)
        score = self.automorph_similarity(score)
        return score
    
    def automorph_similarity(self, s):
        return s + np.sin(2 * np.pi * s) / 4
    
    def automorph_state(self):
        if len(self.generation_state) < 2:
            return
        self.generation_state[-2], self.generation_state[-1] = \
            self.generation_state[-1], self.generation_state[-2]

    # -------- CONTEXTUAL Booleanization for QMC --------
    def _bool_features_for_candidate(self, c: str, sim_norm: List[float], base_probs: np.ndarray,
                                     q_lin: np.ndarray, step: int, a_lin: int,
                                     p_final: np.ndarray, cands: List[str]) -> Dict[str, int]:
        idx = cands.index(c)
        s_norm = sim_norm[idx] if sim_norm else 0.0
        
        # X0: normalized similarity >= sim_thresh (base feature)
        X0 = 1 if s_norm >= self.sim_thresh else 0

        # X1: near top alignment in q_lin (base feature)
        top_idx = int(np.argmax(q_lin))
        X1 = 1 if (idx == top_idx or (q_lin[top_idx] - q_lin[idx]) <= self.align_thresh) else 0

        # X2: alternation step active (temporal context)
        X2 = 1 if ((step + 1) % self.alt_period == 0) else 0

        # X3: candidate aligns with least-covered anchor (onto coverage)
        min_hits = self.anchor_hits.min()
        under = np.where(self.anchor_hits == min_hits)[0]
        v_c = self.feat.vec(c)
        a_c = self._nearest_anchor_idx(v_c)
        X3 = 1 if a_c in under else 0

        # X4: final p >= pmin (probability threshold)
        X4 = 1 if p_final[idx] >= self.pmin else 0

        # X5: candidate exists in model[seed] (always true for enumerated)
        X5 = 1

        # ===== CONTEXTUAL FEATURES =====
        
        # X6: bigram frequency above threshold (n-gram context)
        if len(self.context.history) >= 1:
            prev = self.context.history[-1]
            bigram_freq = self.context.get_bigram_frequency(prev, c)
            X6 = 1 if bigram_freq >= self.bigram_thresh else 0
        else:
            X6 = 0

        # X7: trigram exists in generation history (stronger n-gram)
        if len(self.context.history) >= 2:
            prev2, prev1 = self.context.history[-2], self.context.history[-1]
            trigram_freq = self.context.get_trigram_frequency(prev2, prev1, c)
            X7 = 1 if trigram_freq > 0 else 0
        else:
            X7 = 0

        # X8: no recent repetition (diversity constraint)
        X8 = 0 if self.context.has_repetition(c, lookback=4) else 1

        # X9: momentum alignment (directional coherence)
        v_c = self.feat.vec(c)
        if len(self.context.history) >= 1:
            v_prev = self.feat.vec(self.context.history[-1])
            if np.linalg.norm(v_prev) > 1e-9 and np.linalg.norm(v_c) > 1e-9:
                cos_sim = np.dot(v_prev, v_c) / (np.linalg.norm(v_prev) * np.linalg.norm(v_c))
                X9 = 1 if cos_sim >= self.momentum_thresh else 0
            else:
                X9 = 0
        else:
            X9 = 0

        # X10: positional bias (candidate appeared at this position % 10 before)
        X10 = 1 if self.context.get_positional_bias(c, step) > 0 else 0

        # X11: high coherence state (running coherence above threshold)
        X11 = 1 if self.context.coherence_score >= 0.6 else 0

        # X12: early in generation (step < 20, different behavior early vs late)
        X12 = 1 if step < 20 else 0

        return {
            'X0': X0, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5,
            'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'X10': X10, 'X11': X11, 'X12': X12
        }

    def _log_qmc_row(self, X: Dict[str, int], accept: int):
        self.qmc_logs.append({'X': X, 'Y': accept})

    def generate(self,seed,length=80, enable_qmc_logging=False):
        words=seed.split()[:2]
        while len(words)<2:
            words.append(self.tokens[len(words)%len(self.tokens)])
        seed=tuple(words)
        if seed not in self.model: seed=self.keys[np.random.randint(len(self.keys))]
        out=list(seed)
        self.generation_state = list(seed)
        self.context = ContextTracker(window_size=8)  # Reset context
        
        # Initialize context with seed
        for i, w in enumerate(out):
            self.context.update(w, i)
        
        print(f"\n[Automorphic Surjection Generator] seed: {' '.join(seed)}")
        
        for step in range(length):
            cands=self.model.get(seed,[])
            if not cands:
                seed=self.keys[np.random.randint(len(self.keys))]; continue

            sim_scores=[self.surjection_similarity(out[-2],c) for c in cands]
            if not sim_scores: continue

            norm = (max(sim_scores) + 1e-9)
            sim_norm = [s / norm for s in sim_scores]
            base = torch.softmax(torch.tensor(sim_norm, dtype=torch.float),dim=0).numpy()

            q_lin, a_lin = self._linearize_toward_context(cands, (out[-2], out[-1]))
            p_lin = (1.0 - self.alpha_linear) * base + self.alpha_linear * q_lin

            if (step + 1) % self.alt_period == 0:
                q_onto, a_onto = self._onto_reweight(cands)
                p = (1.0 - self.beta_onto) * p_lin + self.beta_onto * q_onto
            else:
                p = p_lin

            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum()

            # QMC logging with contextual features
            if enable_qmc_logging:
                for ci, c in enumerate(cands):
                    X = self._bool_features_for_candidate(
                        c, sim_norm, base, q_lin, step, a_lin, p, cands
                    )
                    # Enhanced contextual policy: accept if high similarity AND good context
                    # (X0 and X4) and (X1 or (X2 and X3)) and (X8 and (X9 or X6))
                    accept = int(
                        (X['X0'] and X['X4']) and 
                        (X['X1'] or (X['X2'] and X['X3'])) and
                        (X['X8'] and (X['X9'] or X['X6']))
                    )
                    self._log_qmc_row(X, accept)

            next_word=np.random.choice(cands,p=p)

            # Update coverage
            v_next = self.feat.vec(next_word)
            a_chosen = self._nearest_anchor_idx(v_next)
            self.anchor_hits[a_chosen] += 1

            # Update context tracker
            self.context.update(next_word, step + len(out))
            if len(out) >= 2:
                v_prev = self.feat.vec(out[-1])
                self.context.update_momentum(v_prev, v_next)
            
            # Update coherence based on similarity score
            if sim_scores:
                coherence = sim_norm[cands.index(next_word)]
                self.context.update_coherence(coherence)

            self.generation_state.append(next_word)
            if (step + 1) % 5 == 0:
                self.automorph_state()

            out.append(next_word)
            seed=tuple(out[-2:])
        
        return " ".join(out)

# ================================================================
# QMC training + gated generation
# ================================================================
def learn_minimized_gate_from_logs(logs: List[Dict], varorder: List[str]) -> Tuple[List[str], str]:
    on = []
    dc = []
    for row in logs:
        X = row['X']; y = row['Y']
        bits = ''.join(str(X[v]) for v in varorder)
        mi = int(bits, 2)
        if y == 1:
            on.append(mi)
    n_bits = len(varorder)
    implicants = minimize_sop(n_bits, on, dc)
    expr = implicants_to_expr(implicants, varorder)
    return implicants, expr

def generate_with_implicants(gen: SurjectionGenerator, seed: str, length=80, gate=None, varorder=None):
    words=seed.split()[:2]
    while len(words)<2:
        words.append(gen.tokens[len(words)%len(gen.tokens)])
    seed=tuple(words)
    if seed not in gen.model: seed=gen.keys[np.random.randint(len(gen.keys))]
    out=list(seed)
    gen.generation_state = list(seed)
    gen.context = ContextTracker(window_size=8)  # Reset context
    
    # Initialize context with seed
    for i, w in enumerate(out):
        gen.context.update(w, i)
    
    print(f"\n[Automorphic Surjection Generator + QMC Gate] seed: {' '.join(seed)}")

    for step in range(length):
        cands=gen.model.get(seed,[])
        if not cands:
            seed=gen.keys[np.random.randint(len(gen.keys))]; continue

        sim_scores=[gen.surjection_similarity(out[-2],c) for c in cands]
        if not sim_scores: continue
        norm = (max(sim_scores) + 1e-9)
        sim_norm = [s / norm for s in sim_scores]
        base = torch.softmax(torch.tensor(sim_norm, dtype=torch.float),dim=0).numpy()

        q_lin, a_lin = gen._linearize_toward_context(cands, (out[-2], out[-1]))
        p_lin = (1.0 - gen.alpha_linear) * base + gen.alpha_linear * q_lin
        if (step + 1) % gen.alt_period == 0:
            q_onto, a_onto = gen._onto_reweight(cands)
            p = (1.0 - gen.beta_onto) * p_lin + gen.beta_onto * q_onto
        else:
            p = p_lin

        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()

        # Apply implicant gate with contextual features
        mask = np.ones(len(cands), dtype=int)
        if gate is not None and varorder is not None:
            for ci, c in enumerate(cands):
                X = gen._bool_features_for_candidate(
                    c, sim_norm, base, q_lin, step, a_lin, p, cands
                )
                bits = [int(X[v]) for v in varorder]
                mask[ci] = gate(bits)

        if mask.sum() == 0:
            mask[np.argmax(p)] = 1

        p_masked = p * mask
        p_masked = p_masked / p_masked.sum()

        next_word=np.random.choice(cands,p=p_masked)

        v_next = gen.feat.vec(next_word)
        a_chosen = gen._nearest_anchor_idx(v_next)
        gen.anchor_hits[a_chosen] += 1
        
        # Update context tracker
        gen.context.update(next_word, step + len(out))
        if len(out) >= 2:
            v_prev = gen.feat.vec(out[-1])
            gen.context.update_momentum(v_prev, v_next)
        
        coherence = sim_norm[cands.index(next_word)]
        gen.context.update_coherence(coherence)
        
        gen.generation_state.append(next_word)
        if (step + 1) % 5 == 0:
            gen.automorph_state()

        out.append(next_word)
        seed=tuple(out[-2:])
    return " ".join(out)

# ================================================================
# BUILD MODEL + RUN
# ================================================================
def build_ngram(tokens,n=2):
    m=defaultdict(list)
    for i in range(len(tokens)-n):
        m[tuple(tokens[i:i+n])].append(tokens[i+n])
    return m

def main():
    print("=== CONTEXTUAL AUTOMORPHIC SURJECTION TEXT GENERATOR ===")
    print("QMC with temporal, n-gram, and momentum-based features\n")
    
    path=input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("file missing"); return
    toks=open(path,encoding="utf-8").read().lower().split()
    model=build_ngram(toks,2)
    
    print(f"Loaded {len(toks)} tokens")
    print(f"Model size: {len(model)} n-grams")
    
    g=SurjectionGenerator(toks,model)

    
    # Phase C: interactive generation with contextual implicant gating
    while True:
        s=input("\nseed (exit to quit): ")
        if s=="exit":break
        # Phase A: collect contextual logs for QMC
        print("\n[Phase A] Collecting contextual boolean logs for QMC...")
        _ = g.generate(s, length=400, enable_qmc_logging=True)
        print(f"Collected {len(g.qmc_logs)} candidate rows with contextual features")

        # Phase B: learn minimized SOP with contextual variables
        varorder = ['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12']
        
        implicants, expr = learn_minimized_gate_from_logs(g.qmc_logs, varorder)
        print("\n[QMC Result] Implicants:")
        for imp in implicants:
            print("  -", imp)
        print(f"\n[QMC Result] Minimized SOP over {len(varorder)} contextual variables:\n  {expr}")

        gate = make_sop_evaluator(implicants, varorder)

        print("\n"+generate_with_implicants(g, s, length=620, gate=gate, varorder=varorder))

if __name__=="__main__":
    main()

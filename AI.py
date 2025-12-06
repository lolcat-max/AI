"""
fourd_muscle_complete.py

Complete 4D muscle with:
- Dict-compatible diagram processing
- Unified operation patterns (math + grammar)
- Trivector association prelimination
- Numeric Euclidean space
- All pipeline components integrated
"""
KB_len = 999
LEN = 999
import numpy as np
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
import random

# ------------------------
# Utilities
# ------------------------
def flatten4(t):
    """Flatten a 4D activation (C,X,Y,Z) to 1D vector"""
    return t.reshape(-1)

def unflatten4(v, shape):
    """Reshape flat vector v into 4D shape"""
    return v.reshape(shape)

def l2norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / max(n, eps)

# ------------------------
# TRIVECTOR ASSOCIATION PRELIMINATION
# ------------------------
class TrivectorAssociation:
    """
    Preliminary association filter using trivector (3-way) binding.
    Encodes complex relationships between token triples in VSA space.
    """
    
    def __init__(self, vsa_dim=1024):
        self.vsa_dim = vsa_dim
        self.trivector_cache = {}
        self.association_history = deque(maxlen=1000)
        self.binding_strength = {}
        
        # Trivector binding operators
        self.binding_ops = {
            'circular': self._circular_bind,
            'cascade': self._cascade_bind,
            'holographic': self._holographic_bind,
            'geometric': self._geometric_bind,
        }
        
    def _circular_bind(self, a, b, c):
        """Circular convolution binding: (A ⊛ B) ⊛ C"""
        ab = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
        abc = np.fft.ifft(np.fft.fft(ab) * np.fft.fft(c)).real
        return l2norm(abc)
    
    def _cascade_bind(self, a, b, c):
        """Cascade binding: A + (B ⊛ C)"""
        bc = np.fft.ifft(np.fft.fft(b) * np.fft.fft(c)).real
        abc = a + bc
        return l2norm(abc)
    
    def _holographic_bind(self, a, b, c):
        """Holographic reduced representation"""
        fa = np.fft.fft(a)
        fb = np.fft.fft(b)
        fc = np.fft.fft(c)
        result = fa * fb * fc
        abc = np.fft.ifft(result).real
        return l2norm(abc)
    
    def _geometric_bind(self, a, b, c):
        """Geometric algebra binding via rotations"""
        theta_b = np.sum(b) * 2 * np.pi
        theta_c = np.sum(c) * 2 * np.pi
        result = a.copy()
        result = result * np.cos(theta_b) + np.roll(result, 1) * np.sin(theta_b)
        result = result * np.cos(theta_c) + np.roll(result, 2) * np.sin(theta_c)
        return l2norm(result)
    
    def preliminate(self, token_triple, vsa4d, method='circular'):
        """Preliminary association scoring for token triple."""
        tok_a, tok_b, tok_c = token_triple
        cache_key = (tok_a, tok_b, tok_c, method)
        if cache_key in self.trivector_cache:
            return self.trivector_cache[cache_key]
        
        vec_a = vsa4d.vec_flat(tok_a)
        vec_b = vsa4d.vec_flat(tok_b)
        vec_c = vsa4d.vec_flat(tok_c)
        
        bind_func = self.binding_ops.get(method, self._circular_bind)
        trivector = bind_func(vec_a, vec_b, vec_c)
        
        strength_a = float(np.dot(trivector, vec_a))
        strength_b = float(np.dot(trivector, vec_b))
        strength_c = float(np.dot(trivector, vec_c))
        
        overall_strength = np.power(
            np.abs(strength_a * strength_b * strength_c), 
            1/3000
        ) * np.sign(strength_a * strength_b * strength_c)
        
        result = {
            'trivector': trivector,
            'strength': overall_strength,
            'components': (strength_a, strength_b, strength_c),
        }
        
        self.trivector_cache[cache_key] = result
        return result
    
    def extract_triples_from_context(self, ctx):
        """Extract all possible token triples from context."""
        triples = []
        for i in range(len(ctx) - 2):
            triples.append((ctx[i], ctx[i+1], ctx[i+2]))
        
        if len(ctx) >= 5:
            for i in range(len(ctx) - 4):
                triples.append((ctx[i], ctx[i+2], ctx[i+4]))
        
        return triples
    
    def preliminate_candidates(self, candidates, ctx, vsa4d, method='circular'):
        """Preliminary filter candidates based on trivector associations."""
        if not candidates or len(ctx) < 2:
            return {tok: 1.0 for tok in candidates}
        
        triples = self.extract_triples_from_context(ctx[-8:])
        if not triples:
            return {tok: 1.0 for tok in candidates}
        
        scores = {}
        for cand in candidates:
            cand_score = 0.0
            cand_vec = vsa4d.vec_flat(cand)
            
            for triple in triples:
                prelim = self.preliminate(triple, vsa4d, method=method)
                trivec = prelim['trivector']
                sim = float(np.dot(cand_vec, trivec))
                weighted_sim = sim * prelim['strength']
                cand_score += weighted_sim
            
            scores[cand] = cand_score / max(len(triples), 1)
        
        if scores:
            min_score = min(scores.values())
            max_score = max(scores.values())
            score_range = max_score - min_score
            
            if score_range > 1e-12:
                scores = {k: (v - min_score) / score_range for k, v in scores.items()}
            else:
                scores = {k: 1.0 for k in scores}
        
        exp_scores = {k: np.exp(v * 2.0) for k, v in scores.items()}
        Z = sum(exp_scores.values())
        if Z > 1e-12:
            scores = {k: v / Z for k, v in exp_scores.items()}
        
        return scores
    
    def apply_prelimination(self, probs_dict, ctx, vsa4d, alpha=0.3, method='circular'):
        """Apply trivector prelimination filter to probability distribution."""
        if not probs_dict or len(ctx) < 2:
            return probs_dict
        
        candidates = list(probs_dict.keys())
        prelim_scores = self.preliminate_candidates(candidates, ctx, vsa4d, method=method)
        
        blended = {}
        for tok in candidates:
            orig_p = probs_dict[tok]
            prelim_p = prelim_scores.get(tok, 0.5)
            blended[tok] = (1 - alpha) * orig_p + alpha * prelim_p
        
        Z = sum(blended.values())
        if Z > 1e-12:
            blended = {k: v / Z for k, v in blended.items()}
        else:
            return probs_dict
        
        self.association_history.append({
            'ctx': tuple(ctx[-3:]),
            'top_token': max(blended.items(), key=lambda x: x[1])[0],
            'method': method,
        })
        
        return blended

# ------------------------
# DIAGRAM PROCESSOR (Dict-compatible)
# ------------------------
class DiagramProcessor:
    """
    Applies the dual-path diagram logic to probability dictionaries.
    Path A: tokens → axis0_group → exp → axis1_group → merge
    Path B: tokens → axis1_group → axis0_group → merge
    """
    def __init__(self, n_axis0_groups=8, n_axis1_groups=8):
        self.n_axis0_groups = n_axis0_groups
        self.n_axis1_groups = n_axis1_groups
        self.token_axis0_map = {}
        self.token_axis1_map = {}
        
    def assign_axes(self, token):
        """Assign a token to axis0 and axis1 groups deterministically."""
        if token not in self.token_axis0_map:
            h = hash(token)
            self.token_axis0_map[token] = (h & 0xFF) % self.n_axis0_groups
            self.token_axis1_map[token] = ((h >> 8) & 0xFF) % self.n_axis1_groups
        return self.token_axis0_map[token], self.token_axis1_map[token]
    
    def apply_diagram_dict(self, probs_dict, blend_weight=0.3):
        """Apply diagram dual-path processing to probability dictionary."""
        if not probs_dict:
            return probs_dict
        
        tokens = list(probs_dict.keys())
        
        axis0_groups = defaultdict(list)
        axis1_groups = defaultdict(list)
        
        for tok in tokens:
            a0, a1 = self.assign_axes(tok)
            p = probs_dict[tok]
            axis0_groups[a0].append((tok, p))
            axis1_groups[a1].append((tok, p))
        
        # PATH A: Axis0 → exp → Axis1
        pathA_scores = {}
        for a0_id, tok_probs in axis0_groups.items():
            group_sum = sum(p for _, p in tok_probs)
            exp_val = np.exp(group_sum)
            
            for tok, p in tok_probs:
                _, a1 = self.assign_axes(tok)
                if tok not in pathA_scores:
                    pathA_scores[tok] = 0.0
                pathA_scores[tok] += exp_val * (p / max(group_sum, 1e-12))
        
        # PATH B: Axis1 → Axis0
        pathB_scores = {}
        for a1_id, tok_probs in axis1_groups.items():
            group_sum = sum(p for _, p in tok_probs)
            
            for tok, p in tok_probs:
                a0, _ = self.assign_axes(tok)
                if tok not in pathB_scores:
                    pathB_scores[tok] = 0.0
                pathB_scores[tok] += group_sum * (p / max(group_sum, 1e-12))
        
        # MERGE
        merged_scores = {}
        for tok in tokens:
            scoreA = pathA_scores.get(tok, 0.0)
            scoreB = pathB_scores.get(tok, 0.0)
            merged_scores[tok] = scoreA + scoreB
        
        Z_merged = sum(merged_scores.values())
        if Z_merged > 1e-12:
            merged_scores = {k: v/Z_merged for k, v in merged_scores.items()}
        
        # Blend with original
        blended = {}
        for tok in tokens:
            orig_p = probs_dict[tok]
            diagram_p = merged_scores.get(tok, 0.0)
            blended[tok] = (1 - blend_weight) * orig_p + blend_weight * diagram_p
        
        Z = sum(blended.values())
        if Z <= 1e-12:
            return probs_dict
        return {k: v/Z for k, v in blended.items()}

# ------------------------
# 4D VSA (muscle embedding)
# ------------------------
class VSA4D:
    """Create random unit 4D embeddings for tokens (C,X,Y,Z)."""
    def __init__(self, shape=(4,8,8,4), seed=1234):
        self.shape = shape
        self.dim = np.prod(shape)
        self.book = {}
        self.rng = np.random.RandomState(seed)

    def vec_flat(self, sym):
        if sym not in self.book:
            v = self.rng.normal(size=(self.dim,))
            self.book[sym] = l2norm(v)
        return self.book[sym]

    def vec4(self, sym):
        return unflatten4(self.vec_flat(sym), self.shape)

# ------------------------
# Zero-Kernel Projector (4D aware)
# ------------------------
class ZeroKernelProjector4D:
    """
    Maps contexts to flat vectors built from 4D VSA embeddings.
    Projections use SVD on stacked context vectors.
    """
    def __init__(self, vsa_shape, n_components=64, kernel_dim=128, alpha=0.6, max_contexts=4000):
        self.vsa_shape = vsa_shape
        self.vsa_dim = np.prod(vsa_shape)
        self.n_components = n_components
        self.kernel_dim = kernel_dim
        self.alpha = alpha
        self.max_contexts = max_contexts
        self.mean = None
        self.U_data = None
        self.U_kernel = None
        self.sigma = None

    def fit(self, corp, vsa4d):
        X_rows = []
        ctx_count = 0
        for seq in corp:
            if ctx_count >= self.max_contexts:
                break
            for i in range(1, len(seq)):
                if ctx_count >= self.max_contexts:
                    break
                ctx = seq[max(0, i-8):i]
                vec = np.zeros((self.vsa_dim,), dtype=np.float64)
                for tok in ctx:
                    vec += vsa4d.vec_flat(tok)
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec /= norm
                    X_rows.append(vec)
                    ctx_count += 1
        if len(X_rows) == 0:
            return
        X = np.vstack(X_rows)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.sigma = s
        k = min(self.n_components, len(s))
        self.U_data = Vt[:k].T
        ks = min(self.kernel_dim, Vt.shape[0] - k)
        if ks > 0:
            self.U_kernel = Vt[k:k+ks].T
        else:
            self.U_kernel = None
        print(f"✓ Kernel fitted: data_dims={k}, kernel_dims={ks}")

    def project_probs(self, token_probs, vsa4d):
        """Returns blended probs biased toward kernel (null-space) directions."""
        if not token_probs or self.U_data is None or self.mean is None:
            return token_probs
        pvec = np.zeros((self.vsa_dim,), dtype=np.float64)
        for tok, p in token_probs.items():
            pvec += p * vsa4d.vec_flat(tok)
        pvec_c = pvec - self.mean
        data_comp = self.U_data @ (self.U_data.T @ pvec_c)
        kernel_comp = pvec_c - data_comp
        kn = np.linalg.norm(kernel_comp)
        if kn > 1e-10:
            kernel_comp /= kn
        ks = {tok: max(0.0, float(np.dot(vsa4d.vec_flat(tok) - self.mean, kernel_comp))) 
              for tok in token_probs}
        Z = sum(ks.values())
        if Z <= 1e-12:
            return token_probs
        ks = {k: v/Z for k,v in ks.items()}
        blended = {}
        for tok, p in token_probs.items():
            blended[tok] = (1 - self.alpha)*p + self.alpha*ks.get(tok, 0.0)
        Z2 = sum(blended.values())
        if Z2 <= 0:
            return token_probs
        return {k: v/Z2 for k,v in blended.items()}

# ------------------------
# DomeSpiral adapted
# ------------------------
class DomeSpiral4D:
    def __init__(self, n_spirals=40, dome_height=2.0, decay=0.15):
        self.n_spirals = n_spirals
        self.dome_height = dome_height
        self.decay = decay
        self.token_angles = {}
        self.spiral_phase = 0.0

    def assign_angle(self, tok):
        if tok not in self.token_angles:
            h = (hash(tok) & 0xFFFFFFFF) / 0xFFFFFFFF
            self.token_angles[tok] = h * 2*np.pi
        return self.token_angles[tok]

    def dome_z(self, r):
        return self.dome_height * (1 - r**2)

    def spiral_weight(self, theta, r, spiral_idx):
        spiral_base = 2*np.pi * spiral_idx / max(1, self.n_spirals)
        spiral_theta = spiral_base + 3.0*r + self.spiral_phase
        delta = np.arctan2(np.sin(theta - spiral_theta), np.cos(theta - spiral_theta))
        sigma2 = 0.32
        weight = np.exp(- (delta**2) / sigma2)
        weight *= (0.5 + 0.5 * self.dome_z(r)/self.dome_height)
        return weight

    def modulate(self, probs, ctx_len, blend=0.3):
        if not probs:
            return {}
        self.spiral_phase = (ctx_len * 0.8) % (2*np.pi)
        toks_sorted = sorted(probs.items(), key=lambda x:-x[1])
        n = len(toks_sorted)
        weights = {}
        for rank, (tok, p) in enumerate(toks_sorted):
            r = (rank+1)/(n+1)
            theta = self.assign_angle(tok)
            ss = 0.0
            for si in range(self.n_spirals):
                ss += self.spiral_weight(theta, r, si)
            weights[tok] = (ss/self.n_spirals) * (self.decay ** r)
        S = sum(weights.values())
        if S > 0:
            weights = {k: v/S for k,v in weights.items()}
        res = {tok: (1-blend)*p + blend*weights.get(tok, 0.0) for tok,p in probs.items()}
        Z = sum(res.values())
        if Z<=0:
            return probs
        return {k: v/Z for k,v in res.items()}

# ------------------------
# Knowledge subsets
# ------------------------
class KnowledgeSubsets4D:
    def __init__(self, n_clusters=8, cluster_size=200):
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.clusters = {}
        self.token_to_cluster = {}
        self.cluster_centers = {}

    def build(self, vocab, vsa4d):
        if not vocab:
            return
        vecs = {tok: vsa4d.vec_flat(tok) for tok in vocab}
        centers = list(vocab)[:self.n_clusters]
        self.cluster_centers = {i: centers[i] for i in range(len(centers))}
        clusters = defaultdict(list)
        for _ in range(3):
            clusters.clear()
            self.token_to_cluster.clear()
            for tok in vocab:
                v = vecs[tok]
                best, bsim = 0, -1.0
                for cid in self.cluster_centers:
                    c_tok = self.cluster_centers[cid]
                    sim = float(np.dot(v, vecs[c_tok]))
                    if sim > bsim:
                        bsim, best = sim, cid
                clusters[best].append(tok)
                self.token_to_cluster[tok] = best
            for cid, toks in clusters.items():
                if not toks:
                    continue
                best_center, best_avg = toks[0], -1.0
                for cand in toks:
                    cv = vecs[cand]
                    sims = [float(np.dot(cv, vecs[o])) for o in toks if o!=cand]
                    avg = np.mean(sims) if sims else -1.0
                    if avg > best_avg:
                        best_avg, best_center = avg, cand
                self.cluster_centers[cid] = best_center
        self.clusters = {cid: set(toks[:self.cluster_size]) for cid,toks in clusters.items() if toks}
        print(f"✓ Built {len(self.clusters)} knowledge clusters")

    def get_active(self, ctx):
        if not ctx:
            return set()
        active = set()
        for tok in ctx[-80:]:
            cid = self.token_to_cluster.get(tok)
            if cid is not None:
                active.add(cid)
        return active

    def boost(self, probs, active_clusters, boost_strength=0.4):
        if not active_clusters or not probs:
            return probs
        boosted = dict(probs)
        cluster_bonus = {}
        for cid in active_clusters:
            toks = self.clusters.get(cid, set())
            if not toks:
                cluster_bonus[cid] = 0.0
                continue
            bonus = sum(probs.get(tok, 0.0) for tok in toks)/max(1,len(toks))
            cluster_bonus[cid] = bonus
        for tok,p in probs.items():
            cid = self.token_to_cluster.get(tok)
            if cid in active_clusters:
                factor = 1.0 + boost_strength * cluster_bonus.get(cid, 0.0)
                boosted[tok] = p * factor
        Z = sum(boosted.values())
        if Z <= 0:
            return probs
        return {k: v/Z for k,v in boosted.items()}

# ------------------------
# Stochastic Cardinal Ordering (SCO)
# ------------------------
class StochasticCardinalOrder4D:
    def __init__(self, inversion_strength=1.7, noise_scale=0.05):
        self.inv_str = inversion_strength
        self.noise_scale = noise_scale
        self.attribution_history = deque(maxlen=500)
        self.cardinal_memory = {}

    def cardinal_rank(self, probs):
        sorted_items = sorted(probs.items(), key=lambda x:-x[1])
        return {tok: {"original_prob": p, "cardinal": i+1, "inverted_weight": (i+1)**self.inv_str}
                for i,(tok,p) in enumerate(sorted_items)}

    def stochastic_perturb(self, ranks):
        pert = {}
        for tok,data in ranks.items():
            g = -np.log(-np.log(np.random.uniform(1e-6, 1-1e-6)))
            pert[tok] = data["inverted_weight"] * (1 + self.noise_scale * g)
        return pert

    def attribution_transform(self, probs, ctx_attribution=0.5):
        if not probs:
            return {}
        ranks = self.cardinal_rank(probs)
        pert = self.stochastic_perturb(ranks)
        inv_factor = 0.3 + 0.7*ctx_attribution
        s = sum(pert.values()) or 1.0
        final = {}
        for tok, w in pert.items():
            orig = ranks[tok]["original_prob"]
            blended = (1-inv_factor)*orig + inv_factor*(w/s)
            final[tok] = blended
            self.cardinal_memory[tok] = self.cardinal_memory.get(tok, 0) + ranks[tok]["cardinal"]
        Z = sum(final.values())
        if Z <= 0.1:
            return probs
        return {k: v/Z for k,v in final.items()}

    def record_attribution(self, ctx, tok, prob):
        self.attribution_history.append({"ctx": tuple(ctx), "tok": tok, "prob": prob})

# ------------------------
# The 4D Muscle (Complete)
# ------------------------
class FourDMuscle:
    def __init__(self, vsa_shape=(4,8,8,4)):
        self.vsa = VSA4D(shape=vsa_shape)
        self.kernel = ZeroKernelProjector4D(vsa_shape=vsa_shape, n_components=64, kernel_dim=128, alpha=0.6)
        self.dome = DomeSpiral4D(n_spirals=50, dome_height=3.0, decay=0.15)
        self.knowledge = KnowledgeSubsets4D(n_clusters=12, cluster_size=500)
        self.sco = StochasticCardinalOrder4D(inversion_strength=2.5, noise_scale=0.08)
        self.diagram = DiagramProcessor(n_axis0_groups=8, n_axis1_groups=8)
        self.trivector = TrivectorAssociation(vsa_dim=self.vsa.dim)
        self.uni = Counter()
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.ctx_index = {}
        self.ctx_index_norm = {}

    def train(self, corp):
        print("="*70)
        print("Training 4D Muscle...")
        print("="*70)
        
        # Build unigram & rings
        for seq in corp:
            for i,t in enumerate(seq):
                self.uni[t] += 1
                if i>0:
                    prev = seq[i-1]
                    for r in range(len(self.rings)):
                        self.rings[r][prev][t] += 1 + 0.2 * np.cos(2*np.pi*r/len(self.rings))
        print(f"✓ Built unigram: {len(self.uni)} tokens")
        
        # Context index
        ctx_idx = defaultdict(Counter)
        for seq in corp:
            L = len(seq)
            for i,t in enumerate(seq):
                for j in range(max(0, i-6), min(L, i+7)):
                    if j==i: continue
                    ctx_idx[t][seq[j]] += 1
        self.ctx_index = dict(ctx_idx)
        for tok, ctr in self.ctx_index.items():
            tot = sum(ctr.values())
            self.ctx_index_norm[tok] = {k:v/tot for k,v in ctr.items()} if tot>0 else {}
        print(f"✓ Built context index: {len(self.ctx_index_norm)} tokens")
        
        # Knowledge subsets
        vocab = list(self.uni.keys())
        self.knowledge.build(vocab, self.vsa)
        
        # Kernel
        self.kernel.fit(corp, self.vsa)
        
        print("="*70)
        print("✓ Training complete")
        print("="*70)
        print()

    def probs_raw(self, ctx):
        if not ctx:
            t = sum(self.uni.values())
            return {w: c/t for w,c in self.uni.items()} if t else {}
        last = ctx[-1]
        agg = Counter()
        for ri, ring in enumerate(self.rings):
            if last in ring:
                row = ring[last]
                tot = sum(row.values())
                if tot<=0: continue
                factor = 1.0 + ri/len(self.rings)
                for nt,c in row.items():
                    agg[nt] += (c/tot) * factor
        tot = sum(agg.values())
        if tot>0:
            return {k:v/tot for k,v in agg.items()}
        return self.probs_raw([])

    def compute_intent_scores(self, ctx, base_probs):
        if not base_probs: return {}
        tokens = list(base_probs.keys())
        scores = {t:0.0 for t in tokens}
        
        if ctx:
            last = ctx[-1]
            for ri, ring in enumerate(self.rings):
                if last in ring:
                    row = ring[last]
                    tt = sum(row.values())
                    if tt>0:
                        w_ring = 1.0 + ri/len(self.rings)
                        for t in tokens:
                            c = row.get(t,0)
                            if c>0:
                                scores[t] += w_ring * (c/tt)
        
        # Self-context similarity
        cur_sig = self.current_context_signature(ctx, window=32)
        if cur_sig:
            for t in tokens:
                neigh = self.ctx_index_norm.get(t)
                if not neigh: continue
                s = 0.0
                for w,nw in neigh.items():
                    cw = cur_sig.get(w)
                    if cw is not None:
                        s += cw * nw
                scores[t] += 0.9 * s
        
        # Cluster activation
        active = self.knowledge.get_active(ctx)
        if active:
            for cid in active:
                toks = self.knowledge.clusters.get(cid,set())
                if not toks: continue
                cluster_base = sum(base_probs.get(x,0.0) for x in toks)/max(1,len(toks))
                for t in tokens:
                    if self.knowledge.token_to_cluster.get(t)==cid:
                        scores[t] += 0.7*cluster_base
        
        # Unigram boost
        totuni = sum(self.uni.values())
        if totuni>0:
            for t in tokens:
                f = self.uni[t]/totuni
                scores[t] += 0.15 * np.sqrt(f + 1e-12)
        
        return scores

    def apply_intent(self, base_probs, intent_scores, intent_alpha=6.0, intent_temp=1.0):
        if not base_probs or not intent_scores:
            return base_probs
        toks = list(base_probs.keys())
        p = np.array([base_probs[t] for t in toks], dtype=np.float64)
        s = np.array([intent_scores.get(t,0.0) for t in toks], dtype=np.float64)
        if np.all(s==0):
            return base_probs
        s = (s - s.mean())/(s.std()+1e-12)
        logm = intent_alpha * s / max(intent_temp, 1e-6)
        m = np.exp(logm)
        p_mod = p * m
        Z = p_mod.sum()
        if Z<=0:
            return base_probs
        p_mod /= Z
        return {tok: float(v) for tok,v in zip(toks,p_mod)}

    def current_context_signature(self, ctx, window=32):
        if not ctx:
            return {}
        sig = Counter(ctx[-window:])
        tot = sum(sig.values())
        if tot>0:
            for k in sig: sig[k] /= tot
        return dict(sig)

    def probs(self, ctx):
        raw = self.probs_raw(ctx)
        if not raw: return raw
        
        # 1) Permutation activation
        tokens = list(raw.keys())
        base = np.array([raw[t] for t in tokens], dtype=np.float64)
        base = np.maximum(base, 1e-12)
        base /= base.sum()
        g = -np.log(-np.log(np.random.uniform(1e-6,1-1e-6, size=base.shape)))
        logp = np.log(base) + 0.12 * g
        p_tilde = np.exp(logp - logp.max())
        p_tilde /= p_tilde.sum()
        perm_probs = {tok:float(p_tilde[i]) for i,tok in enumerate(tokens)}

        # 2) Intent scoring
        intent_scores = self.compute_intent_scores(ctx, perm_probs)
        intented = self.apply_intent(perm_probs, intent_scores)

        # 3) DIAGRAM PROCESSING (dual-path)
        diagrammed = self.diagram.apply_diagram_dict(intented, blend_weight=0.3)

        # 4) TRIVECTOR PRELIMINATION
        trivec_filtered = self.trivector.apply_prelimination(
            diagrammed, ctx, self.vsa, alpha=0.25, method='holographic'
        )

        # 5) Dome spiral
        domeed = self.dome.modulate(trivec_filtered, len(ctx), blend=0.25)

        # 6) Knowledge clusters
        active = self.knowledge.get_active(ctx)
        kn = self.knowledge.boost(domeed, active, boost_strength=0.4)

        # 7) Kernel projection
        kerneled = self.kernel.project_probs(kn, self.vsa)

        # 8) SCO reordering
        ctx_attr = 0.5
        final = self.sco.attribution_transform(kerneled, ctx_attr)

        return final

    def sample(self, ctx, temperature=1.0):
        ps = self.probs(ctx)
        if not ps:
            return None
        toks = list(ps.keys())
        p = np.array([ps[t] for t in toks], dtype=np.float64)
        p = np.maximum(p, 0)
        if p.sum()<=0:
            return None
        
        # Temperature
        if temperature!=1.0:
            logits = np.log(p + 1e-12)/temperature
            logits = np.exp(logits - logits.max())
            logits /= logits.sum()
            p = logits
        p /= p.sum()
        idx = np.random.choice(len(toks), p=p)
        tok = toks[idx]
        
        # Record attribution
        self.sco.record_attribution(tuple(ctx[-2:]), tok, float(p[idx]))
        return tok

# ------------------------
# Example usage
# ------------------------
def example_usage():
    print("="*70)
    print("4D MUSCLE - COMPLETE SYSTEM")
    print("Diagram + Trivector")
    print("="*70)
    print()
    
    with open(input("Corpus filename: "), encoding="UTF-8") as f:
        text = f.read().lower()
    
    sentences = [s.strip() for s in text.split(".") if s.strip()][:KB_len]
    corpus = [sent.split() for sent in sentences]
    
    print(f"Loaded {len(corpus)} sentences")
    print()
    
    muscle = FourDMuscle(vsa_shape=(4,8,8,4))
    muscle.train(corpus)
    
    while True:
        user_input = input("\nUSER: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        ctx = user_input.split()
        print(f"Starting context: {ctx}")
        print("Generated: ", end="", flush=True)
        
        out = []
        for i in range(LEN):
            tok = muscle.sample(ctx, temperature=0.9)
            if tok is None:
                break
            out.append(tok)
            print(tok, end=" ", flush=True)
            ctx.append(tok)
        
        print(f"\n\nGenerated {len(out)} tokens")
        print(f"Pipeline stats:")
        print(f"  - Trivector cache: {len(muscle.trivector.trivector_cache)} entries")
        print(f"  - SCO memory: {len(muscle.sco.cardinal_memory)} tokens")
        print(f"  - Diagram axis0 map: {len(muscle.diagram.token_axis0_map)} tokens")

if __name__ == "__main__":
    example_usage()

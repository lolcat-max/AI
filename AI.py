"""
fourd_muscle.py (FIXED)

Convert the ConcGen pipeline into a '4D muscle' module:
- internal embeddings / activations are 4D tensors (C, X, Y, Z)
- algorithmic building blocks adapted to 4D
- retains ZeroKernelProjector, DomeSpiral, KnowledgeSubsets, SCO, etc.
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
# 4D VSA (muscle embedding)
# ------------------------
class VSA4D:
    """Create random unit 4D embeddings for tokens (C,X,Y,Z)."""
    def __init__(self, shape=(4,8,8,4), seed=1234):
        # shape = (channels, x, y, z)
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
    Works like your ZeroKernelProjector but maps contexts to flat vectors
    built from 4D VSA embeddings. Projections use SVD on stacked context vectors.
    """
    def __init__(self, vsa_shape, n_components=64, kernel_dim=128, alpha=0.6, max_contexts=4000):
        self.vsa_shape = vsa_shape
        self.vsa_dim = np.prod(vsa_shape)
        self.n_components = n_components
        self.kernel_dim = kernel_dim
        self.alpha = alpha
        self.max_contexts = max_contexts

        # learned
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
                # create a superposition flat vector
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
        self.U_data = Vt[:k].T  # (dim, k)
        ks = min(self.kernel_dim, Vt.shape[0] - k)
        if ks > 0:
            self.U_kernel = Vt[k:k+ks].T
        else:
            self.U_kernel = None

    def project_probs(self, token_probs, vsa4d):
        """
        token_probs: dict tok -> prob
        Returns blended probs where we bias toward kernel (null-space) directions.
        """
        if not token_probs or self.U_data is None or self.mean is None:
            return token_probs
        # build prob vector
        pvec = np.zeros((self.vsa_dim,), dtype=np.float64)
        for tok, p in token_probs.items():
            pvec += p * vsa4d.vec_flat(tok)
        pvec_c = pvec - self.mean
        # data component
        data_comp = self.U_data @ (self.U_data.T @ pvec_c)
        kernel_comp = pvec_c - data_comp
        kn = np.linalg.norm(kernel_comp)
        if kn > 1e-10:
            kernel_comp /= kn
        # score tokens by dot with kernel_comp (flat)
        ks = {tok: max(0.0, float(np.dot(vsa4d.vec_flat(tok) - self.mean, kernel_comp))) for tok in token_probs}
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
# DomeSpiral adapted (uses angles per token but can modulate 4D fields)
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
        # blend
        res = {tok: (1-blend)*p + blend*weights.get(tok, 0.0) for tok,p in probs.items()}
        Z = sum(res.values())
        if Z<=0:
            return probs
        return {k: v/Z for k,v in res.items()}

# ------------------------
# Knowledge subsets (keeps tokens grouped, same logic)
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
            # recompute centers
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
# The 4D Muscle (wrapper combining everything)
# ------------------------
class FourDMuscle:
    def __init__(self, vsa_shape=(4,8,8,4)):
        self.vsa = VSA4D(shape=vsa_shape)
        self.kernel = ZeroKernelProjector4D(vsa_shape=vsa_shape, n_components=64, kernel_dim=128, alpha=0.6)
        self.dome = DomeSpiral4D(n_spirals=50, dome_height=3.0, decay=0.15)
        self.knowledge = KnowledgeSubsets4D(n_clusters=12, cluster_size=500)
        self.sco = StochasticCardinalOrder4D(inversion_strength=2.5, noise_scale=0.08)
        self.uni = Counter()
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.ctx_index = {}
        self.ctx_index_norm = {}

    def train(self, corp):
        # build unigram & rings & ctx_index (lightweight)
        for seq in corp:
            for i,t in enumerate(seq):
                self.uni[t] += 1
                if i>0:
                    prev = seq[i-1]
                    for r in range(len(self.rings)):
                        self.rings[r][prev][t] += 1 + 0.2 * np.cos(2*np.pi*r/len(self.rings))
        # context index
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
        # knowledge subsets
        vocab = list(self.uni.keys())
        self.knowledge.build(vocab, self.vsa)
        # kernel
        self.kernel.fit(corp, self.vsa)

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
        # self-context similarity
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
        # cluster activation
        active = self.knowledge.get_active(ctx)
        if active:
            for cid in active:
                toks = self.knowledge.clusters.get(cid,set())
                if not toks: continue
                cluster_base = sum(base_probs.get(x,0.0) for x in toks)/max(1,len(toks))
                for t in tokens:
                    if self.knowledge.token_to_cluster.get(t)==cid:
                        scores[t] += 0.7*cluster_base
        # unigram boost
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

    def apply_diagram_flow(self, probs_matrix, user_input_indices, token_list):
        """
        Implements the diagram architecture:
        - Takes probability matrix as input
        - Uses gradient-based range selection
        - Applies feed-forward control
        - Returns generation feedback adjusted probabilities
        """
        # Gradient where with range (from user input)
        if len(user_input_indices) > 0:
            grad_range = np.zeros_like(probs_matrix)
            for idx in user_input_indices:
                if idx < len(probs_matrix):
                    grad_range[idx] = 1.0
        else:
            grad_range = np.ones_like(probs_matrix)
        
        # np.exp() application (exponential transform)
        exp_transform = np.exp(probs_matrix * grad_range)
        
        # Feed forward control (normalize and scale)
        feed_forward = exp_transform / (exp_transform.sum() + 1e-12)
        
        # Generation feedback (modulate by dataset similarity using token space)
        # Build a weighted embedding from current probabilities
        prob_embedding = np.zeros((self.vsa.dim,), dtype=np.float64)
        for i, tok in enumerate(token_list):
            prob_embedding += feed_forward[i] * self.vsa.vec_flat(tok)
        
        # Compare with dataset tokens to get feedback signal
        dataset_tokens = list(self.uni.keys())[:min(20, len(self.uni))]
        similarities = []
        for dtok in dataset_tokens:
            sim = np.dot(prob_embedding, self.vsa.vec_flat(dtok))
            similarities.append(sim)
        
        if similarities:
            dataset_influence = np.mean(similarities)
            feedback_factor = 1.0 + 0.3 * np.tanh(dataset_influence)
            feed_forward *= feedback_factor
        
        # Final normalization
        output = feed_forward / (feed_forward.sum() + 1e-12)
        return output

    def probs(self, ctx):
        raw = self.probs_raw(ctx)
        if not raw: return raw
        # perm activation (light version)
        tokens = list(raw.keys())
        base = np.array([raw[t] for t in tokens], dtype=np.float64)
        base = np.maximum(base, 1e-12)
        base /= base.sum()
        
        # DIAGRAM INTEGRATION: User input indices from context
        user_input_indices = []
        if ctx:
            # Map recent context tokens to indices
            for tok in ctx[-3:]:
                if tok in tokens:
                    user_input_indices.append(tokens.index(tok))
        
        # DIAGRAM INTEGRATION: Apply the diagram flow (pass token list instead of embeddings)
        probs_matrix = np.log(base + 1e-12)
        diagram_output = self.apply_diagram_flow(probs_matrix, user_input_indices, tokens)
        
        # Convert back to dict and blend with original
        diagram_probs = {tok: float(diagram_output[i]) for i, tok in enumerate(tokens)}
        
        # Blend diagram output with Gumbel perturbation
        g = -np.log(-np.log(np.random.uniform(1e-6,1-1e-6, size=base.shape)))
        logp = np.log(base) + 0.12 * g
        p_tilde = np.exp(logp - logp.max())
        p_tilde /= p_tilde.sum()
        gumbel_probs = {tok:float(p_tilde[i]) for i,tok in enumerate(tokens)}
        
        # Merge diagram and gumbel (70% diagram, 30% gumbel)
        perm_probs = {}
        for tok in tokens:
            perm_probs[tok] = 0.7 * diagram_probs[tok] + 0.3 * gumbel_probs[tok]
        
        # Normalize
        Z = sum(perm_probs.values())
        if Z > 0:
            perm_probs = {k: v/Z for k, v in perm_probs.items()}

        # intent
        intent_scores = self.compute_intent_scores(ctx, perm_probs)
        intented = self.apply_intent(perm_probs, intent_scores)

        # dome spiral
        domeed = self.dome.modulate(intented, len(ctx), blend=0.25)

        # knowledge clusters
        active = self.knowledge.get_active(ctx)
        kn = self.knowledge.boost(domeed, active, boost_strength=0.4)

        # kernel projection -> blend to kernel
        kerneled = self.kernel.project_probs(kn, self.vsa)

        # SCO reorder
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
        # temperature
        if temperature!=1.0:
            logits = np.log(p + 1e-12)/temperature
            logits = np.exp(logits - logits.max())
            logits /= logits.sum()
            p = logits
        p /= p.sum()
        idx = np.random.choice(len(toks), p=p)
        tok = toks[idx]
        # record attribution
        self.sco.record_attribution(tuple(ctx[-2:]), tok, float(p[idx]))
        return tok

# ------------------------
# Example usage
# ------------------------
def example_usage():
    with open(input("Filename: "), encoding="UTF-8") as f:
        text = f.read().lower().split(".")[:KB_len]
    corpus = []
    for sentence in text:
        corpus.append(sentence.split())
    muscle = FourDMuscle(vsa_shape=(1,8,8,2))
    muscle.train(corpus)
    while True:
        ctx = input("USER: ").split()
        print("starting ctx:", ctx)
        out = []
        for i in range(LEN):
            tok = muscle.sample(ctx, temperature=0.9)
            if tok is None:
                break
            out.append(tok)
            ctx.append(tok)
        print("generated:", ' '.join(out))

if __name__ == "__main__":
    example_usage()

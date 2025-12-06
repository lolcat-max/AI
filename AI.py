"""
fourd_muscle.py (FIXED)

Convert the ConcGen pipeline into a '4D muscle' module:
- internal embeddings / activations are 4D tensors (C, X, Y, Z)
- algorithmic building blocks adapted to 4D
- retains ZeroKernelProjector, DomeSpiral, KnowledgeSubsets, SCO, etc.
- multiprocessing for fast dataset processing
"""
KB_len = 999
LEN = 999
import numpy as np
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
import random
from multiprocessing import Pool, cpu_count
from functools import partial

# ------------------------
# Multiprocessing helpers
# ------------------------
def process_sequence_unigrams(seq):
    """Process a single sequence for unigram counts"""
    return Counter(seq)

def process_sequence_rings(args):
    """Process a single sequence for ring transitions"""
    seq, n_rings = args
    rings = [defaultdict(Counter) for _ in range(n_rings)]
    for i in range(1, len(seq)):
        prev = seq[i-1]
        curr = seq[i]
        for r in range(n_rings):
            rings[r][prev][curr] += 1 + 0.2 * np.cos(2*np.pi*r/n_rings)
    return rings

def process_sequence_context(seq):
    """Process a single sequence for context index"""
    ctx_idx = defaultdict(Counter)
    L = len(seq)
    for i, t in enumerate(seq):
        for j in range(max(0, i-6), min(L, i+7)):
            if j == i:
                continue
            ctx_idx[t][seq[j]] += 1
    return dict(ctx_idx)

def merge_counters(counter_list):
    """Merge a list of Counters into one"""
    result = Counter()
    for c in counter_list:
        result.update(c)
    return result

def merge_ring_dicts(ring_list):
    """Merge ring dictionaries from multiple processes"""
    n_rings = len(ring_list[0])
    merged = [defaultdict(Counter) for _ in range(n_rings)]
    for rings in ring_list:
        for r in range(n_rings):
            for prev, counts in rings[r].items():
                merged[r][prev].update(counts)
    return merged

def merge_context_dicts(ctx_list):
    """Merge context index dictionaries"""
    merged = defaultdict(Counter)
    for ctx_dict in ctx_list:
        for tok, counts in ctx_dict.items():
            merged[tok].update(counts)
    return dict(merged)

def generate_single_sequence(args):
    """
    Generate a single sequence (for multiprocessing).
    Returns the generated token sequence.
    """
    muscle_data, initial_ctx, max_len, temperature, seed = args
    
    # Set seed for this worker
    np.random.seed(seed)
    random.seed(seed)
    
    # Reconstruct muscle object (lightweight reconstruction)
    muscle = FourDMuscle(vsa_shape=muscle_data['vsa_shape'])
    muscle.uni = muscle_data['uni']
    muscle.rings = muscle_data['rings']
    muscle.ctx_index = muscle_data['ctx_index']
    muscle.ctx_index_norm = muscle_data['ctx_index_norm']
    muscle.vsa.book = muscle_data['vsa_book']
    muscle.knowledge.clusters = muscle_data['knowledge_clusters']
    muscle.knowledge.token_to_cluster = muscle_data['knowledge_token_to_cluster']
    muscle.knowledge.cluster_centers = muscle_data['knowledge_cluster_centers']
    muscle.kernel.mean = muscle_data['kernel_mean']
    muscle.kernel.U_data = muscle_data['kernel_U_data']
    muscle.kernel.U_kernel = muscle_data['kernel_U_kernel']
    
    # Generate sequence
    ctx = list(initial_ctx)
    out = []
    for _ in range(max_len):
        tok = muscle.sample(ctx, temperature=temperature)
        if tok is None:
            break
        out.append(tok)
        ctx.append(tok)
    
    return out

def compute_token_similarities(args):
    """
    Compute similarities between a token and all vocabulary tokens.
    Used for parallel clustering in KnowledgeSubsets.
    """
    tok, vocab, vecs = args
    v = vecs[tok]
    similarities = {}
    for other_tok in vocab:
        if tok != other_tok:
            similarities[other_tok] = float(np.dot(v, vecs[other_tok]))
    return tok, similarities

def assign_tokens_to_clusters(args):
    """
    Assign a batch of tokens to their nearest cluster.
    """
    tokens, cluster_centers, vecs = args
    assignments = {}
    for tok in tokens:
        v = vecs[tok]
        best_cluster, best_sim = 0, -1.0
        for cid, c_tok in cluster_centers.items():
            sim = float(np.dot(v, vecs[c_tok]))
            if sim > best_sim:
                best_sim, best_cluster = sim, cid
        assignments[tok] = best_cluster
    return assignments

def find_best_center_for_cluster(args):
    """
    Find the best center token for a cluster (token with highest avg similarity).
    """
    cluster_tokens, vecs = args
    if not cluster_tokens:
        return None
    
    best_center, best_avg = cluster_tokens[0], -1.0
    for cand in cluster_tokens:
        cv = vecs[cand]
        sims = [float(np.dot(cv, vecs[other])) for other in cluster_tokens if other != cand]
        avg = np.mean(sims) if sims else -1.0
        if avg > best_avg:
            best_avg, best_center = avg, cand
    
    return best_center
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
# Numeric Euclidean Space
# ------------------------
class NumericEuclideanSpace:
    """
    Separate Euclidean space for processing numeric tokens with logic operations.
    Numbers are embedded in a different geometric space than text tokens.
    """
    def __init__(self, dim=64):
        self.dim = dim
        self.numeric_cache = {}
        self.logic_operators = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / (y + 1e-12),
            'mod': lambda x, y: x % (y + 1e-12),
            'pow': lambda x, y: np.power(x, np.clip(y, -10, 10)),
            'max': lambda x, y: np.maximum(x, y),
            'min': lambda x, y: np.minimum(x, y),
        }
    
    def is_numeric(self, tok):
        """Check if token represents a number"""
        try:
            float(tok)
            return True
        except (ValueError, TypeError):
            return False
    
    def numeric_embedding(self, tok):
        """
        Embed a numeric token in Euclidean space using multiple geometric projections.
        Creates a rich representation across different mathematical spaces.
        """
        if tok in self.numeric_cache:
            return self.numeric_cache[tok]
        
        try:
            val = float(tok)
        except (ValueError, TypeError):
            # Non-numeric fallback: hash-based embedding
            h = hash(tok) & 0xFFFFFFFF
            emb = np.array([np.sin(h * i * 0.01) for i in range(self.dim)])
            emb = l2norm(emb)
            self.numeric_cache[tok] = emb
            return emb
        
        # Handle special values
        if not np.isfinite(val):
            # Infinity or NaN - use special embedding
            h = hash(str(tok)) & 0xFFFFFFFF
            emb = np.array([np.cos(h * i * 0.01) for i in range(self.dim)])
            emb = l2norm(emb)
            self.numeric_cache[tok] = emb
            return emb
        
        # Clip extreme values to prevent overflow
        val = np.clip(val, -1e10, 1e10)
        
        # Multi-space numeric embedding
        emb = np.zeros(self.dim)
        
        # 1. Linear space (raw magnitude encoding)
        emb[0:8] = np.tanh(val * np.linspace(0.01, 1.0, 8))
        
        # 2. Logarithmic space (scale-invariant)
        log_val = np.log1p(np.abs(val)) * np.sign(val)
        emb[8:16] = np.tanh(log_val * np.linspace(0.1, 2.0, 8))
        
        # 3. Trigonometric space (periodic features)
        emb[16:24] = np.sin(val * np.linspace(0.1, 10.0, 8))
        emb[24:32] = np.cos(val * np.linspace(0.1, 10.0, 8))
        
        # 4. Fractional space (decimal structure)
        try:
            int_val = int(val)
            frac = val - int_val
        except (OverflowError, ValueError):
            frac = 0.0
        emb[32:40] = np.sin(frac * 2 * np.pi * np.arange(1, 9))
        
        # 5. Modular arithmetic spaces
        for i, mod in enumerate([2, 3, 5, 7, 11, 13, 17, 19]):
            try:
                emb[40 + i] = np.sin(2 * np.pi * (val % mod) / mod)
            except (OverflowError, ValueError):
                emb[40 + i] = 0.0
        
        # 6. Sign and magnitude decomposition
        emb[48:56] = np.sign(val) * np.exp(-np.abs(val) * np.linspace(0.01, 1.0, 8))
        
        # 7. Polynomial basis (with clipping to prevent overflow)
        safe_val = np.clip(np.abs(val), 0.0, 1000.0)
        emb[56:64] = np.array([safe_val**p for p in np.linspace(0.5, 2.0, 8)])
        # Clip polynomial results
        emb[56:64] = np.clip(emb[56:64], -1e6, 1e6)
        
        # Normalize
        emb = l2norm(emb)
        self.numeric_cache[tok] = emb
        return emb
    
    def apply_logic(self, num_embeddings, operator='add'):
        """
        Apply logical/arithmetic operations between numeric embeddings.
        Operates in the numeric Euclidean space.
        """
        if len(num_embeddings) < 2:
            return num_embeddings[0] if num_embeddings else np.zeros(self.dim)
        
        op_func = self.logic_operators.get(operator, self.logic_operators['add'])
        
        # Pairwise operation reduction
        result = num_embeddings[0].copy()
        for emb in num_embeddings[1:]:
            result = op_func(result, emb)
            # Clip to prevent overflow accumulation
            result = np.clip(result, -1e6, 1e6)
        
        # Handle any NaN or inf values
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return l2norm(result)
    
    def merge_with_text_space(self, numeric_emb, text_emb, blend=0.5):
        """
        Project numeric embedding into text embedding space using learned manifold.
        Creates a bridge between numeric and text Euclidean spaces.
        """
        # Ensure compatible dimensions
        if len(numeric_emb) < len(text_emb):
            # Expand numeric embedding
            numeric_expanded = np.zeros_like(text_emb)
            # Repeat and blend
            repeats = len(text_emb) // len(numeric_emb) + 1
            numeric_repeated = np.tile(numeric_emb, repeats)[:len(text_emb)]
            numeric_expanded = numeric_repeated
        elif len(numeric_emb) > len(text_emb):
            # Compress numeric embedding using average pooling
            pool_size = len(numeric_emb) // len(text_emb)
            numeric_expanded = np.zeros_like(text_emb)
            for i in range(len(text_emb)):
                start = i * pool_size
                end = min(start + pool_size, len(numeric_emb))
                numeric_expanded[i] = np.mean(numeric_emb[start:end])
        else:
            numeric_expanded = numeric_emb
        
        # Blend in joint manifold
        merged = blend * numeric_expanded + (1 - blend) * text_emb
        
        # Apply non-linear projection (simulates manifold curvature)
        merged = np.tanh(merged * 2.0)
        
        return l2norm(merged)

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

    def build(self, vocab, vsa4d, n_processes=None):
        """
        Build knowledge clusters with multiprocessing support.
        
        Args:
            vocab: List of tokens
            vsa4d: VSA4D object for embeddings
            n_processes: Number of processes (None = auto)
        """
        if not vocab:
            return
        
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        
        # Precompute all vectors (shared across iterations)
        vecs = {tok: vsa4d.vec_flat(tok) for tok in vocab}
        
        # Initialize centers
        centers = list(vocab)[:self.n_clusters]
        self.cluster_centers = {i: centers[i] for i in range(len(centers))}
        
        # K-means iterations with parallel token assignment
        for iteration in range(3):
            clusters = defaultdict(list)
            self.token_to_cluster.clear()
            
            # Split vocabulary into chunks for parallel processing
            chunk_size = max(1, len(vocab) // (n_processes * 2))
            vocab_chunks = [vocab[i:i+chunk_size] for i in range(0, len(vocab), chunk_size)]
            
            # Parallel token assignment
            args_list = [(chunk, self.cluster_centers, vecs) for chunk in vocab_chunks]
            
            with Pool(n_processes) as pool:
                results = pool.map(assign_tokens_to_clusters, args_list)
            
            # Merge assignments
            for assignments in results:
                for tok, cid in assignments.items():
                    clusters[cid].append(tok)
                    self.token_to_cluster[tok] = cid
            
            # Recompute centers in parallel
            cluster_args = [(clusters[cid], vecs) for cid in self.cluster_centers.keys()]
            
            with Pool(n_processes) as pool:
                new_centers = pool.map(find_best_center_for_cluster, cluster_args)
            
            for cid, new_center in zip(self.cluster_centers.keys(), new_centers):
                if new_center is not None:
                    self.cluster_centers[cid] = new_center
        
        # Finalize clusters with size limit
        self.clusters = {cid: set(toks[:self.cluster_size]) 
                        for cid, toks in clusters.items() if toks}

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
        self.numeric_space = NumericEuclideanSpace(dim=64)
        self.kernel = ZeroKernelProjector4D(vsa_shape=vsa_shape, n_components=64, kernel_dim=128, alpha=0.6)
        self.dome = DomeSpiral4D(n_spirals=50, dome_height=3.0, decay=0.15)
        self.knowledge = KnowledgeSubsets4D(n_clusters=12, cluster_size=500)
        self.sco = StochasticCardinalOrder4D(inversion_strength=2.5, noise_scale=0.08)
        self.uni = Counter()
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.ctx_index = {}
        self.ctx_index_norm = {}

    def train(self, corp, n_processes=None):
        """
        Train the model on corpus using multiprocessing.
        
        Args:
            corp: List of sequences (each sequence is a list of tokens)
            n_processes: Number of processes to use (None = auto-detect)
        """
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        
        print(f"Training with {n_processes} processes on {len(corp)} sequences...")
        
        # Multiprocessing for unigram counts
        print("Building unigrams...")
        with Pool(n_processes) as pool:
            unigram_results = pool.map(process_sequence_unigrams, corp)
        self.uni = merge_counters(unigram_results)
        
        # Multiprocessing for ring transitions
        print("Building ring transitions...")
        ring_args = [(seq, len(self.rings)) for seq in corp]
        with Pool(n_processes) as pool:
            ring_results = pool.map(process_sequence_rings, ring_args)
        self.rings = merge_ring_dicts(ring_results)
        
        # Multiprocessing for context index
        print("Building context index...")
        with Pool(n_processes) as pool:
            ctx_results = pool.map(process_sequence_context, corp)
        self.ctx_index = merge_context_dicts(ctx_results)
        
        # Normalize context index
        print("Normalizing context index...")
        for tok, ctr in self.ctx_index.items():
            tot = sum(ctr.values())
            self.ctx_index_norm[tok] = {k: v/tot for k, v in ctr.items()} if tot > 0 else {}
        
        # Knowledge subsets with multiprocessing
        print("Building knowledge subsets (parallel k-means clustering)...")
        vocab = list(self.uni.keys())
        self.knowledge.build(vocab, self.vsa, n_processes=n_processes)
        
        # Kernel projector (single-threaded due to SVD)
        print("Fitting kernel projector...")
        self.kernel.fit(corp, self.vsa)
        
        print("Training complete!")

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
        
        # Separate numeric context analysis
        numeric_ctx = [tok for tok in ctx if self.numeric_space.is_numeric(tok)]
        if numeric_ctx:
            # Compute numeric logic pattern
            numeric_embs = [self.numeric_space.numeric_embedding(tok) for tok in numeric_ctx]
            numeric_pattern = self.numeric_space.apply_logic(numeric_embs, operator='add')
            
            # Boost tokens that align with numeric pattern
            for t in tokens:
                if self.numeric_space.is_numeric(t):
                    t_emb = self.numeric_space.numeric_embedding(t)
                    numeric_sim = np.dot(numeric_pattern, t_emb)
                    scores[t] += 1.2 * np.tanh(numeric_sim)
        
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
        Implements the diagram architecture with numeric logic merging:
        - Takes probability matrix as input
        - Uses gradient-based range selection
        - Applies feed-forward control
        - Merges numeric embeddings from different Euclidean space
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
        
        # NUMERIC LOGIC MERGING: Separate handling for numeric tokens
        numeric_tokens = [tok for tok in token_list if self.numeric_space.is_numeric(tok)]
        
        if numeric_tokens:
            # Extract numeric embeddings in their own Euclidean space
            numeric_embeddings = [self.numeric_space.numeric_embedding(tok) 
                                 for tok in numeric_tokens]
            
            # Apply logic operations (addition in this case, but could be any operator)
            merged_numeric = self.numeric_space.apply_logic(numeric_embeddings, operator='add')
            
            # Boost probabilities for numeric tokens using merged representation
            for i, tok in enumerate(token_list):
                if self.numeric_space.is_numeric(tok):
                    numeric_emb = self.numeric_space.numeric_embedding(tok)
                    # Similarity in numeric space
                    similarity = np.dot(merged_numeric, numeric_emb)
                    # Apply boost
                    feed_forward[i] *= (1.0 + 0.4 * np.tanh(similarity))
            
            # Renormalize after numeric boost
            feed_forward /= (feed_forward.sum() + 1e-12)
        
        # Generation feedback (modulate by dataset similarity using token space)
        # Build a weighted embedding from current probabilities
        prob_embedding = np.zeros((self.vsa.dim,), dtype=np.float64)
        for i, tok in enumerate(token_list):
            if self.numeric_space.is_numeric(tok):
                # Merge numeric embedding into text space
                num_emb = self.numeric_space.numeric_embedding(tok)
                text_emb = self.vsa.vec_flat(tok)
                merged = self.numeric_space.merge_with_text_space(num_emb, text_emb, blend=0.6)
                prob_embedding += feed_forward[i] * merged
            else:
                prob_embedding += feed_forward[i] * self.vsa.vec_flat(tok)
        
        # Compare with dataset tokens to get feedback signal
        dataset_tokens = list(self.uni.keys())[:min(20, len(self.uni))]
        similarities = []
        for dtok in dataset_tokens:
            if self.numeric_space.is_numeric(dtok):
                num_emb = self.numeric_space.numeric_embedding(dtok)
                text_emb = self.vsa.vec_flat(dtok)
                merged = self.numeric_space.merge_with_text_space(num_emb, text_emb, blend=0.6)
                sim = np.dot(prob_embedding, merged)
            else:
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
    
    def serialize_for_multiprocessing(self):
        """
        Serialize model data for multiprocessing.
        Returns a dictionary with all necessary data.
        """
        return {
            'vsa_shape': self.vsa.shape,
            'vsa_book': self.vsa.book,
            'uni': self.uni,
            'rings': self.rings,
            'ctx_index': self.ctx_index,
            'ctx_index_norm': self.ctx_index_norm,
            'knowledge_clusters': self.knowledge.clusters,
            'knowledge_token_to_cluster': self.knowledge.token_to_cluster,
            'knowledge_cluster_centers': self.knowledge.cluster_centers,
            'kernel_mean': self.kernel.mean,
            'kernel_U_data': self.kernel.U_data,
            'kernel_U_kernel': self.kernel.U_kernel,
        }
    
    def generate_parallel(self, initial_ctx, n_sequences=4, max_len=None, 
                         temperature=0.9, n_processes=None):
        """
        Generate multiple sequences in parallel.
        
        Args:
            initial_ctx: Starting context (list of tokens)
            n_sequences: Number of sequences to generate
            max_len: Maximum length per sequence (default: LEN)
            temperature: Sampling temperature
            n_processes: Number of processes (None = auto)
        
        Returns:
            List of generated sequences (each is a list of tokens)
        """
        if max_len is None:
            max_len = LEN
        
        if n_processes is None:
            n_processes = min(n_sequences, max(1, cpu_count() - 1))
        
        print(f"Generating {n_sequences} sequences with {n_processes} processes...")
        
        # Serialize model data
        muscle_data = self.serialize_for_multiprocessing()
        
        # Create arguments for each generation task
        args_list = [
            (muscle_data, initial_ctx, max_len, temperature, 
             np.random.randint(0, 1000000))
            for _ in range(n_sequences)
        ]
        
        # Generate in parallel
        with Pool(n_processes) as pool:
            results = pool.map(generate_single_sequence, args_list)
        
        return results

# ------------------------
# Example usage
# ------------------------
def example_usage():
    print("=== 4D Muscle Text Generator ===")
    filename = input("Filename: ")
    
    print(f"\nLoading and preprocessing {filename}...")
    with open(filename, encoding="UTF-8") as f:
        text = f.read().lower().split(".")[:KB_len]
    
    corpus = []
    for sentence in text:
        tokens = sentence.split()
        if tokens:  # Skip empty sentences
            corpus.append(tokens)
    
    print(f"Loaded {len(corpus)} sentences")
    
    # Initialize model
    print("\nInitializing 4D Muscle model...")
    muscle = FourDMuscle(vsa_shape=(1,8,8,2))
    
    # Train with multiprocessing
    n_procs = input("Number of processes (press Enter for auto): ").strip()
    n_procs = int(n_procs) if n_procs else None
    
    muscle.train(corpus, n_processes=n_procs)
    
    print("\n=== Generation Mode ===")
    print("Commands:")
    print("  - Enter context words for single generation")
    print("  - Type 'parallel N' to generate N sequences in parallel")
    print("  - Type 'batch' for batch parallel generation")
    print("  - Type 'quit' to exit\n")
    
    while True:
        user_input = input("USER: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Check for parallel generation command
        if user_input.lower().startswith('parallel'):
            parts = user_input.split()
            n_seq = int(parts[1]) if len(parts) > 1 else 4
            
            ctx_input = input("Enter starting context: ").strip()
            ctx = ctx_input.split() if ctx_input else []
            
            temp = input("Temperature (default 0.9): ").strip()
            temp = float(temp) if temp else 0.9
            
            max_len = input("Max length per sequence (default 999): ").strip()
            max_len = int(max_len) if max_len else LEN
            
            print()
            results = muscle.generate_parallel(ctx, n_sequences=n_seq, 
                                              max_len=max_len, temperature=temp)
            
            print("\n=== PARALLEL GENERATION RESULTS ===")
            for i, seq in enumerate(results, 1):
                print(f"\nSequence {i} ({len(seq)} tokens):")
                print(' '.join(seq))
            print()
            continue
        
        # Check for batch mode
        if user_input.lower() == 'batch':
            n_seq = input("Number of sequences: ").strip()
            n_seq = int(n_seq) if n_seq else 4
            
            ctx_input = input("Enter starting context: ").strip()
            ctx = ctx_input.split() if ctx_input else []
            
            print()
            results = muscle.generate_parallel(ctx, n_sequences=n_seq)
            
            print("\n=== BATCH GENERATION RESULTS ===")
            for i, seq in enumerate(results, 1):
                print(f"\n[{i}] " + ' '.join(seq))
            print()
            continue
        
        # Single generation
        ctx = user_input.split()
        print(f"Starting context: {ctx}")
        print("Generating...\n")
        
        out = []
        for i in range(LEN):
            tok = muscle.sample(ctx, temperature=0.9)
            if tok is None:
                break
            out.append(tok)
            ctx.append(tok)
        
        print("GENERATED:", ' '.join(out))
        print()

if __name__ == "__main__":
    example_usage()

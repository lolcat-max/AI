# =====================================================================
# COUNTER-INTUITIVE STOCHASTIC CARDINAL ORDERING ON ATTRIBUTION
# WITH DOME SPIRAL + SELF-CONTEXT + KNOWLEDGE SUBSETS + UNKNOWN CTX
# + PERMUTATION ACTIVATIONS
# =====================================================================
KB_len = 99999

import numpy as np
import random
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm


# ---------------------------------------------------------------------
# KNOWLEDGE SUBSET CLUSTERING
# ---------------------------------------------------------------------
class KnowledgeSubsets:
    """
    Clusters tokens into coarse 'knowledge subsets' using VSA cosine similarity.
    Used to softly boost tokens from clusters that are active in the current context.
    """
    def __init__(self, n_clusters=8, cluster_size=50):
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.clusters = {}           # cluster_id -> set(tokens)
        self.token_to_cluster = {}   # tok -> cluster_id
        self.cluster_centers = {}    # cluster_id -> representative token

    def build_clusters(self, vocab, vsa):
        if not vocab:
            return

        vecs = {tok: vsa.vec(tok) for tok in vocab}
        clusters = defaultdict(list)
        centers = list(vocab)[:self.n_clusters]
        self.cluster_centers = {i: tok for i, tok in enumerate(centers)}

        for _ in range(3):
            clusters.clear()
            self.token_to_cluster.clear()

            for tok in vocab:
                best_cid, best_sim = 0, -1.0
                v = vecs[tok]
                for cid in range(self.n_clusters):
                    c_tok = self.cluster_centers[cid]
                    sim = float(np.dot(v, vecs[c_tok]))
                    if sim > best_sim:
                        best_sim, best_cid = sim, cid
                clusters[best_cid].append(tok)
                self.token_to_cluster[tok] = best_cid

            for cid in range(self.n_clusters):
                toks = clusters[cid]
                if not toks:
                    continue
                best_center, best_avg = toks[0], -1.0
                for cand in toks:
                    cv = vecs[cand]
                    sims = [float(np.dot(cv, vecs[o])) for o in toks if o != cand]
                    avg_sim = np.mean(sims) if sims else -1.0
                    if avg_sim > best_avg:
                        best_avg, best_center = avg_sim, cand
                self.cluster_centers[cid] = best_center

        self.clusters = {cid: set(toks[:self.cluster_size]) for cid, toks in clusters.items() if toks}
        print(f"✓ Built {len(self.clusters)} knowledge clusters")


    def get_context_clusters(self, ctx):
        if not ctx:
            return set()
        active = set()
        for tok in ctx[-80:]:
            cid = self.token_to_cluster.get(tok)
            if cid is not None:
                active.add(cid)
        return active

    def boost_cluster_neighbors(self, probs: dict, active_clusters: set, boost_strength=0.4) -> dict:
        if not active_clusters or not probs:
            return probs

        boosted = dict(probs)
        cluster_bonus = {}

        for cid in active_clusters:
            cluster_toks = self.clusters.get(cid, set())
            if not cluster_toks:
                cluster_bonus[cid] = 0.0
                continue
            bonus = sum(probs.get(tok, 0.0) for tok in cluster_toks)
            cluster_bonus[cid] = bonus / len(cluster_toks)

        for tok, p in probs.items():
            cid = self.token_to_cluster.get(tok)
            if cid in active_clusters:
                bonus = cluster_bonus.get(cid, 0.0)
                factor = 1.0 + boost_strength * bonus
                boosted[tok] = p * factor

        Z = sum(boosted.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in boosted.items()}


# ---------------------------------------------------------------------
# UNKNOWN CONTEXT PAIR CLUSTERER (user-defined)
# ---------------------------------------------------------------------
class UnknownContextClusterer:
    """
    Tracks rare/unknown bigrams and allows user-defined clustering of them.
    During generation, boosts continuations that belong to active unknown clusters.
    """
    def __init__(self, min_count=3):
        self.min_count = min_count
        self.pair_counts = Counter()          # (prev_tok, tok) -> count
        self.user_clusters = {}               # cluster_id -> set((prev_tok, tok))
        self.pair_to_cluster = {}             # (prev_tok, tok) -> cluster_id
        self.next_cluster_id = 0

    def observe_pair(self, prev_tok, tok, known: bool):
        if not known and prev_tok is not None and tok is not None:
            self.pair_counts[(prev_tok, tok)] += 1

    def get_unknown_pairs(self, threshold=None):
        if threshold is None:
            threshold = self.min_count
        return [
            pair for pair, c in self.pair_counts.items()
            if c >= threshold and pair not in self.pair_to_cluster
        ]

    def create_cluster(self, pairs):
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        self.user_clusters[cid] = set(pairs)
        for p in pairs:
            self.pair_to_cluster[p] = cid
        return cid

    def active_cluster_boost(self, probs: dict, ctx) -> dict:
        if not probs or len(ctx) < 1:
            return probs

        prev_tok = ctx[-1]
        boosted = dict(probs)
        active_cids = set()
        for tok in probs.keys():
            pair = (prev_tok, tok)
            cid = self.pair_to_cluster.get(pair)
            if cid is not None:
                active_cids.add(cid)

        if not active_cids:
            return probs

        cluster_activation = {cid: 0.0 for cid in active_cids}
        for pair, cid in self.pair_to_cluster.items():
            if cid not in active_cids:
                continue
            ptok, ntok = pair
            if ptok == prev_tok and ntok in probs:
                cluster_activation[cid] += probs[ntok]

        max_act = max(cluster_activation.values()) if cluster_activation else 0.0
        if max_act <= 0:
            return probs

        base_strength = 0.3
        for tok, p in probs.items():
            pair = (prev_tok, tok)
            cid = self.pair_to_cluster.get(pair)
            if cid in active_cids:
                act = cluster_activation.get(cid, 0.0) / max_act
                factor = 1.0 + base_strength * act
                boosted[tok] = p * factor

        Z = sum(boosted.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in boosted.items()}


# ---------------------------------------------------------------------
# DOME SPIRAL PROBABILITY MODULATION
# ---------------------------------------------------------------------
class DomeSpiral:
    """
    Dome + spiral geometry that imposes rotating interference structure over the probability simplex.
    """
    def __init__(self, n_spirals=70, dome_height=2.0, decay=0.15):
        self.n_spirals = n_spirals
        self.dome_height = dome_height
        self.decay = decay
        self.token_angles = {}
        self.spiral_phase = 0.0

    def assign_angle(self, tok: str) -> float:
        if tok not in self.token_angles:
            h = (hash(tok) & 0xFFFFFFFF) / 0xFFFFFFFF
            self.token_angles[tok] = h * 2 * np.pi
        return self.token_angles[tok]

    def dome_z(self, r: float) -> float:
        return self.dome_height * (1 - r**2)

    def spiral_weight(self, theta: float, r: float, spiral_idx: int) -> float:
        spiral_base = 2 * np.pi * spiral_idx / self.n_spirals
        spiral_theta = spiral_base + 3.0 * r + self.spiral_phase
        delta = np.arctan2(np.sin(theta - spiral_theta), np.cos(theta - spiral_theta))
        sigma2 = 0.32
        weight = np.exp(-delta**2 / sigma2)
        weight *= (0.5 + 0.5 * self.dome_z(r) / self.dome_height)
        return weight

    def compute_spiral_weights(self, probs: dict, ctx_len: int) -> dict:
        if not probs:
            return {}
        self.spiral_phase = (ctx_len * 0.8) % (2 * np.pi)
        sorted_toks = sorted(probs.items(), key=lambda x: -x[1])
        n = len(sorted_toks)
        weights = {}
        for rank, (tok, p) in enumerate(sorted_toks):
            r = (rank + 1) / (n + 1)
            theta = self.assign_angle(tok)
            spiral_sum = 0.0
            for si in range(self.n_spirals):
                spiral_sum += self.spiral_weight(theta, r, si)
            weights[tok] = (spiral_sum / self.n_spirals) * (self.decay ** r)
        return weights

    def modulate_probs(self, probs: dict, ctx_len: int, blend: float = 0.3) -> dict:
        if not probs:
            return {}
        spiral_w = self.compute_spiral_weights(probs, ctx_len)
        sw_total = sum(spiral_w.values())
        if sw_total > 0:
            spiral_w = {k: v / sw_total for k, v in spiral_w.items()}
        result = {tok: (1 - blend) * p + blend * spiral_w.get(tok, 0.0)
                  for tok, p in probs.items()}
        Z = sum(result.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in result.items()}


# ---------------------------------------------------------------------
# STOCHASTIC CARDINAL ORDERING (SCO)
# ---------------------------------------------------------------------
class StochasticCardinalOrder:
    def __init__(self, inversion_strength=1.7, noise_scale=0.05):
        self.inv_str = inversion_strength
        self.noise_scale = noise_scale
        self.attribution_history = deque(maxlen=500)
        self.cardinal_memory = {}

    def cardinal_rank(self, probs: dict) -> dict:
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        return {
            tok: {
                "original_prob": p,
                "cardinal": i + 1,
                "inverted_weight": (i + 1) ** self.inv_str,
            }
            for i, (tok, p) in enumerate(sorted_items)
        }

    def stochastic_perturb(self, ranks: dict) -> dict:
        perturbed = {}
        for tok, data in ranks.items():
            g = -np.log(-np.log(np.random.uniform(0.001, 0.999)))
            perturbed[tok] = data["inverted_weight"] * (1 + self.noise_scale * g)
        return perturbed

    def attribution_transform(self, probs: dict, ctx_attribution: float = 0.5) -> dict:
        if not probs:
            return {}
        ranks = self.cardinal_rank(probs)
        perturbed = self.stochastic_perturb(ranks)
        inv_factor = 0.3 + 0.7 * ctx_attribution
        pert_sum = sum(perturbed.values()) or 1.0

        final_scores = {}
        for tok, pert_weight in perturbed.items():
            orig_p = ranks[tok]["original_prob"]
            blended = (1 - inv_factor) * orig_p + inv_factor * (pert_weight / pert_sum)
            final_scores[tok] = blended
            self.cardinal_memory[tok] = self.cardinal_memory.get(tok, 0) + ranks[tok]["cardinal"]

        Z = sum(final_scores.values())
        if Z <= 0.1:
            return probs
        return {k: v / Z for k, v in final_scores.items()}

    def get_context_attribution(self, ctx: tuple) -> float:
        if not self.attribution_history:
            return 0.5
        matches = sum(1 for h in self.attribution_history if h.get("ctx") == ctx)
        return 1 - np.exp(-matches / 10000.0)

    def record_attribution(self, ctx: tuple, chosen_tok: str, final_prob: float):
        self.attribution_history.append(
            {
                "ctx": ctx,
                "tok": chosen_tok,
                "prob": final_prob,
                "cardinal": self.cardinal_memory.get(chosen_tok, 0),
            }
        )


# ---------------------------------------------------------------------
# ACTIVE MEMORY
# ---------------------------------------------------------------------
class MemoryType(Enum):
    DATAPOINT = "raw"
    RULE = "rule"


@dataclass
class MemoryNode:
    id: str
    type: MemoryType
    content: dict


class ActiveMemory:
    def __init__(self):
        self.nodes = {}
        self.dp_count = 0
        self.rule_count = 0
        self.rules = defaultdict(set)

    def record(self, ctx, tok, p, state, step):
        nid = f"dp{self.dp_count}"
        self.dp_count += 1
        self.nodes[nid] = MemoryNode(
            nid, MemoryType.DATAPOINT, {"ctx": ctx, "tok": tok, "p": p}
        )
        sig = ("low", ctx)
        if sig not in self.nodes:
            self.nodes[sig] = set()
        self.nodes[sig].add(nid)
        if len(self.nodes[sig]) >= 3:
            rid = f"r{self.rule_count}_lowboost"
            self.rule_count += 1
            self.rules["lowboost"].add(rid)
            self.nodes[rid] = MemoryNode(
                rid, MemoryType.RULE, {"sig": sig}
            )

    def infer_boost(self, ctx, minp):
        return len(self.rules["lowboost"]) > 0 and minp < 0.45


# ---------------------------------------------------------------------
# VSA
# ---------------------------------------------------------------------
class VSA:
    def __init__(self, d=512):
        self.d = d
        self.book = {}

    def vec(self, sym):
        if sym not in self.book:
            th = np.random.uniform(0, 2 * np.pi, self.d // 2)
            v = np.hstack([np.cos(th), np.sin(th)])
            self.book[sym] = v / (np.linalg.norm(v) + 1e-8)
        return self.book[sym]


# ---------------------------------------------------------------------
# CONCENTRIC ENCODER (full pipeline)
# ---------------------------------------------------------------------
def batch_proc(batch):
    uni = Counter()
    rings = [defaultdict(Counter) for _ in range(5)]
    for seq in batch:
        for i, t in enumerate(seq):
            uni[t] += 1
            if i > 0:
                pr = seq[i - 1]
                pa = (i % 360) * np.pi / 3000.0
                for r in range(5):
                    rings[r][pr][t] += 1 + 0.4 * np.cos(2 * np.pi * r / 5 - pa)
    return uni, rings


class ConcEnc:
    def __init__(self):
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.uni = Counter()
        self.lpstate = defaultdict(lambda: {"act": False, "cnt": 0})
        self.sco = StochasticCardinalOrder(inversion_strength=10.65, noise_scale=0.12)
        self.dome = DomeSpiral(n_spirals=70, dome_height=12.0, decay=0.15)
        self.ctx_index = {}
        self.ctx_index_norm = {}
        self.knowledge = KnowledgeSubsets(n_clusters=81, cluster_size=50000)
        self.unknown_ctx = UnknownContextClusterer(min_count=3)
        self.intent_alpha = 10.55   # how hard intent steers the probs
        self.intent_temp  = 1.0    # temperature on intent scores
        # permutation activation hyperparams
        self.perm_K = 5
        self.perm_noise = 0.12



    # ---------------- INTENT FROM DATABASE ----------------
    def compute_intent_scores(self, ctx, base_probs: dict) -> dict:
        """
        Build an 'intent score' for each token from all database-like
        structures: rings, self-context index, knowledge clusters,
        unknown context clusters, and unigram frequencies.
        Returns a dict token -> raw intent score (not normalized).
        """
        if not base_probs:
            return {}

        tokens = list(base_probs.keys())
        scores = {t: 0.0 for t in tokens}

        # 1) Local transition intent from rings (successors of last token)
        if ctx:
            last = ctx[-1]
            for ri, ring in enumerate(self.rings):
                if last in ring:
                    row = ring[last]
                    tt = sum(row.values())
                    if tt > 0:
                        w_ring = 1.0 + ri / 5.0
                        for t in tokens:
                            c = row.get(t, 0)
                            if c > 0:
                                scores[t] += w_ring * (c / tt)

        # 2) Self-context similarity from ctx_index_norm
        cur_sig_ctr = self.current_context_signature(ctx, window=64)
        cur_sig = dict(cur_sig_ctr)
        if cur_sig:
            for t in tokens:
                neigh = self.ctx_index_norm.get(t)
                if not neigh:
                    continue
                s = 0.0
                for w, nw in neigh.items():
                    cw = cur_sig.get(w)
                    if cw is not None:
                        s += cw * nw
                scores[t] += 0.9 * s

        # 3) Knowledge subset activation over last 80 tokens
        active_clusters = self.knowledge.get_context_clusters(ctx)
        if active_clusters:
            # precompute cluster average score from base_probs
            cluster_base = {}
            for cid in active_clusters:
                toks = self.knowledge.clusters.get(cid, set())
                if not toks:
                    cluster_base[cid] = 0.0
                    continue
                cluster_base[cid] = sum(base_probs.get(x, 0.0) for x in toks) / len(toks)
            for t in tokens:
                cid = self.knowledge.token_to_cluster.get(t)
                if cid in active_clusters:
                    scores[t] += 0.7 * cluster_base.get(cid, 0.0)

        # 4) Unknown context clusters conditioned on last token
        if ctx:
            prev_tok = ctx[-1]
            for t in tokens:
                pair = (prev_tok, t)
                cid = self.unknown_ctx.pair_to_cluster.get(pair)
                if cid is not None:
                    # weight by how active that cluster is under base_probs
                    act = 0.0
                    for (pt, nt), cid2 in self.unknown_ctx.pair_to_cluster.items():
                        if cid2 == cid and pt == prev_tok and nt in base_probs:
                            act += base_probs[nt]
                    scores[t] += 0.8 * act

        # 5) Global unigram frequency as weak prior
        total_uni = sum(self.uni.values())
        if total_uni > 0:
            for t in tokens:
                f = self.uni[t] / total_uni
                # light bias toward tokens well represented in corpus
                scores[t] += 0.15 * np.sqrt(f + 1e-12)

        return scores

    def apply_intent(self, base_probs: dict, intent_scores: dict) -> dict:
        """
        Turn intent scores into a multiplicative modulation over probs.
        """
        if not base_probs or not intent_scores:
            return base_probs

        tokens = list(base_probs.keys())
        p = np.array([base_probs[t] for t in tokens], dtype=np.float64)

        s = np.array([intent_scores.get(t, 0.0) for t in tokens], dtype=np.float64)
        # avoid degenerate all-zero
        if np.all(s == 0):
            return base_probs

        # normalize intent scores and turn into exponent
        s = (s - s.mean()) / (s.std() + 1e-8)
        logm = self.intent_alpha * s / max(self.intent_temp, 1e-6)
        m = np.exp(logm)

        p_mod = p * m
        Z = p_mod.sum()
        if Z <= 0:
            return base_probs
        p_mod /= Z
        return {tok: float(v) for tok, v in zip(tokens, p_mod)}

    def train(self, corp, vsa):
        # Phase 1: rings
        batches = [corp[i : i + 4000] for i in range(0, len(corp), 4000)]
        with ProcessPoolExecutor(2) as ex:
            res = list(
                tqdm(
                    ex.map(batch_proc, batches),
                    total=len(batches),
                    desc="Training rings",
                )
            )
        for u, rs in res:
            self.uni.update(u)
            for ri, rdata in enumerate(rs):
                for pr, ts in rdata.items():
                    self.rings[ri][pr].update(ts)

        # Phase 2: self-context index
        print("Building self-context index...")
        ctx_idx = defaultdict(Counter)
        for seq in tqdm(corp, desc="Context indexing"):
            L = len(seq)
            for i, tok in enumerate(seq):
                for j in range(max(0, i - 8), min(L, i + 9)):
                    if j == i:
                        continue
                    ctx_idx[tok][seq[j]] += 1
        self.ctx_index = dict(ctx_idx)
        self.ctx_index_norm = {}
        for tok, ctr in ctx_idx.items():
            tot = sum(ctr.values())
            self.ctx_index_norm[tok] = {k: v / tot for k, v in ctr.items()} if tot > 0 else {}
        print(f"✓ Indexed {len(self.ctx_index_norm)} tokens' contexts")

        # Phase 3: knowledge subsets
        vocab = list(self.uni.keys())
        self.knowledge.build_clusters(vocab, vsa)

    def is_known_pair(self, prev_tok, next_tok):
        for ring in self.rings:
            if prev_tok in ring and next_tok in ring[prev_tok]:
                return True
        return False

    def current_context_signature(self, ctx, window=32):
        if not ctx:
            return Counter()
        sig = Counter(ctx[-window:])
        tot = sum(sig.values())
        if tot > 0:
            for k in sig:
                sig[k] /= tot
        return sig

    def self_context_boost(self, base_probs: dict, ctx):
        if not base_probs:
            return base_probs
        cur_sig_ctr = self.current_context_signature(ctx)
        if not cur_sig_ctr:
            return base_probs
        cur_sig = dict(cur_sig_ctr)

        scores = {}
        for tok, p in base_probs.items():
            neigh = self.ctx_index_norm.get(tok)
            if not neigh:
                scores[tok] = 0.0
                continue
            s = 0.0
            for w, nw in neigh.items():
                cw = cur_sig.get(w)
                if cw is not None:
                    s += cw * nw
            scores[tok] = s

        max_s = max(scores.values()) if scores else 0.0
        if max_s <= 0:
            return base_probs

        boosted = {}
        for tok, p in base_probs.items():
            factor = 1.0 + 0.6 * (scores[tok] / max_s)
            boosted[tok] = p * factor

        Z = sum(boosted.values())
        if Z <= 0:
            return base_probs
        return {k: v / Z for k, v in boosted.items()}

    def probs_raw(self, ctx):
        if not ctx:
            t = sum(self.uni.values())
            return {w: c / t for w, c in self.uni.items()} if t else {}
        last = ctx[-1]
        ag = Counter()
        for ri, ring in enumerate(self.rings):
            if last in ring:
                row = ring[last]
                tt = sum(row.values())
                if tt <= 0:
                    continue
                for nt, c in row.items():
                    ag[nt] += (c / tt) * (1 + ri / 5.0)
        tt = sum(ag.values())
        return {k: v / tt for k, v in ag.items()} if tt else self.probs_raw([])

    # -----------------------------------------------------------------
    # PERMUTATION ACTIVATION GENERATION
    # -----------------------------------------------------------------
     # -----------------------------------------------------------------
    # PERMUTATION ACTIVATION GENERATION (FREQ-BASED)
    # -----------------------------------------------------------------
    def perm_activation(self, probs: dict) -> dict:
        """
        Frequency-based permutation activations:
        - treat probs as a freq vector
        - apply K random monotone transforms in prob-space (no index shuffle)
        - each transform acts like a 'permutation' over the simplex geometry
        - average them and renormalize
        """
        if not probs:
            return probs

        tokens = list(probs.keys())
        base = np.array([probs[t] for t in tokens], dtype=np.float64)
        base = np.maximum(base, 1e-12)
        base /= base.sum()

        def one_pass(p):
            # random temperature in [0.7, 1.3]
            temp = np.exp(np.random.uniform(np.log(10.7), np.log(100.3)))
            # random exponent in [0.7, 1.4]
            alpha = np.random.uniform(0.1, 1.4)

            # Gumbel noise in log-space, scaled by perm_noise
            g = -np.log(-np.log(np.random.uniform(0.001, 0.999, size=p.shape)))
            logp = np.log(p) / temp + self.perm_noise * g

            # monotone power transform in prob-space
            p_tilde = np.exp(logp - logp.max())
            p_tilde = p_tilde ** alpha

            Z = p_tilde.sum()
            if Z <= 0:
                return p
            return p_tilde / Z

        acc = np.zeros_like(base)
        for _ in range(self.perm_K):
            acc += one_pass(base)

        acc /= max(1, self.perm_K)
        Z = acc.sum()
        if Z <= 0:
            return probs
        acc /= Z
        return {tok: float(p) for tok, p in zip(tokens, acc)}

    def probs(self, ctx):
        raw = self.probs_raw(ctx)
        if not raw:
            return raw

        # 1) frequency-based permutation activations
        perm_probs = self.perm_activation(raw)

        # 2) global intent from whole database, applied multiplicatively
        intent_scores = self.compute_intent_scores(ctx, perm_probs)
        intent_probs = self.apply_intent(perm_probs, intent_scores)

        # 3) dome spiral over simplex geometry
        spiral_probs = self.dome.modulate_probs(intent_probs, len(ctx), blend=0.25)

        # 4) knowledge subsets (local boost over already intent-shaped probs)
        active_clusters = self.knowledge.get_context_clusters(ctx)
        knowledge_probs = self.knowledge.boost_cluster_neighbors(
            spiral_probs, active_clusters, boost_strength=0.4
        )

        # 5) self-context refinement
        sc_probs = self.self_context_boost(knowledge_probs, ctx)

        # 6) unknown context clusters
        uctx_probs = self.unknown_ctx.active_cluster_boost(sc_probs, ctx)

        # 7) SCO as final stochastic cardinal reordering
        ctx_tuple = tuple(ctx[-2:]) if len(ctx) >= 2 else tuple(ctx)
        attr = self.sco.get_context_attribution(ctx_tuple)
        return self.sco.attribution_transform(uctx_probs, attr)


    def lp_bias(self, pr, ctx, st, sp):
        stt = self.lpstate[ctx]
        if sp < 0.42 and not stt["act"]:
            stt["act"], stt["cnt"] = True, 0
        elif stt["act"] and stt["cnt"] < 150:
            pr[st] = pr.get(st, 0.0) * 10
            stt["cnt"] += 1
        elif stt["cnt"] >= 150:
            stt["act"] = False
        tt = sum(pr.values())
        return {k: v / tt for k, v in pr.items()} if tt else pr


# ---------------------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------------------
class ConcGen:
    def __init__(self):
        self.vsa = VSA()
        self.enc = ConcEnc()
        self.mem = ActiveMemory()
        self.steps = 0

    def fit(self, file):
        with open(file, encoding="UTF-8") as f:
            txt = f.read().lower()[:KB_len]
        snts = [s.split() for s in txt.split(".") if s.strip()]

        # Pre-build VSA for vocab so clustering uses same vectors
        for s in snts:
            for t in s:
                if len(t) % 4 == 0:
                    continue
                else:
                    self.vsa.vec(t)
        self.enc.train(snts, self.vsa)
        print("TRAINED ✓ (SCO + Dome + Self-Context + Knowledge + Unknown Ctx + PermActs)")

    def gen(self, seed, n=600, t=0.95, show_progress=False, stream=True):
        ctx = list(seed)
        res = []

        iterator = range(n)

        for step in iterator:
            ckey = tuple(ctx[-1:]) if len(ctx) >= 2 else tuple(ctx)
            ps = self.enc.probs(ctx)
            if not ps:
                print(f"\n[!] Stopped at step {step}: no continuations")
                break

            if self.mem.infer_boost(ckey, min(ps.values())):
                ps = {k: v * 1.3 for k, v in ps.items()}
                Zb = sum(ps.values())
                if Zb > 0:
                    ps = {k: v / Zb for k, v in ps.items()}

            pv = np.log(np.array(list(ps.values())) + 1e-10) / t
            pv = np.exp(pv - np.max(pv))
            pv /= pv.sum()

            keys = list(ps.keys())

            for i in range(len(pv)):
                nt = np.random.choice(keys, p=pv)
                sp = ps[nt]
                # unknown-context observation
                prev_tok_for_pair = ctx[-1] if ctx else None
                if prev_tok_for_pair is not None:
                    known = self.enc.is_known_pair(prev_tok_for_pair, nt)
                    self.enc.unknown_ctx.observe_pair(prev_tok_for_pair, nt, known)
                    ctx.append(known)
                if self.mem.infer_boost(ckey, min(ps.values())):
                    ps = {k: v * 1.3 for k, v in ps.items()}
                    Zb = sum(ps.values())
                    if Zb > 0:
                        ps = {k: v / Zb for k, v in ps.items()}

            self.enc.sco.record_attribution(ckey, nt, ps.items())
            self.mem.record(ckey, nt, ps.items(), self.enc.lpstate[ckey], self.steps)
            ps = self.enc.lp_bias(ps, ckey, nt, sp)

            res.append(nt)
            if stream:
                print(nt, end=" ", flush=True)
            ctx.append(nt)
            self.steps += 1

        if stream:
            print()
        return res


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")
    g = ConcGen()
    fn = input("Corpus file: ")
    g.fit(fn)
    while True:
        cmd = input("USER: ").strip()

        # treat command as seed string
        sd = cmd.split()
        out = g.gen(sd, stream=True)
        print(f"STATS: {g.mem.dp_count} dps, {g.mem.rule_count} rules, {g.steps} steps")
        print(
            f"SCO: {len(g.enc.sco.cardinal_memory)} tokens, "
            f"{len(g.enc.sco.attribution_history)} attrs"
        )
        print(
            f"DOME: {len(g.enc.dome.token_angles)} mapped, "
            f"phase={g.enc.dome.spiral_phase:.3f}"
        )
        print(f"CTX: {len(g.enc.ctx_index_norm)} tokens indexed")
        print(f"KNOWLEDGE: {len(g.enc.knowledge.clusters)} clusters")
        print(f"UNKNOWN CTX CLUSTERS: {len(g.enc.unknown_ctx.user_clusters)}")

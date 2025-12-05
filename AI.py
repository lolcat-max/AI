# =====================================================================
# COUNTER-INTUITIVE STOCHASTIC CARDINAL ORDERING ON ATTRIBUTION
# =====================================================================

import numpy as np
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------
# STOCHASTIC CARDINAL ORDERING (SCO)
# Counter-intuitive: inverts natural ordering to favor unlikely events
# ---------------------------------------------------------------------
class StochasticCardinalOrder:
    """
    Implements counter-intuitive ordering where:
    - Cardinal rank is INVERTED (low prob -> high weight)
    - Stochastic noise perturbs the ordering
    - Attribution scores are transformed non-monotonically
    """
    def __init__(self, inversion_strength=0.7, noise_scale=0.15):
        self.inv_str = inversion_strength      # How much to invert rankings
        self.noise_scale = noise_scale         # Stochastic perturbation
        self.attribution_history = deque(maxlen=500)
        self.cardinal_memory = {}              # Token -> cumulative cardinal score
        
    def cardinal_rank(self, probs: dict) -> dict:
        """
        Assign cardinal ranks (1 = highest prob, N = lowest)
        Then INVERT: low cardinal rank gets high weight
        """
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        n = len(sorted_items)
        ranks = {}
        for i, (tok, p) in enumerate(sorted_items):
            cardinal = i + 1  # 1-indexed rank
            # Counter-intuitive inversion: rank N becomes weight N, rank 1 becomes weight 1
            # So rare items (high cardinal rank) get boosted
            inverted_weight = cardinal ** self.inv_str
            ranks[tok] = {
                'original_prob': p,
                'cardinal': cardinal,
                'inverted_weight': inverted_weight
            }
        return ranks
    
    def stochastic_perturb(self, ranks: dict) -> dict:
        """
        Add stochastic noise to break deterministic ordering
        Uses Gumbel noise for proper stochastic ordering
        """
        perturbed = {}
        for tok, data in ranks.items():
            # Gumbel noise for stochastic ordering (like Gumbel-softmax)
            gumbel = -np.log(-np.log(np.random.uniform(0.001, 0.999)))
            noise = gumbel * self.noise_scale
            
            # Perturb the inverted weight
            perturbed[tok] = data['inverted_weight'] * (1 + noise)
        return perturbed
    
    def attribution_transform(self, probs: dict, ctx_attribution: float = 0.5) -> dict:
        """
        Transform probabilities using counter-intuitive SCO:
        1. Compute cardinal ranks
        2. Invert rankings (rare = high weight)
        3. Perturb stochastically
        4. Modulate by attribution score
        
        ctx_attribution: how "attributed" this context is (0=novel, 1=seen often)
        """
        if not probs:
            return {}
        
        # Step 1 & 2: Cardinal ranking with inversion
        ranks = self.cardinal_rank(probs)
        
        # Step 3: Stochastic perturbation
        perturbed = self.stochastic_perturb(ranks)
        
        # Step 4: Attribution modulation
        # Counter-intuitive: HIGH attribution -> MORE inversion (explore known contexts)
        #                    LOW attribution -> LESS inversion (exploit novel contexts)
        inv_factor = 0.3 + 0.7 * ctx_attribution  # Range [0.3, 1.0]
        
        final_scores = {}
        for tok, pert_weight in perturbed.items():
            orig_p = ranks[tok]['original_prob']
            
            # Blend: interpolate between original and inverted based on attribution
            # High attribution -> trust inversion more (counter-intuitive behavior)
            blended = (1 - inv_factor) * orig_p + inv_factor * (pert_weight / sum(perturbed.values()))
            final_scores[tok] = blended
            
            # Update cardinal memory for this token
            if tok not in self.cardinal_memory:
                self.cardinal_memory[tok] = 0
            self.cardinal_memory[tok] += ranks[tok]['cardinal']
        
        # Normalize
        total = sum(final_scores.values())
        return {k: v/total for k, v in final_scores.items()}
    
    def get_context_attribution(self, ctx: tuple) -> float:
        """
        Compute attribution score for context based on history
        Returns 0-1 (0=never seen, 1=frequently seen)
        """
        if not self.attribution_history:
            return 0.5
        
        # Count how often this context appears in history
        matches = sum(1 for h in self.attribution_history if h.get('ctx') == ctx)
        # Sigmoid-like saturation
        return 1 - np.exp(-matches / 10)
    
    def record_attribution(self, ctx: tuple, chosen_tok: str, final_prob: float):
        """Record choice for attribution tracking"""
        self.attribution_history.append({
            'ctx': ctx,
            'tok': chosen_tok,
            'prob': final_prob,
            'cardinal': self.cardinal_memory.get(chosen_tok, 0)
        })


# ---------------------------------------------------------------------
# ACTIVE MEMORY (unchanged)
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
        self.nodes[nid] = MemoryNode(nid, MemoryType.DATAPOINT, {"ctx": ctx, "tok": tok, "p": p})
        if p < 0.45:
            sig = ("low", ctx)
            if sig not in self.nodes:
                self.nodes[sig] = set()
            self.nodes[sig].add(nid)
            if len(self.nodes[sig]) >= 3:
                rid = f"r{self.rule_count}_lowboost"
                self.rule_count += 1
                self.rules["lowboost"].add(rid)
                self.nodes[rid] = MemoryNode(rid, MemoryType.RULE, {"sig": sig})

    def infer_boost(self, ctx, minp):
        return len(self.rules["lowboost"]) > 0 and minp < 0.45


# ---------------------------------------------------------------------
# VSA (unchanged)
# ---------------------------------------------------------------------
class VSA:
    def __init__(self, d=256):
        self.d = d
        self.book = {}

    def vec(self, sym):
        if sym not in self.book:
            th = np.random.uniform(0, 2*np.pi, self.d//2)
            v = np.hstack([np.cos(th), np.sin(th)])
            self.book[sym] = v / (np.linalg.norm(v) + 1e-8)
        return self.book[sym]


# ---------------------------------------------------------------------
# CONCENTRIC ENCODER (modified to integrate SCO)
# ---------------------------------------------------------------------
def batch_proc(batch):
    uni = Counter()
    rings = [defaultdict(Counter) for _ in range(5)]
    for seq in batch:
        for i, t in enumerate(seq):
            uni[t] += 1
            if i > 0:
                pr = seq[i-1]
                pa = (i % 360) * np.pi / 180
                for r in range(5):
                    ra = 2 * np.pi * r / 5
                    s = np.cos(ra - pa)
                    w = 1 + 0.4 * s
                    rings[r][pr][t] += w
    return uni, rings

class ConcEnc:
    def __init__(self):
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.uni = Counter()
        self.lpstate = defaultdict(lambda: {"act": False, "cnt": 0})
        self.sco = StochasticCardinalOrder(inversion_strength=0.65, noise_scale=0.12)

    def train(self, corp):
        batches = [corp[i:i+400] for i in range(0, len(corp), 400)]
        with ProcessPoolExecutor(2) as ex:
            res = list(tqdm(ex.map(batch_proc, batches), total=len(batches)))
        for u, rs in res:
            self.uni.update(u)
            for ri, rdata in enumerate(rs):
                for pr, ts in rdata.items():
                    self.rings[ri][pr].update(ts)

    def probs_raw(self, ctx):
        """Get raw probabilities before SCO transformation"""
        if not ctx: 
            t = sum(self.uni.values())
            return {w: c/t for w, c in self.uni.items()}
        last = ctx[-1]
        ag = Counter()
        for ri, ring in enumerate(self.rings):
            if last in ring:
                tt = sum(ring[last].values())
                rw = 1 + ri/5
                for nt, c in ring[last].items():
                    ag[nt] += (c/tt) * rw
        tt = sum(ag.values())
        return {k: v/tt for k, v in ag.items()} if tt else self.probs_raw([])

    def probs(self, ctx):
        """Get probabilities with counter-intuitive SCO applied"""
        raw = self.probs_raw(ctx)
        if not raw:
            return raw
        
        # Compute context attribution
        ctx_tuple = tuple(ctx[-2:]) if len(ctx) >= 2 else tuple(ctx)
        attr_score = self.sco.get_context_attribution(ctx_tuple)
        
        # Apply counter-intuitive transformation
        return self.sco.attribution_transform(raw, attr_score)

    def lp_bias(self, pr, ctx, st, sp):
        stt = self.lpstate[ctx]
        if sp < 0.42 and not stt["act"]:
            stt["act"] = True
            stt["cnt"] = 0
        elif stt["act"] and stt["cnt"] < 150:
            pr[st] = pr.get(st, 0) * 10
            stt["cnt"] += 1
        elif stt["cnt"] >= 150:
            stt["act"] = False
        tt = sum(pr.values())
        return {k: v/tt for k, v in pr.items()} if tt else pr


# ---------------------------------------------------------------------
# GENERATOR (with SCO integration)
# ---------------------------------------------------------------------
class ConcGen:
    def __init__(self):
        self.vsa = VSA()
        self.enc = ConcEnc()
        self.mem = ActiveMemory()
        self.steps = 0

    def fit(self, file):
        with open(file, encoding="UTF-8") as f:
            txt = f.read()
        snts = [s.split() for s in txt.split('.') if s.strip()]
        self.enc.train(snts)
        for s in snts:
            for t in s: 
                self.vsa.vec(t)
        print("TRAINED ✓ (with Counter-Intuitive SCO)")

    def gen(self, seed, n=600, t=0.95, show_progress=True):
        """
        Generate text with counter-intuitive SCO sampling.
        
        Args:
            seed: List of seed tokens to start generation
            n: Number of tokens to generate
            t: Temperature for sampling (higher = more random)
            show_progress: Whether to show tqdm progress bar
        
        Returns:
            List of generated tokens
        """
        ctx = list(seed)
        res = []
        
        # Create iterator with optional tqdm progress bar
        iterator = range(n)
        if show_progress:
            iterator = tqdm(
                iterator, 
                desc="Generating", 
                unit="tok",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} tok [{elapsed}<{remaining}, {rate_fmt}]',
                dynamic_ncols=True
            )
        
        for step in iterator:
            ckey = tuple(ctx[-2:]) if len(ctx) >= 2 else tuple(ctx)
            
            # Get SCO-transformed probabilities
            ps = self.enc.probs(ctx)
            
            if not ps:
                if show_progress and hasattr(iterator, 'close'):
                    iterator.close()
                print(f"\n[!] Stopped early at step {step}: no valid continuations")
                break
            
            # Check for memory-based boost
            bp = self.mem.infer_boost(ckey, min(ps.values()))
            if bp: 
                ps = {k: v * 1.3 for k, v in ps.items()}
                total = sum(ps.values())
                ps = {k: v/total for k, v in ps.items()}
            
            # Temperature sampling with numerical stability
            pv = np.log(np.array(list(ps.values())) + 1e-10) / t
            pv = np.exp(pv - np.max(pv))  # Subtract max for stability
            pv /= pv.sum()
            
            # Sample next token
            nt = np.random.choice(list(ps.keys()), p=pv)
            sp = ps[nt]
            
            # Record attribution for SCO tracking
            self.enc.sco.record_attribution(ckey, nt, sp)
            
            # Record to active memory
            stt = self.enc.lpstate[ckey]
            self.mem.record(ckey, nt, sp, stt, self.steps)
            
            # Apply low-probability bias
            ps = self.enc.lp_bias(ps, ckey, nt, sp)
            
            # Append result and update context
            res.append(nt)
            ctx.append(nt)
            self.steps += 1
            
            # Update progress bar with stats
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix(
                    tok=nt[:10] if len(nt) > 10 else nt,
                    p=f"{sp:.3f}",
                    attr=len(self.enc.sco.attribution_history),
                    refresh=True
                )
        
        # Print summary after generation
        if show_progress:
            print(f"\n[✓] Generated {len(res)} tokens")
        
        return res



if __name__ == "__main__":
    mp.set_start_method("spawn")
    g = ConcGen()
    fn = input("Corpus file: ")
    g.fit(fn)
    while True:
        sd = input("Seed: ").split()
        out = g.gen(sd)
        print("OUT:", " ".join(out))
        print(f"STATS: {g.mem.dp_count} dps, {g.mem.rule_count} rules, {g.steps} steps")
        print(f"SCO: {len(g.enc.sco.cardinal_memory)} tokens tracked, "
              f"{len(g.enc.sco.attribution_history)} attributions")

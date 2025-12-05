# =====================================================================
# COUNTER-INTUITIVE STOCHASTIC CARDINAL ORDERING ON ATTRIBUTION
# WITH DOME SPIRAL PROBABILITY MODULATION
# =====================================================================
KB_len = 99999
import numpy as np
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------
# DOME SPIRAL PROBABILITY MODULATION
# ---------------------------------------------------------------------
class DomeSpiral:
    def __init__(self, n_spirals=7, dome_height=1.0, decay=0.85):
        self.n_spirals = n_spirals
        self.dome_height = dome_height
        self.decay = decay
        self.token_angles = {}
        self.spiral_phase = 0.0
        
    def assign_angle(self, tok: str) -> float:
        if tok not in self.token_angles:
            h = hash(tok) & 0xFFFFFFFF
            self.token_angles[tok] = (h / 0xFFFFFFFF) * 2 * np.pi
        return self.token_angles[tok]
    
    def dome_z(self, r: float) -> float:
        return self.dome_height * (1 - r**2)
    
    def spiral_weight(self, theta: float, r: float, spiral_idx: int) -> float:
        spiral_base = 2 * np.pi * spiral_idx / self.n_spirals
        alpha = 3.0
        spiral_theta = spiral_base + alpha * r + self.spiral_phase
        delta = np.arctan2(np.sin(theta - spiral_theta), np.cos(theta - spiral_theta))
        sigma = 0.4
        weight = np.exp(-delta**2 / (2 * sigma**2))
        weight *= (0.5 + 0.5 * self.dome_z(r) / self.dome_height)
        return weight
    
    def compute_spiral_weights(self, probs: dict, ctx_len: int) -> dict:
        if not probs: return {}
        self.spiral_phase = (ctx_len * 0.1) % (2 * np.pi)
        sorted_toks = sorted(probs.items(), key=lambda x: -x[1])
        n = len(sorted_toks)
        weights = {}
        for rank, (tok, p) in enumerate(sorted_toks):
            r = (rank + 1) / (n + 1)
            theta = self.assign_angle(tok)
            spiral_sum = sum(self.spiral_weight(theta, r, si) for si in range(self.n_spirals))
            weights[tok] = (spiral_sum / self.n_spirals) * (self.decay ** r)
        return weights
    
    def modulate_probs(self, probs: dict, ctx_len: int, blend: float = 0.3) -> dict:
        if not probs: return {}
        spiral_w = self.compute_spiral_weights(probs, ctx_len)
        sw_total = sum(spiral_w.values())
        if sw_total > 0: spiral_w = {k: v/sw_total for k, v in spiral_w.items()}
        result = {tok: (1 - blend) * p + blend * spiral_w.get(tok, 0) for tok, p in probs.items()}
        total = sum(result.values())
        return {k: v/total for k, v in result.items()}

# ---------------------------------------------------------------------
# STOCHASTIC CARDINAL ORDERING (SCO)
# ---------------------------------------------------------------------
class StochasticCardinalOrder:
    def __init__(self, inversion_strength=0.7, noise_scale=0.15):
        self.inv_str = inversion_strength
        self.noise_scale = noise_scale
        self.attribution_history = deque(maxlen=500)
        self.cardinal_memory = {}
        
    def cardinal_rank(self, probs: dict) -> dict:
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        return {tok: {'original_prob': p, 'cardinal': i+1, 'inverted_weight': (i+1)**self.inv_str}
                for i, (tok, p) in enumerate(sorted_items)}
    
    def stochastic_perturb(self, ranks: dict) -> dict:
        return {tok: data['inverted_weight'] * (1 + self.noise_scale * -np.log(-np.log(np.random.uniform(0.001, 0.999))))
                for tok, data in ranks.items()}
    
    def attribution_transform(self, probs: dict, ctx_attribution: float = 0.5) -> dict:
        if not probs: return {}
        ranks = self.cardinal_rank(probs)
        perturbed = self.stochastic_perturb(ranks)
        inv_factor = 0.3 + 0.7 * ctx_attribution
        pert_sum = sum(perturbed.values())
        final_scores = {}
        for tok, pert_weight in perturbed.items():
            final_scores[tok] = (1 - inv_factor) * ranks[tok]['original_prob'] + inv_factor * (pert_weight / pert_sum)
            self.cardinal_memory[tok] = self.cardinal_memory.get(tok, 0) + ranks[tok]['cardinal']
        total = sum(final_scores.values())
        return {k: v/total for k, v in final_scores.items()}
    
    def get_context_attribution(self, ctx: tuple) -> float:
        if not self.attribution_history: return 0.5
        return 1 - np.exp(-sum(1 for h in self.attribution_history if h.get('ctx') == ctx) / 10)
    
    def record_attribution(self, ctx: tuple, chosen_tok: str, final_prob: float):
        self.attribution_history.append({'ctx': ctx, 'tok': chosen_tok, 'prob': final_prob,
                                         'cardinal': self.cardinal_memory.get(chosen_tok, 0)})

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
        self.nodes[nid] = MemoryNode(nid, MemoryType.DATAPOINT, {"ctx": ctx, "tok": tok, "p": p})
        if p < 0.45:
            sig = ("low", ctx)
            if sig not in self.nodes: self.nodes[sig] = set()
            self.nodes[sig].add(nid)
            if len(self.nodes[sig]) >= 3:
                rid = f"r{self.rule_count}_lowboost"
                self.rule_count += 1
                self.rules["lowboost"].add(rid)
                self.nodes[rid] = MemoryNode(rid, MemoryType.RULE, {"sig": sig})

    def infer_boost(self, ctx, minp):
        return len(self.rules["lowboost"]) > 0 and minp < 0.45

# ---------------------------------------------------------------------
# VSA
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
# CONCENTRIC ENCODER (with SCO + Dome Spiral)
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
                    rings[r][pr][t] += 1 + 0.4 * np.cos(2 * np.pi * r / 5 - pa)
    return uni, rings

class ConcEnc:
    def __init__(self):
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.uni = Counter()
        self.lpstate = defaultdict(lambda: {"act": False, "cnt": 0})
        self.sco = StochasticCardinalOrder(inversion_strength=0.65, noise_scale=0.12)
        self.dome = DomeSpiral(n_spirals=7, dome_height=1.0, decay=0.85)

    def train(self, corp):
        batches = [corp[i:i+400] for i in range(0, len(corp), 400)]
        with ProcessPoolExecutor(2) as ex:
            res = list(tqdm(ex.map(batch_proc, batches), total=len(batches), desc="Training"))
        for u, rs in res:
            self.uni.update(u)
            for ri, rdata in enumerate(rs):
                for pr, ts in rdata.items():
                    self.rings[ri][pr].update(ts)

    def probs_raw(self, ctx):
        if not ctx: 
            t = sum(self.uni.values())
            return {w: c/t for w, c in self.uni.items()}
        last = ctx[-1]
        ag = Counter()
        for ri, ring in enumerate(self.rings):
            if last in ring:
                tt = sum(ring[last].values())
                for nt, c in ring[last].items():
                    ag[nt] += (c/tt) * (1 + ri/5)
        tt = sum(ag.values())
        return {k: v/tt for k, v in ag.items()} if tt else self.probs_raw([])

    def probs(self, ctx):
        raw = self.probs_raw(ctx)
        if not raw: return raw
        spiral_probs = self.dome.modulate_probs(raw, len(ctx), blend=0.25)
        ctx_tuple = tuple(ctx[-2:]) if len(ctx) >= 2 else tuple(ctx)
        return self.sco.attribution_transform(spiral_probs, self.sco.get_context_attribution(ctx_tuple))

    def lp_bias(self, pr, ctx, st, sp):
        stt = self.lpstate[ctx]
        if sp < 0.42 and not stt["act"]:
            stt["act"], stt["cnt"] = True, 0
        elif stt["act"] and stt["cnt"] < 150:
            pr[st] = pr.get(st, 0) * 10
            stt["cnt"] += 1
        elif stt["cnt"] >= 150:
            stt["act"] = False
        tt = sum(pr.values())
        return {k: v/tt for k, v in pr.items()} if tt else pr

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
            txt = f.read()[:KB_len]
        snts = [s.split() for s in txt.split('.') if s.strip()]
        self.enc.train(snts)
        for s in snts:
            for t in s: self.vsa.vec(t)
        print("TRAINED ✓ (with Counter-Intuitive SCO + Dome Spiral)")

    def gen(self, seed, n=600, t=0.95, show_progress=True):
        ctx = list(seed)
        res = []
        iterator = tqdm(range(n), desc="Generating", unit="tok", ncols=100,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} tok [{elapsed}<{remaining}, {rate_fmt}]',
                       dynamic_ncols=True) if show_progress else range(n)
        
        for step in iterator:
            ckey = tuple(ctx[-2:]) if len(ctx) >= 2 else tuple(ctx)
            ps = self.enc.probs(ctx)
            if not ps:
                if show_progress: iterator.close()
                print(f"\n[!] Stopped at step {step}: no continuations")
                break
            
            if self.mem.infer_boost(ckey, min(ps.values())):
                ps = {k: v*1.3 for k, v in ps.items()}
                ps = {k: v/sum(ps.values()) for k, v in ps.items()}
            
            pv = np.log(np.array(list(ps.values())) + 1e-10) / t
            pv = np.exp(pv - np.max(pv))
            pv /= pv.sum()
            
            nt = np.random.choice(list(ps.keys()), p=pv)
            sp = ps[nt]
            
            self.enc.sco.record_attribution(ckey, nt, sp)
            self.mem.record(ckey, nt, sp, self.enc.lpstate[ckey], self.steps)
            ps = self.enc.lp_bias(ps, ckey, nt, sp)
            
            res.append(nt)
            ctx.append(nt)
            self.steps += 1
            
            if show_progress:
                iterator.set_postfix(tok=nt[:10], p=f"{sp:.3f}", attr=len(self.enc.sco.attribution_history))
        
        if show_progress: print(f"\n[✓] Generated {len(res)} tokens")
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
        print(f"SCO: {len(g.enc.sco.cardinal_memory)} tokens, {len(g.enc.sco.attribution_history)} attrs")
        print(f"DOME: {len(g.enc.dome.token_angles)} mapped, phase={g.enc.dome.spiral_phase:.3f}")

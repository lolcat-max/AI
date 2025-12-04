
# =====================================================================
# FULL SYSTEM: CONCENTRIC + VSA + MEMORY
# =====================================================================

import numpy as np
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ACTIVE MEMORY
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
        boost_rules = self.rules["lowboost"]
        return len(boost_rules) > 0 and minp < 0.45

# VSA
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

# CONCENTRIC ENCODER
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

    def train(self, corp):
        batches = [corp[i:i+400] for i in range(0, len(corp), 400)]
        with ProcessPoolExecutor(2) as ex:
            res = list(tqdm(ex.map(batch_proc, batches), total=len(batches)))
        for u, rs in res:
            self.uni.update(u)
            for ri, rdata in enumerate(rs):
                for pr, ts in rdata.items():
                    self.rings[ri][pr].update(ts)

    def probs(self, ctx):
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
        return {k: v/tt for k, v in ag.items()} if tt else self.probs([])

    def lp_bias(self, pr, ctx, st, sp):
        stt = self.lpstate[ctx]
        if sp < 0.42 and not stt["act"]:
            stt["act"] = True
            stt["cnt"] = 0
        elif stt["act"] and stt["cnt"] < 150:
            pr[st] *= 10
            stt["cnt"] += 1
        elif stt["cnt"] >= 150:
            stt["act"] = False
        tt = sum(pr.values())
        return {k: v/tt for k, v in pr.items()}

# GENERATOR
class ConcGen:
    def __init__(self):
        self.vsa = VSA()
        self.enc = ConcEnc()
        self.mem = ActiveMemory()
        self.steps = 0

    def fit(self, file):
        with open(file,encoding="UTF-8") as f:
            txt = f.read()
        snts = [s.split() for s in txt.split('.') if s.strip()]
        self.enc.train(snts)
        for s in snts:
            for t in s: self.vsa.vec(t)
        print("TRAINED ✓")

    def gen(self, seed, n=600, t=0.95):
        ctx = list(seed)
        res = []
        for _ in range(n):
            ckey = tuple(ctx[-2:])
            ps = self.enc.probs(ctx)
            bp = self.mem.infer_boost(ckey, min(ps.values()))
            if bp: 
                for k in ps: ps[k] *= 1.3
            pv = np.log(list(ps.values())) / t
            pv = np.exp(pv - np.max(pv))
            pv /= pv.sum()
            nt = np.random.choice(list(ps), p=pv)
            sp = ps[nt]
            stt = self.enc.lpstate[ckey]
            self.mem.record(ckey, nt, sp, stt, self.steps)
            ps = self.enc.lp_bias(ps, ckey, nt, sp)
            res.append(nt)
            ctx.append(nt)
            self.steps += 1
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

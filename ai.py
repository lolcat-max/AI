import numpy as np
import torch
from collections import Counter, defaultdict
import os

# ================================================================
# SURJECTION FIELD + OPS
# ================================================================
class SurjectionField:
    """Holds directional weights between tokens."""
    def __init__(self):
        self.map = {}
    def register(self, a,b,v): self.map[(a,b)] = float(v)
    def lookup(self,a,b): return self.map.get((a,b),None)

class SurjectionOps:
    """Single math primitive: surject(u→v)."""
    def __init__(self,field=None): self.field=field or SurjectionField()
    def surject(self,u,v,a=None,b=None):
        u=np.asarray(u,float); v=np.asarray(v,float)
        n=min(len(u),len(v))
        if n==0: return 0.5
        dot=np.dot(u[:n],v[:n]); nv2=np.dot(v[:n],v[:n])+1e-9
        corr=1.0
        if a and b:
            val=self.field.lookup(a,b)
            if val is not None: corr=0.7+0.6*np.tanh(val)
        return float(np.clip(0.5*(np.tanh(corr*dot/nv2)+1),0,1))

# ================================================================
# WORD FEATURES (only for surjection vectors)
# ================================================================
class WordFeatures:
    def __init__(self,tokens):
        self.freq=Counter(tokens); self.total=max(1,len(tokens))
    def vec(self,w):
        L=len(w); f=self.freq.get(w,1)
        return np.array([
            L/10, sum(c.isalpha() for c in w)/(L+1),
            sum(c in "aeiou" for c in w)/(L+1),
            np.log(f+1)/np.log(self.total+1),
            1/(f+1)
        ],float)

# ================================================================
# SURJECTION GENERATOR
# ================================================================
class SurjectionGenerator:
    def __init__(self,tokens,model):
        self.tokens=tokens
        self.model=model
        self.keys=list(model.keys())
        self.field=SurjectionField()
        self.ops=SurjectionOps(self.field)
        self.feat=WordFeatures(tokens)
        self._auto_pairs()

    def _auto_pairs(self):
        big=Counter(zip(self.tokens[:-1],self.tokens[1:]))
        if not big:return
        m=max(big.values())
        for (a,b),c in big.items():
            self.field.register(a,b,c/m)

    def surjection_similarity(self,a,b):
        va,vb=self.feat.vec(a),self.feat.vec(b)
        return self.ops.surject(va,vb,a,b)

    def generate(self,seed,length=80):
        words=seed.split()[:2]
        while len(words)<2:
            words.append(self.tokens[len(words)%len(self.tokens)])
        seed=tuple(words)
        if seed not in self.model: seed=self.keys[np.random.randint(len(self.keys))]
        out=list(seed)
        print(f"\n[Surjection text gen] seed: {' '.join(seed)}")
        for _ in range(length):
            cands=self.model.get(seed,[])
            if not cands:
                seed=self.keys[np.random.randint(len(self.keys))]; continue
            scores=[self.surjection_similarity(out[-1],c) for c in cands]
            if not scores: continue
            probs=torch.softmax(torch.tensor(scores, dtype=torch.float),dim=0).numpy()
            next_word=np.random.choice(cands,p=probs)
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
    path=input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("file missing"); return
    toks=open(path,encoding="utf-8").read().lower().split()
    model=build_ngram(toks,2)
    g=SurjectionGenerator(toks,model)
    while True:
        s=input("\nseed (exit to quit): ")
        if s=="exit":break
        print("\n"+g.generate(s,120))

if __name__=="__main__":
    main()

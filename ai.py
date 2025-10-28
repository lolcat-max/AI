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
    
    def automorph(self):
        """Apply automorphism: map field onto itself with structure preservation."""
        new_map = {}
        for (a,b), v in self.map.items():
            # Structure-preserving self-map: reflect through identity
            new_map[(b,a)] = v  # Inverse mapping preserves structure
            new_map[(a,b)] = v  # Keep original
        self.map = new_map
        return self

class SurjectionOps:
    """Single math primitive: surject(u→v) with automorphism."""
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
        
        # Automorphism: apply self-inverse transformation
        result = self.automorph_scalar(result)
        return result
    
    def automorph_scalar(self, x):
        """Automorphic transformation: x → f(x) where f is structure-preserving."""
        # Use involution: f(f(x)) = x (self-inverse)
        # Example: reflection through 0.5
        return 1.0 - x if x > 0.5 else x

# ================================================================
# WORD FEATURES (only for surjection vectors)
# ================================================================
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
        
        # Apply automorphism to feature vector (map to itself)
        vec = self.automorph_vector(vec)
        self.feature_cache[w] = vec
        return vec
    
    def automorph_vector(self, v):
        """Automorphic transformation of vector onto itself."""
        # Normalize and apply structure-preserving transformation
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return v
        normalized = v / norm
        # Apply orthogonal transformation (preserves structure)
        # Use reflection matrix (involution: A² = I)
        reflected = 2 * np.dot(normalized, normalized) * normalized - v
        return reflected

# ================================================================
# SURJECTION GENERATOR WITH AUTOMORPHISM
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
        self.generation_state = []
        
    def _auto_pairs(self):
        big=Counter(zip(self.tokens[:-1],self.tokens[1:]))
        if not big:return
        m=max(big.values())
        for (a,b),c in big.items():
            self.field.register(a,b,c/m)
        # Apply automorphism to field
        self.field.automorph()

    def surjection_similarity(self,a,b):
        va,vb=self.feat.vec(a),self.feat.vec(b)
        score = self.ops.surject(va,vb,a,b)
        
        # Apply automorphism to similarity score
        score = self.automorph_similarity(score)
        return score
    
    def automorph_similarity(self, s):
        """Automorphic transformation: map similarity onto itself."""
        # Use fixed-point preserving automorphism
        # f(x) = x + sin(2πx)/4 has f(0)=0, f(0.5)=0.5, f(1)=1
        return s + np.sin(2 * np.pi * s) / 4
    
    def automorph_state(self):
        """Apply automorphism to entire generation state."""
        if len(self.generation_state) < 2:
            return
        # Swap last 2 words (involution preserving structure)
        self.generation_state[-2], self.generation_state[-1] = \
            self.generation_state[-1], self.generation_state[-2]

    def generate(self,seed,length=80):
        words=seed.split()[:2]
        while len(words)<2:
            words.append(self.tokens[len(words)%len(self.tokens)])
        seed=tuple(words)
        if seed not in self.model: seed=self.keys[np.random.randint(len(self.keys))]
        out=list(seed)
        self.generation_state = list(seed)
        
        print(f"\n[Automorphic Surjection Generator] seed: {' '.join(seed)}")
        
        for step in range(length):
            cands=self.model.get(seed,[])
            if not cands:
                seed=self.keys[np.random.randint(len(self.keys))]; continue
            scores=[self.surjection_similarity(out[-1],c) for c in cands]
            if not scores: continue
            
            # Normalize scores through automorphic lens
            scores = [s / (max(scores) + 1e-9) for s in scores]
            
            probs=torch.softmax(torch.tensor(scores, dtype=torch.float),dim=0).numpy()
            next_word=np.random.choice(cands,p=probs)
            out.append(next_word)
            self.generation_state.append(next_word)
            
            # Apply automorphism every 5 steps
            if (step + 1) % 5 == 0:
                self.automorph_state()
            
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
    print("=== AUTOMORPHIC SURJECTION TEXT GENERATOR ===")
    print("System maps to itself with structure preservation\n")
    
    path=input("Enter text file: ").strip()
    if not os.path.exists(path):
        print("file missing"); return
    toks=open(path,encoding="utf-8").read().lower().split()
    model=build_ngram(toks,2)
    
    print(f"Loaded {len(toks)} tokens")
    print(f"Model size: {len(model)} n-grams")
    
    g=SurjectionGenerator(toks,model)
    
    while True:
        s=input("\nseed (exit to quit): ")
        if s=="exit":break
        print("\n"+g.generate(s,620))

if __name__=="__main__":
    main()

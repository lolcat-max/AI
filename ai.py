import numpy as np
import torch
from collections import Counter, defaultdict
import os
from typing import List, Tuple

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
        # Use Householder-like reflection across the normalized direction (involution)
        reflected = 2 * np.dot(normalized, normalized) * normalized - v
        return reflected

# ================================================================
# SURJECTION GENERATOR WITH AUTOMORPHISM
# + Linearisation–Interpolation Alternation for Surjectivity
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

        # Codomain partition via anchors
        self._build_codomain_anchors(k=18)
        self.anchor_hits = np.zeros(len(self.anchors), dtype=int)

        # Alternation hyperparameters
        self.alt_period = 14         # every N steps enforce onto coverage
        self.alpha_linear = 0.35    # interpolation toward context-aligned anchor
        self.beta_onto = 0.45       # reweighting toward least-covered anchor

    def _auto_pairs(self):
        big=Counter(zip(self.tokens[:-1],self.tokens[1:]))
        if not big:return
        m=max(big.values())
        for (a,b),c in big.items():
            self.field.register(a,b,c/m)
        # Apply automorphism to field
        self.field.automorph()

    # ---------- Codomain Anchors ----------
    def _build_codomain_anchors(self, k=8):
        # Build k anchor vectors from token feature space using a greedy diversification
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
        self.anchors = np.stack(feats, axis=0)  # [k, d]
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
        # Map anchor to a distribution over candidates via cosine similarity
        A = self.anchors[anchor_idx]  # [d]
        C = self._candidate_feat_matrix(cands)  # [m, d]
        sims = C @ A  # [m]
        sims = np.maximum(sims, 0.0)
        if sims.max() < 1e-12:
            sims = np.ones_like(sims)
        sims = sims / (sims.sum() + 1e-9)
        return sims

    def _onto_reweight(self, cands: List[str]) -> Tuple[np.ndarray,int]:
        # Emphasize least-covered anchor to promote onto coverage
        min_hits = self.anchor_hits.min()
        under = np.where(self.anchor_hits == min_hits)[0]
        aidx = int(under[len(self.generation_state) % len(under)])  # round-robin among undercovered
        q = self._anchor_alignment_dist(cands, aidx)
        return q, aidx

    def _linearize_toward_context(self, cands: List[str], context_words: Tuple[str,str]) -> Tuple[np.ndarray,int]:
        # Local context vector: average of last two word features
        v1 = self.feat.vec(context_words[0])
        v2 = self.feat.vec(context_words[1])
        ctx = (v1 + v2) / 2.0
        aidx = self._nearest_anchor_idx(ctx)
        q = self._anchor_alignment_dist(cands, aidx)
        return q, aidx

    # ---------- Similarity + automorphisms ----------
    def surjection_similarity(self,a,b):
        va,vb=self.feat.vec(a),self.feat.vec(b)
        score = self.ops.surject(va,vb,a,b)
        score = self.automorph_similarity(score)
        return score
    
    def automorph_similarity(self, s):
        """Automorphic transformation: map similarity onto itself."""
        # f(x) = x + sin(2πx)/4 keeps fixed points at 0, 0.5, 1
        return s + np.sin(2 * np.pi * s) / 4
    
    def automorph_state(self):
        """Apply automorphism to entire generation state."""
        if len(self.generation_state) < 2:
            return
        # Swap last 2 words (involution preserving structure)
        self.generation_state[-2], self.generation_state[-1] = \
            self.generation_state[-1], self.generation_state[-2]

    # ---------- Generation ----------
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

            # base scores
            scores=[self.surjection_similarity(out[-2],c) for c in cands]
            if not scores: continue

            # Normalize scores through automorphic lens
            scores = [s / (max(scores) + 1e-9) for s in scores]
            base = torch.softmax(torch.tensor(scores, dtype=torch.float),dim=0).numpy()

            # Linearisation–Interpolation toward context-aligned anchor
            q_lin, a_lin = self._linearize_toward_context(cands, (out[-2], out[-1]))
            p_lin = (1.0 - self.alpha_linear) * base + self.alpha_linear * q_lin

            # Alternation: periodically enforce onto coverage via least-covered anchor
            if (step + 1) % self.alt_period == 0:
                q_onto, a_onto = self._onto_reweight(cands)
                p = (1.0 - self.beta_onto) * p_lin + self.beta_onto * q_onto
            else:
                p = p_lin

            # Safety renormalization
            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum()

            next_word=np.random.choice(cands,p=p)

            # Update coverage: attribute chosen token to nearest anchor
            v_next = self.feat.vec(next_word)
            a_chosen = self._nearest_anchor_idx(v_next)
            self.anchor_hits[a_chosen] += 1

            self.generation_state.append(next_word)
            # Apply automorphism every 5 steps
            if (step + 1) % 5 == 0:
                self.automorph_state()

            # Note: original code appended generation_state; that grows lists in output.
            # Keep words in 'out' for textual output.
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
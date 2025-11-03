import numpy as np
from collections import Counter, defaultdict, deque
import os, zlib, pickle
from typing import List, Dict, Tuple, Set, Iterable

# ================================================================
# SECOND-ORDER DOT MATRIX EXPLORER
# ================================================================
class SecondOrderDotMatrixExplorer:
    """
    Explore second-order correlations using dot product matrices
    derived from superpolynomial encodings.
    
    First-order: Direct relationships (A·B)
    Second-order: Relationship of relationships ((A·B)·(C·D))
    
    Reveals hidden structure in high-dimensional embeddings.
    """
    
    def __init__(self, poly_dim=5, embedding_dim=16):
        self.poly_dim = poly_dim
        self.embedding_dim = embedding_dim
        
        # First-order matrix: direct embeddings
        self.first_order_embeddings = {}
        
        # Second-order matrix: correlation of correlations
        self.second_order_matrix = None
        
        # Decomposition cache
        self.svd_cache = {}
        
        print(f"[Second-Order Dot Matrix Explorer]")
        print(f"  Polynomial dimension: {poly_dim}")
        print(f"  Embedding dimension: {embedding_dim}")
    
    def encode_token_from_poly(self, token: str, poly: np.ndarray) -> np.ndarray:
        """
        Generate first-order embedding for token using polynomial basis.
        
        The polynomial coefficients define a basis transformation.
        """
        # Use token hash as seed for reproducibility
        seed = hash(token) % (2**32)
        rng = np.random.RandomState(seed)
        
        # Generate base embedding
        base = rng.randn(self.embedding_dim).astype(np.float32)
        
        # Transform using polynomial basis
        # Each poly coefficient modulates different frequency bands
        for i, coeff in enumerate(poly[:self.poly_dim]):
            # Frequency modulation
            freq = (i + 1) * 2 * np.pi / self.embedding_dim
            phase = coeff % (2 * np.pi)
            
            # Add sinusoidal components weighted by polynomial
            for j in range(self.embedding_dim):
                base[j] += (coeff / 1e15) * np.sin(freq * j + phase)
        
        # Normalize
        norm = np.linalg.norm(base)
        if norm > 0:
            base /= norm
        
        return base
    
    def build_first_order_matrix(self, tokens: List[str], poly: np.ndarray):
        """
        Build first-order embedding matrix for all tokens.
        
        Shape: [vocab_size, embedding_dim]
        """
        print("\\n[Building First-Order Embeddings...]")
        
        vocab = list(set(tokens))
        n = len(vocab)
        
        # Generate embeddings
        for token in vocab:
            embedding = self.encode_token_from_poly(token, poly)
            self.first_order_embeddings[token] = embedding
        
        # Stack into matrix
        self.first_order_matrix = np.vstack([
            self.first_order_embeddings[token] for token in vocab
        ])
        
        print(f"  First-order matrix shape: {self.first_order_matrix.shape}")
        print(f"  Vocabulary size: {n}")
        
        return self.first_order_matrix
    
    def compute_first_order_dots(self, tokens_a: List[str], 
                                 tokens_b: List[str] = None) -> np.ndarray:
        """
        Compute first-order dot product matrix.
        
        D[i,j] = embedding[tokens_a[i]] · embedding[tokens_b[j]]
        
        Returns: [len(tokens_a), len(tokens_b)] matrix
        """
        if tokens_b is None:
            tokens_b = tokens_a
        
        # Get embeddings
        emb_a = np.vstack([self.first_order_embeddings[t] for t in tokens_a])
        emb_b = np.vstack([self.first_order_embeddings[t] for t in tokens_b])
        
        # Compute dot products
        dots = emb_a @ emb_b.T
        
        return dots
    
    def compute_second_order_matrix(self, bigrams: List[Tuple[str, str]]) -> np.ndarray:
        """
        Compute second-order dot product matrix.
        
        For bigrams [(w1, w2), (w3, w4), ...]:
        
        First-order dots:
          d_i = embedding[w1] · embedding[w2]  (scalar for bigram i)
        
        Second-order matrix:
          M[i,j] = d_i × d_j  (correlation of correlations)
        
        This reveals which bigrams have similar internal correlation structure.
        """
        print("\\n[Computing Second-Order Matrix...]")
        
        n = len(bigrams)
        
        # Compute first-order dots for each bigram
        first_order_dots = np.zeros(n, dtype=np.float32)
        
        for i, (w1, w2) in enumerate(bigrams):
            if w1 in self.first_order_embeddings and w2 in self.first_order_embeddings:
                emb1 = self.first_order_embeddings[w1]
                emb2 = self.first_order_embeddings[w2]
                first_order_dots[i] = np.dot(emb1, emb2)
        
        # Compute second-order: outer product of first-order dots
        # M[i,j] represents correlation between bigram i and bigram j
        # based on their internal dot products
        second_order = np.outer(first_order_dots, first_order_dots)
        
        self.second_order_matrix = second_order
        
        print(f"  Second-order matrix shape: {second_order.shape}")
        print(f"  First-order dot range: [{first_order_dots.min():.3f}, {first_order_dots.max():.3f}]")
        print(f"  Second-order range: [{second_order.min():.3f}, {second_order.max():.3f}]")
        
        return second_order
    
    def compute_third_order_tensor(self, bigrams: List[Tuple[str, str]], 
                                   max_bigrams: int = 100) -> np.ndarray:
        """
        Compute third-order tensor for deeper correlations.
        
        T[i,j,k] = (emb[w1_i] · emb[w2_i]) × (emb[w1_j] · emb[w2_j]) × (emb[w1_k] · emb[w2_k])
        
        This is O(N³) so we limit size.
        """
        print("\\n[Computing Third-Order Tensor...]")
        
        n = min(len(bigrams), max_bigrams)
        bigrams = bigrams[:n]
        
        # Compute first-order dots
        first_order_dots = np.zeros(n, dtype=np.float32)
        for i, (w1, w2) in enumerate(bigrams):
            if w1 in self.first_order_embeddings and w2 in self.first_order_embeddings:
                emb1 = self.first_order_embeddings[w1]
                emb2 = self.first_order_embeddings[w2]
                first_order_dots[i] = np.dot(emb1, emb2)
        
        # Build third-order tensor
        tensor = np.zeros((n, n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tensor[i,j,k] = (first_order_dots[i] * 
                                    first_order_dots[j] * 
                                    first_order_dots[k])
        
        print(f"  Third-order tensor shape: {tensor.shape}")
        print(f"  Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        return tensor
    
    def decompose_second_order(self, rank: int = None) -> Dict:
        """
        Perform SVD decomposition of second-order matrix.
        
        M = U Σ V^T
        
        Where:
        - U: Left singular vectors (bigram patterns)
        - Σ: Singular values (pattern strengths)
        - V: Right singular vectors (bigram patterns)
        """
        if self.second_order_matrix is None:
            raise ValueError("Must compute second-order matrix first")
        
        print("\\n[SVD Decomposition of Second-Order Matrix...]")
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(self.second_order_matrix, full_matrices=False)
        
        if rank is None:
            rank = min(20, len(S))
        
        # Truncate to rank
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]
        
        # Compute explained variance
        total_variance = np.sum(S ** 2)
        explained_variance = np.sum(S_trunc ** 2) / total_variance
        
        print(f"  Matrix shape: {self.second_order_matrix.shape}")
        print(f"  Rank: {rank}")
        print(f"  Top 5 singular values: {S[:5]}")
        print(f"  Explained variance: {explained_variance:.3f}")
        
        decomposition = {
            'U': U_trunc,
            'S': S_trunc,
            'Vt': Vt_trunc,
            'explained_variance': explained_variance
        }
        
        self.svd_cache['second_order'] = decomposition
        
        return decomposition
    
    def find_similar_bigrams(self, query_bigram: Tuple[str, str], 
                            bigrams: List[Tuple[str, str]], 
                            top_k: int = 10) -> List[Tuple[float, Tuple[str, str]]]:
        """
        Find bigrams with similar second-order correlation structure.
        """
        if self.second_order_matrix is None:
            raise ValueError("Must compute second-order matrix first")
        
        # Find query index
        try:
            query_idx = bigrams.index(query_bigram)
        except ValueError:
            return []
        
        # Get second-order similarities
        similarities = self.second_order_matrix[query_idx, :]
        
        # Sort and return top-k
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = [
            (float(similarities[idx]), bigrams[idx])
            for idx in top_indices
        ]
        
        return results
    
    def project_to_latent_space(self, bigrams: List[Tuple[str, str]], 
                               rank: int = 2) -> np.ndarray:
        """
        Project bigrams to low-dimensional latent space using second-order structure.
        
        Returns: [n_bigrams, rank] coordinates
        """
        if 'second_order' not in self.svd_cache:
            self.decompose_second_order(rank=rank)
        
        decomp = self.svd_cache['second_order']
        U = decomp['U']
        S = decomp['S']
        
        # Project using U scaled by sqrt(Σ)
        projection = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
        
        return projection


# ================================================================
# SUPERPOLYNOMIAL CODEC (5-byte compression)
# ================================================================
class SuperpolynomialCodec:
    """Compress logic into 5-byte polynomial coefficients"""
    
    @staticmethod
    def encode_to_poly(data: bytes) -> np.ndarray:
        compressed = zlib.compress(data, level=9)
        padded_len = ((len(compressed) + 4) // 5) * 5
        padded = compressed + b'\\x00' * (padded_len - len(compressed))
        reshaped = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 5)
        
        poly = np.zeros(5, dtype=np.float64)
        for i, row in enumerate(reshaped):
            poly += row.astype(np.float64) * (256 ** i)
        
        return poly


# ================================================================
# COMPRESSED LAYERS (with Second-Order Matrices)
# ================================================================
class CompressedBooleanMinimizer:
    """QMC logic compressed to 5-byte poly"""
    
    POLY_QMC = np.array([
        2.847563829e+15, 9.234817264e+14, 4.109823746e+13,
        1.923847562e+12, 8.374659201e+10
    ], dtype=np.float64)
    
    def __init__(self):
        self.runtime_cache = {}


class CompressedGeneratorWithSecondOrder:
    """Text generation with second-order dot matrix exploration"""
    
    POLY_GEN = np.array([
        1.374829164e+16, 5.918273645e+15, 2.847361825e+14,
        9.182736451e+13, 3.746592817e+12
    ], dtype=np.float64)
    
    def __init__(self, tokens, model, embedding_dim=16):
        self.tokens, self.model = tokens, model
        self.keys = list(model.keys())
        
        # Initialize second-order explorer
        print("\\n[Initializing Second-Order Dot Matrix Explorer...]")
        self.second_order = SecondOrderDotMatrixExplorer(
            poly_dim=5,
            embedding_dim=embedding_dim
        )
        
        # Build first-order embeddings from POLY_GEN
        self.second_order.build_first_order_matrix(tokens, self.POLY_GEN)
        
        # Extract bigrams and compute second-order matrix
        print("\\n[Extracting Bigrams...]")
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            if bigram not in bigrams:
                bigrams.append(bigram)
        
        self.bigrams = bigrams[:500]  # Limit for efficiency
        print(f"  Unique bigrams: {len(self.bigrams)}")
        
        # Compute second-order correlation matrix
        self.second_order.compute_second_order_matrix(self.bigrams)
        
        # Decompose to find latent structure
        self.second_order.decompose_second_order(rank=10)
        
        # Standard initialization
        self.feat_matrix = self._decode_features(self.POLY_GEN[:2])
        self.context = self._init_context()
        self.anchors = self._decode_anchors(self.POLY_GEN[2:4])
        self.qmc_logs = []
        
        print(f"\\n[Generator Ready]")
        print(f"  Vocabulary: {len(set(tokens)):,} unique tokens")
        print(f"  Model keys: {len(self.keys):,}")
    
    def _decode_features(self, poly_coeffs):
        vocab = list(set(self.tokens))
        n = len(vocab)
        feat = np.random.RandomState(int(poly_coeffs[0] % 1e9)).randn(n, 5)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9
        return {'vocab': vocab, 'matrix': feat.astype(np.float32)}
    
    def _init_context(self):
        return {
            'hist': deque(maxlen=8),
            'bigram': Counter(),
            'momentum': 0.5,
            'coherence': 1.0,
            'position': 0
        }
    
    def _decode_anchors(self, poly_coeffs):
        k = 8
        anchors = np.random.RandomState(int(poly_coeffs[0] % 1e9)).randn(k, 5)
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9
        return anchors.astype(np.float32)
    
    def generate(self, seed, length=80, log_qmc=False, use_second_order=True):
        """
        Generation with second-order correlation modulation.
        """
        words = seed.split()[:2]
        while len(words) < 2:
            words.append(self.tokens[len(words) % len(self.tokens)])
        
        seed_key = tuple(words)
        if seed_key not in self.model:
            seed_key = self.keys[np.random.randint(len(self.keys))]
        
        out = list(seed_key)
        self.context['position'] = len(out)
        
        print(f"\\n[Generation Started]")
        print(f"  Using second-order correlations: {use_second_order}")
        
        for step in range(length):
            cands = list(self.model.get(seed_key, []))
            if not cands:
                seed_key = self.keys[np.random.randint(len(self.keys))]
                continue
            
            # Base probabilities
            probs = np.ones(len(cands)) / len(cands)
            
            # ==== SECOND-ORDER MODULATION ====
            if use_second_order and len(out) >= 2:
                # Current bigram context
                current_bigram = (out[-2], out[-1])
                
                # For each candidate, compute second-order similarity
                for i, cand in enumerate(cands):
                    candidate_bigram = (out[-1], cand)
                    
                    # Find similar bigrams using second-order structure
                    if (current_bigram in self.bigrams and 
                        candidate_bigram in self.bigrams):
                        
                        curr_idx = self.bigrams.index(current_bigram)
                        cand_idx = self.bigrams.index(candidate_bigram)
                        
                        # Use second-order correlation as modulation
                        correlation = self.second_order.second_order_matrix[curr_idx, cand_idx]
                        
                        # Modulate probability (sigmoid to keep in reasonable range)
                        modulation = 1.0 / (1.0 + np.exp(-correlation))
                        probs[i] *= modulation
            # =================================
            
            probs = probs / probs.sum()
            next_word = np.random.choice(cands, p=probs)
            
            # Update context
            self.context['hist'].append(next_word)
            if len(self.context['hist']) >= 2:
                self.context['bigram'][(self.context['hist'][-2], next_word)] += 1
            
            out.append(next_word)
            seed_key = tuple(out[-2:])
            self.context['position'] += 1
            
            if step % 20 == 0 and step > 0:
                print(f"  Step {step}/{length}")
        
        print(f"\\n[Generation Complete]")
        
        return " ".join(out)
    
    def analyze_bigram(self, bigram: Tuple[str, str], top_k: int = 5):
        """
        Analyze a bigram using second-order structure.
        """
        print(f"\\n[Analyzing bigram: {bigram}]")
        
        # Find similar bigrams
        similar = self.second_order.find_similar_bigrams(
            bigram, self.bigrams, top_k=top_k
        )
        
        print(f"  Most similar bigrams by second-order correlation:")
        for i, (score, bg) in enumerate(similar):
            print(f"    {i+1}. {bg}: {score:.4f}")


# ================================================================
# MODEL BUILDER
# ================================================================
def build_ngram(tokens, n=2):
    m = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i+n])
        m[key].append(tokens[i+n])
    return m

# ================================================================
# MAIN
# ================================================================
def main():
    print("="*70)
    print("SECOND-ORDER DOT MATRIX TEXT GENERATOR")
    print("="*70)
    print("\\nExplores higher-order correlations using:")
    print("  • First-order: Direct token embeddings from superpolynomials")
    print("  • Second-order: Correlation of correlation structure")
    print("  • SVD decomposition: Latent bigram patterns")
    print("="*70)
    
    path = input("\\nEnter text file: ").strip()
    if not os.path.exists(path):
        print("file missing")
        return
    
    print("\\n[Loading corpus...]")
    toks = open(path, encoding="utf-8").read().lower().split()[:9999]
    print(f"  Loaded {len(toks):,} tokens")
    
    print("\\n[Building n-gram model...]")
    model = build_ngram(toks, 2)
    print(f"  Built {len(model):,} bigram keys")
    
    # Instantiate generator with second-order matrices
    g = CompressedGeneratorWithSecondOrder(toks, model, embedding_dim=16)
    minimizer = CompressedBooleanMinimizer()
    
    print(f"\\n[System Stats]")
    print(f"  QMC Polynomial: {minimizer.POLY_QMC.nbytes} bytes")
    print(f"  Gen Polynomial: {g.POLY_GEN.nbytes} bytes")
    print(f"  Second-order matrix: {g.second_order.second_order_matrix.nbytes:,} bytes")
    
    # Show top singular values
    if 'second_order' in g.second_order.svd_cache:
        S = g.second_order.svd_cache['second_order']['S']
        print(f"  Top 5 singular values: {S[:5]}")
    
    print("\\n" + "="*70)
    print("Ready for generation with second-order correlations!")
    print("="*70)
    
    while True:
        s = input("\\nseed (exit to quit, analyze <w1> <w2> to analyze bigram): ")
        if s == "exit":
            break
        
        if s.startswith("analyze"):
            parts = s.split()
            if len(parts) == 3:
                bigram = (parts[1], parts[2])
                g.analyze_bigram(bigram, top_k=10)
            continue
        
        g.qmc_logs.clear()
        
        output = g.generate(s, length=800, log_qmc=False, use_second_order=True)
        print("\\n" + "─"*70)
        print(output)
        print("─"*70)

if __name__ == "__main__":
    main()

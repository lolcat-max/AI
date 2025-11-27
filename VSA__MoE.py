import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import threading
import os

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            default = args[0]
            super().__init__(default)
        else:
            super().__init__(int)

# =====================================================================
# 2D POLARIZATION VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions  # Must be even for 2D polarization
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_polarized_vector(self, normalize: bool = True) -> np.ndarray:
        """Generate vector from 2D polar coordinates (polarization states)."""
        dim_2d = self.dimensions // 2
        theta = np.random.uniform(0, 2 * np.pi, dim_2d)
        r = np.ones(dim_2d)
        x_channel = r * np.cos(theta)
        y_channel = r * np.sin(theta)
        vec = np.stack([x_channel, y_channel], axis=0).reshape(-1)
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def bind_polarized(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """Polarization-aware binding with 2D channel swapping."""
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        dim = len(vec_a) // 2
        
        fft_a_swapped = np.ones_like(fft_a, dtype=complex)
        fft_a_swapped[:dim] = fft_b[dim:]
        fft_a_swapped[dim:] = fft_b[:dim]
        
        result = np.fft.ifft(fft_a + fft_a_swapped)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9)

    def add_to_codebook(self, symbol: str) -> np.ndarray:
        with self.lock:
            if symbol not in self.codebook:
                self.codebook[symbol] = self.create_polarized_vector()
            return self.codebook[symbol]

    def save_codebook(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
        print(f"✓ Polarized codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        print(f"✓ Polarized codebook loaded from {filepath}")

# =====================================================================
# POLARIZATION TRANSITION ENCODER (N-GRAM COUNTS)
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        self.unigram_counts = IntDefaultDict()
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.bigram_transitions = defaultdict(IntDefaultDict)
        self.trigram_transitions = defaultdict(IntDefaultDict)
        self.lock = threading.Lock()

    def encode_unigram(self, token: str):
        with self.lock:
            self.unigram_counts[token] += 1

    def encode_bigram(self, token1: str, token2: str):
        with self.lock:
            self.bigram_transitions[token1][token2] += 1
            key = (token1, token2)
            if key not in self.bigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                self.bigram_vectors[key] = self.vsa.bind_polarized(vec1, vec2)

    def encode_trigram(self, token1: str, token2: str, token3: str):
        with self.lock:
            self.trigram_transitions[(token1, token2)][token3] += 1
            key = (token1, token2, token3)
            if key not in self.trigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                vec3 = self.vsa.add_to_codebook(token3)
                bound12 = self.vsa.bind_polarized(vec1, vec2)
                self.trigram_vectors[key] = self.vsa.bind_polarized(bound12, vec3)

    def _process_sequence_batch(self, sequences: List[List[str]]):
        for sequence in sequences:
            for token in sequence:
                self.encode_unigram(token)
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning polarized transitions from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self._process_sequence_batch, batches), 
                     total=len(batches), desc="Polarized batches", ncols=80))
        print(f"  ✓ Learned {len(self.unigram_counts)} unigram counts")
        print(f"  ✓ Learned {sum(len(v) for v in self.bigram_transitions.values())} bigram transitions")
        print(f"  ✓ Learned {sum(len(v) for v in self.trigram_transitions.values())} trigram transitions")

    def get_unigram_probabilities(self) -> Dict[str, float]:
        total = sum(self.unigram_counts.values())
        if total == 0:
            return {}
        return {token: count / total for token, count in self.unigram_counts.items()}

    def get_bigram_probabilities(self, last_token: str) -> Optional[Dict[str, float]]:
        if last_token not in self.bigram_transitions: 
            return None
        candidates = self.bigram_transitions[last_token]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def get_trigram_probabilities(self, last_two_tokens: Tuple[str, str]) -> Optional[Dict[str, float]]:
        if last_two_tokens not in self.trigram_transitions: 
            return None
        candidates = self.trigram_transitions[last_two_tokens]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for name, data in [("unigram_counts", self.unigram_counts),
                          ("bigram_transitions", self.bigram_transitions),
                          ("trigram_transitions", self.trigram_transitions),
                          ("bigram_vectors", self.bigram_vectors),
                          ("trigram_vectors", self.trigram_vectors)]:
            with open(os.path.join(directory, f"{name}.pkl"), 'wb') as f:
                pickle.dump(data, f)
        print(f"✓ Polarized transition model saved to {directory}")

    def load_model(self, directory: str):
        for name in ["unigram_counts", "bigram_transitions", "trigram_transitions", 
                     "bigram_vectors", "trigram_vectors"]:
            with open(os.path.join(directory, f"{name}.pkl"), 'rb') as f:
                setattr(self, name, pickle.load(f))
        print(f"✓ Polarized transition model loaded from {directory}")
        
        
        
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import threading
import os

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            default = args[0]
            super().__init__(default)
        else:
            super().__init__(int)

# =====================================================================
# 2D POLARIZATION VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_polarized_vector(self, normalize: bool = True) -> np.ndarray:
        """Generate vector from 2D polar coordinates (polarization states)."""
        dim_2d = self.dimensions // 2
        theta = np.random.uniform(0, 2 * np.pi, dim_2d)
        r = np.ones(dim_2d)
        x_channel = r * np.cos(theta)
        y_channel = r * np.sin(theta)
        vec = np.stack([x_channel, y_channel], axis=0).reshape(-1)
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def bind_polarized(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """Polarization-aware binding with 2D channel swapping."""
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        dim = len(vec_a) // 2
        
        fft_a_swapped = np.ones_like(fft_a, dtype=complex)
        fft_a_swapped[:dim] = fft_b[dim:]
        fft_a_swapped[dim:] = fft_b[:dim]
        
        result = np.fft.ifft(fft_a + fft_a_swapped)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9)

    def add_to_codebook(self, symbol: str) -> np.ndarray:
        with self.lock:
            if symbol not in self.codebook:
                self.codebook[symbol] = self.create_polarized_vector()
            return self.codebook[symbol]

    def save_codebook(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
        print(f"✓ Polarized codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        print(f"✓ Polarized codebook loaded from {filepath}")

# =====================================================================
# POLARIZATION TRANSITION ENCODER (N-GRAM COUNTS)
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        self.unigram_counts = IntDefaultDict()
        self.bigram_vectors = {}
        self.trigram_vectors = {}
        self.bigram_transitions = defaultdict(IntDefaultDict)
        self.trigram_transitions = defaultdict(IntDefaultDict)
        self.lock = threading.Lock()

    def encode_unigram(self, token: str):
        with self.lock:
            self.unigram_counts[token] += 1

    def encode_bigram(self, token1: str, token2: str):
        with self.lock:
            self.bigram_transitions[token1][token2] += 1
            key = (token1, token2)
            if key not in self.bigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                self.bigram_vectors[key] = self.vsa.bind_polarized(vec1, vec2)

    def encode_trigram(self, token1: str, token2: str, token3: str):
        with self.lock:
            self.trigram_transitions[(token1, token2)][token3] += 1
            key = (token1, token2, token3)
            if key not in self.trigram_vectors:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                vec3 = self.vsa.add_to_codebook(token3)
                bound12 = self.vsa.bind_polarized(vec1, vec2)
                self.trigram_vectors[key] = self.vsa.bind_polarized(bound12, vec3)

    def _process_sequence_batch(self, sequences: List[List[str]]):
        for sequence in sequences:
            for token in sequence:
                self.encode_unigram(token)
            for i in range(len(sequence) - 1):
                self.encode_bigram(sequence[i], sequence[i+1])
            for i in range(len(sequence) - 2):
                self.encode_trigram(sequence[i], sequence[i+1], sequence[i+2])

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning polarized transitions from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self._process_sequence_batch, batches), 
                     total=len(batches), desc="Polarized batches", ncols=80))
        print(f"  ✓ Learned {len(self.unigram_counts)} unigram counts")
        print(f"  ✓ Learned {sum(len(v) for v in self.bigram_transitions.values())} bigram transitions")
        print(f"  ✓ Learned {sum(len(v) for v in self.trigram_transitions.values())} trigram transitions")

    def get_unigram_probabilities(self) -> Dict[str, float]:
        total = sum(self.unigram_counts.values())
        if total == 0:
            return {}
        return {token: count / total for token, count in self.unigram_counts.items()}

    def get_bigram_probabilities(self, last_token: str) -> Optional[Dict[str, float]]:
        if last_token not in self.bigram_transitions: 
            return None
        candidates = self.bigram_transitions[last_token]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def get_trigram_probabilities(self, last_two_tokens: Tuple[str, str]) -> Optional[Dict[str, float]]:
        if last_two_tokens not in self.trigram_transitions: 
            return None
        candidates = self.trigram_transitions[last_two_tokens]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for name, data in [("unigram_counts", self.unigram_counts),
                          ("bigram_transitions", self.bigram_transitions),
                          ("trigram_transitions", self.trigram_transitions),
                          ("bigram_vectors", self.bigram_vectors),
                          ("trigram_vectors", self.trigram_vectors)]:
            with open(os.path.join(directory, f"{name}.pkl"), 'wb') as f:
                pickle.dump(data, f)
        print(f"✓ Polarized transition model saved to {directory}")

    def load_model(self, directory: str):
        for name in ["unigram_counts", "bigram_transitions", "trigram_transitions", 
                     "bigram_vectors", "trigram_vectors"]:
            with open(os.path.join(directory, f"{name}.pkl"), 'rb') as f:
                setattr(self, name, pickle.load(f))
        print(f"✓ Polarized transition model loaded from {directory}")

# =====================================================================
# SEMANTICALLY PLAUSIBLE CATEGORY ERROR GENERATOR
# =====================================================================
class CategoryErrorGenerator:
    """Purposely violate semantic categories while maintaining plausibility."""
    def __init__(self, vsa: VectorSymbolicArchitecture, transition_encoder: TransitionEncoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.semantic_categories = self._build_semantic_categories()
    
    def _build_semantic_categories(self) -> Dict[str, List[str]]:
        print("Building semantic category clusters...")
        categories = defaultdict(list)
        for token, vec in tqdm(self.vsa.codebook.items(), desc="Categorizing", ncols=80):
            x_channel = vec[0]
            y_channel = vec[1]
            angle = np.arctan2(y_channel, x_channel)
            category_id = int((angle + np.pi) / (np.pi / 4)) % 8
            categories[f"cat_{category_id}"].append(token)
        print(f"  ✓ Created {len(categories)} semantic categories")
        return dict(categories)
    
    def _get_incompatible_category(self, current_category: str) -> str:
        """Get opposite semantic category (maximally distant)."""
        current_id = int(current_category.split("_")[1])
        opposite_id = (current_id + 4) % 8
        return f"cat_{opposite_id}"
    
    def _get_token_category(self, token: str) -> Optional[str]:
        for cat_id, tokens in self.semantic_categories.items():
            if token in tokens:
                return cat_id
        return None
    
    def _compute_semantic_plausibility(self, candidate: str, context: List[str], 
                                      context_window: int = 3) -> float:
        """Compute semantic plausibility using VSA similarity with context [web:41][web:44][web:46]."""
        if candidate not in self.vsa.codebook:
            return 0.0
        
        candidate_vec = self.vsa.codebook[candidate]
        
        # Compute average similarity to recent context tokens [web:48][web:50]
        recent_context = context[-context_window:]
        similarities = []
        
        for ctx_token in recent_context:
            if ctx_token in self.vsa.codebook:
                ctx_vec = self.vsa.codebook[ctx_token]
                sim = self.vsa.similarity(candidate_vec, ctx_vec)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        # Return mean similarity as plausibility score [web:41][web:46]
        return np.mean(similarities)
    
    def _get_ngram_plausibility(self, candidate: str, context: List[str]) -> float:
        """Use n-gram statistics as additional plausibility signal [web:42][web:48]."""
        plausibility = 0.0
        
        # Check bigram plausibility with last token
        if len(context) >= 1:
            last_token = context[-1]
            if last_token in self.transition_encoder.bigram_transitions:
                bigram_probs = self.transition_encoder.get_bigram_probabilities(last_token)
                if bigram_probs and candidate in bigram_probs:
                    plausibility += bigram_probs[candidate]
        
        # Check trigram plausibility with last two tokens
        if len(context) >= 2:
            last_two = tuple(context[-2:])
            if last_two in self.transition_encoder.trigram_transitions:
                trigram_probs = self.transition_encoder.get_trigram_probabilities(last_two)
                if trigram_probs and candidate in trigram_probs:
                    plausibility += trigram_probs[candidate] * 2.0  # Weight trigrams higher
        
        return plausibility
    
    def stream_generation(self, seed: List[str], max_tokens: int = 50, 
                         temperature: float = 1.0, 
                         error_rate: float = 0.7,
                         plausibility_weight: float = 0.8):
        """Generate with semantically plausible category errors [web:41][web:44][web:48].
        
        Args:
            plausibility_weight: Balance between plausibility (1.0) and randomness (0.0)
        """
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return
        
        for _ in range(max_tokens):
            force_error = np.random.random() < error_rate
            
            if force_error and len(context) >= 1:
                last_token = context[-1]
                last_category = self._get_token_category(last_token)
                
                if last_category:
                    error_category = self._get_incompatible_category(last_category)
                    candidate_tokens = self.semantic_categories.get(error_category, [])
                    
                    if candidate_tokens:
                        # Compute plausibility scores for all candidates [web:41][web:46]
                        plausibility_scores = {}
                        
                        for token in candidate_tokens:
                            # Combine VSA similarity and n-gram plausibility [web:42][web:48]
                            vsa_plausibility = self._compute_semantic_plausibility(token, context)
                            ngram_plausibility = self._get_ngram_plausibility(token, context)
                            
                            # Weighted combination [web:44][web:50]
                            combined = (vsa_plausibility + ngram_plausibility) / 2.0
                            plausibility_scores[token] = combined
                        
                        # Build probability distribution [web:41][web:42]
                        probs = {}
                        total_counts = sum(self.transition_encoder.unigram_counts.values())
                        
                        for token in candidate_tokens:
                            # Base probability from unigram frequency
                            base_prob = self.transition_encoder.unigram_counts.get(token, 1) / total_counts
                            
                            # Modulate by plausibility [web:46][web:48]
                            plausibility = plausibility_scores.get(token, 0.0)
                            plausibility_boost = 1.0 + (plausibility * plausibility_weight * 10.0)
                            
                            probs[token] = base_prob * plausibility_boost
                        
                        # Temperature sampling from plausibility-weighted distribution [web:41]
                        tokens = list(probs.keys())
                        prob_vals = np.array(list(probs.values()))
                        
                        if temperature > 0:
                            prob_vals = np.log(prob_vals + 1e-9) / temperature
                            prob_vals = np.exp(prob_vals)
                        prob_vals /= np.sum(prob_vals)
                        
                        next_token = np.random.choice(tokens, p=prob_vals)
                        yield next_token
                        context.append(next_token)
                        continue
            
            # Fallback: normal n-gram prediction [web:11]
            probs = None
            if len(context) >= 2:
                probs = self.transition_encoder.get_trigram_probabilities(tuple(context[-2:]))
            if probs is None and len(context) >= 1:
                probs = self.transition_encoder.get_bigram_probabilities(context[-1])
            if probs is None:
                probs = self.transition_encoder.get_unigram_probabilities()
            
            if not probs:
                next_token = np.random.choice(list(self.vsa.codebook.keys()))
            else:
                tokens = list(probs.keys())
                prob_vals = np.array(list(probs.values()))
                if temperature > 0:
                    prob_vals = np.log(prob_vals + 1e-9) / temperature
                    prob_vals = np.exp(prob_vals)
                prob_vals /= np.sum(prob_vals)
                next_token = np.random.choice(tokens, p=prob_vals)
            
            yield next_token
            context.append(next_token)

# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("2D POLARIZATION VSA + SEMANTICALLY PLAUSIBLE CATEGORY ERROR")
    print("Category violations constrained by semantic plausibility")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=128)
    trans_encoder = TransitionEncoder(vsa)
    choice = input("[N]ew polarized model or [L]oad existing? ").strip().lower()

    if choice == "l":
        directory = input("Model directory: ").strip()
        vsa.load_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.load_model(directory)
    else:
        filename = input("Corpus filename: ")
        print("Loading polarized corpus...")
        with open(filename, encoding="utf-8") as f: 
            raw_text = f.read()
        sentences = raw_text.split(".")
        corpus = [s.split() for s in tqdm(sentences, desc="Tokenizing", ncols=80) if s.split()]
        print(f"Corpus: {len(corpus)} sequences")
        print("[1] Learning Polarized Transitions (Multithreaded)")
        print("-"*80)
        trans_encoder.learn_transitions(corpus, max_workers=8, batch_size=50)
        print("Building polarized vocabulary...")
        for sentence in tqdm(corpus, desc="Polarized Vocab", ncols=80):
            for token in sentence: 
                vsa.add_to_codebook(token)
        print(f"  ✓ Polarized vocabulary: {len(vsa.codebook)} tokens")
        directory = input("Save polarized model to directory: ").strip()
        vsa.save_codebook(os.path.join(directory, "codebook.pkl"))
        trans_encoder.save_model(directory)
    
    print("\n[2] SEMANTICALLY PLAUSIBLE CATEGORY ERROR GENERATION")
    print("-"*80)
    error_gen = CategoryErrorGenerator(vsa, trans_encoder)
    
    while True:
        user_input = input("USER: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Ultra-low error rate for mostly coherent output
        error_rate = 0.0001
        
        # High plausibility weight steers toward semantic coherence [web:41][web:48]
        plausibility_weight = 0.9
        
        print("AI: ", end='', flush=True)
        for token in error_gen.stream_generation(user_input.split(), 
                                                max_tokens=350, 
                                                temperature=0.7,
                                                error_rate=error_rate,
                                                plausibility_weight=plausibility_weight):
            print(token, end=' ', flush=True)
        print("\n")

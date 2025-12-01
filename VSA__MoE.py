import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque, Counter
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import pickle
import threading
import os
import re

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

    @staticmethod
    def _process_batch_static(batch: List[List[str]]) -> Tuple[Counter, dict, dict]:
        """
        Static worker that processes a batch and returns partial results.
        Does NOT access 'self'.
        """
        local_unigrams = Counter()
        local_bigrams = defaultdict(Counter)
        local_trigrams = defaultdict(Counter)

        # Replicate your original _process_sequence_batch logic here
        for sequence in batch:
            # Example logic (replace with your actual counting logic)
            for i, token in enumerate(sequence):
                local_unigrams[token] += 1
                
                if i > 0:
                    prev = sequence[i-1]
                    local_bigrams[prev][token] += 1
                
                if i > 1:
                    prev2 = sequence[i-2]
                    prev1 = sequence[i-1]
                    local_trigrams[(prev2, prev1)][token] += 1
        
        # Convert defaultdicts to regular dicts for safe pickling/returning
        return local_unigrams, dict(local_bigrams), dict(local_trigrams)

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 5000):
        print("Learning polarized transitions from corpus (Multiprocessing)...")
        
        # Create batches
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            results = list(tqdm(executor.map(self._process_batch_static, batches), 
                               total=len(batches), desc="Polarized batches", ncols=80))

        print("  Aggregating results from workers...")
        
        # Aggregate results from all processes into the main instance
        for partial_uni, partial_bi, partial_tri in results:
            self.unigram_counts.update(partial_uni)
            
            for prev, transitions in partial_bi.items():
                self.bigram_transitions[prev].update(transitions)
                
            for prev_pair, transitions in partial_tri.items():
                self.trigram_transitions[prev_pair].update(transitions)

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
# ONLINE REINFORCEMENT LEARNING FEEDBACK BUFFER
# =====================================================================
class FeedbackBuffer:
    """Real-time human feedback for online RL [web:51][web:55][web:59]."""
    def __init__(self, vsa: VectorSymbolicArchitecture, buffer_size: int = 50):
        self.vsa = vsa
        self.buffer_size = buffer_size
        self.positive_tokens = deque(maxlen=buffer_size)  # "I like that"
        self.negative_tokens = deque(maxlen=buffer_size)  # "bad", "no"
        self.token_rewards = defaultdict(float)  # Running reward per token
        self.lock = threading.Lock()
    
    def add_positive_feedback(self, tokens: List[str], reward: float = 1.0):
        """Add positive feedback for recently generated tokens [web:55][web:59]."""
        with self.lock:
            for token in tokens:
                self.positive_tokens.append(token)
                self.token_rewards[token] += reward
                print(f"  [✓] Rewarded: {token} (+{reward:.2f})")
    
    def add_negative_feedback(self, tokens: List[str], penalty: float = -0.5):
        """Add negative feedback for recently generated tokens [web:55][web:59]."""
        with self.lock:
            for token in tokens:
                self.negative_tokens.append(token)
                self.token_rewards[token] += penalty
                print(f"  [✗] Penalized: {token} ({penalty:.2f})")
    
    def get_token_reward(self, token: str) -> float:
        """Get accumulated reward for a token [web:51][web:55]."""
        return self.token_rewards.get(token, 0.0)
    
    def get_similar_token_reward(self, token: str, top_k: int = 5) -> float:
        """Get reward from semantically similar tokens [web:55][web:59]."""
        if token not in self.vsa.codebook:
            return 0.0
        
        token_vec = self.vsa.codebook[token]
        similar_rewards = []
        
        # Find similar rewarded/penalized tokens
        for rewarded_token, reward in self.token_rewards.items():
            if rewarded_token in self.vsa.codebook:
                sim = self.vsa.similarity(token_vec, self.vsa.codebook[rewarded_token])
                if sim > 0.5:  # Similarity threshold
                    similar_rewards.append(reward * sim)
        
        if similar_rewards:
            return np.mean(sorted(similar_rewards, reverse=True)[:top_k])
        return 0.0

# =====================================================================
# RL-ENHANCED CATEGORY ERROR GENERATOR WITH FEEDBACK
# =====================================================================
class RLCategoryErrorGenerator:
    """Real-time RL with human feedback during generation [web:51][web:55][web:59]."""
    def __init__(self, vsa: VectorSymbolicArchitecture, transition_encoder: TransitionEncoder):
        self.vsa = vsa
        self.transition_encoder = transition_encoder
        self.semantic_categories = self._build_semantic_categories()
        self.feedback_buffer = FeedbackBuffer(vsa, buffer_size=100)
        self.generation_buffer = deque(maxlen=20)  # Track recent generations
    
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
        if candidate not in self.vsa.codebook:
            return 0.0
        
        candidate_vec = self.vsa.codebook[candidate]
        recent_context = context[-context_window:]
        similarities = []
        
        for ctx_token in recent_context:
            if ctx_token in self.vsa.codebook:
                ctx_vec = self.vsa.codebook[ctx_token]
                sim = self.vsa.similarity(candidate_vec, ctx_vec)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
        return np.mean(similarities)
    
    def _get_ngram_plausibility(self, candidate: str, context: List[str]) -> float:
        plausibility = 0.0
        
        if len(context) >= 1:
            last_token = context[-1]
            if last_token in self.transition_encoder.bigram_transitions:
                bigram_probs = self.transition_encoder.get_bigram_probabilities(last_token)
                if bigram_probs and candidate in bigram_probs:
                    plausibility += bigram_probs[candidate]
        
        if len(context) >= 2:
            last_two = tuple(context[-2:])
            if last_two in self.transition_encoder.trigram_transitions:
                trigram_probs = self.transition_encoder.get_trigram_probabilities(last_two)
                if trigram_probs and candidate in trigram_probs:
                    plausibility += trigram_probs[candidate] * 2.0
        
        return plausibility
    
    def apply_feedback_to_probs(self, probs: Dict[str, float], rl_weight: float = 2.0) -> Dict[str, float]:
        """Apply RL feedback to probability distribution [web:51][web:55]."""
        adjusted_probs = {}
        
        for token, prob in probs.items():
            # Direct reward
            direct_reward = self.feedback_buffer.get_token_reward(token)
            
            # Similar token reward (semantic generalization) [web:55]
            similar_reward = self.feedback_buffer.get_similar_token_reward(token)
            
            # Combined RL adjustment [web:51]
            total_reward = direct_reward + (similar_reward * 0.5)
            rl_boost = np.exp(total_reward * rl_weight)
            
            adjusted_probs[token] = prob * rl_boost
        
        # Renormalize
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {k: v/total for k, v in adjusted_probs.items()}
        
        return adjusted_probs
    
    def stream_generation(self, seed: List[str], max_tokens: int = 50, 
                         temperature: float = 1.0, 
                         error_rate: float = 0.0001,
                         plausibility_weight: float = 0.9,
                         rl_weight: float = 2.0):
        """Generate with real-time RL feedback [web:51][web:55][web:59]."""
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
                        plausibility_scores = {}
                        
                        for token in candidate_tokens:
                            vsa_plausibility = self._compute_semantic_plausibility(token, context)
                            ngram_plausibility = self._get_ngram_plausibility(token, context)
                            combined = (vsa_plausibility + ngram_plausibility) / 2.0
                            plausibility_scores[token] = combined
                        
                        probs = {}
                        total_counts = sum(self.transition_encoder.unigram_counts.values())
                        
                        for token in candidate_tokens:
                            base_prob = self.transition_encoder.unigram_counts.get(token, 1) / total_counts
                            plausibility = plausibility_scores.get(token, 0.0)
                            plausibility_boost = 1.0 + (plausibility * plausibility_weight * 10.0)
                            probs[token] = base_prob * plausibility_boost
                        
                            # APPLY RL FEEDBACK [web:51][web:55]
                            probs = self.apply_feedback_to_probs(probs, rl_weight=rl_weight)
                            
                            tokens = list(probs.keys())
                            prob_vals = np.array(list(probs.values()))
                            
                            if temperature > 0:
                                prob_vals = np.log(prob_vals + 1e-9) / temperature
                                prob_vals = np.exp(prob_vals)
                            prob_vals /= np.sum(prob_vals)
                            
                            next_token = np.random.choice(tokens, p=prob_vals)
                            self.generation_buffer.append(next_token)
                            yield next_token
                            context.append(next_token)
                            continue
            
            # Normal n-gram prediction with RL
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
                # APPLY RL FEEDBACK [web:51][web:55]
                probs = self.apply_feedback_to_probs(probs, rl_weight=rl_weight)
                
                tokens = list(probs.keys())
                prob_vals = np.array(list(probs.values()))
                if temperature > 0:
                    prob_vals = np.log(prob_vals + 1e-9) / temperature
                    prob_vals = np.exp(prob_vals)
                prob_vals /= np.sum(prob_vals)
                next_token = np.random.choice(tokens, p=prob_vals)
            
            self.generation_buffer.append(next_token)
            yield next_token
            context.append(next_token)

# =====================================================================
# MAIN ENTRYPOINT WITH INTERACTIVE FEEDBACK
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("2D POLARIZATION VSA + REAL-TIME RL FEEDBACK")
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
    
    print("\n[2] REAL-TIME RL TEXT GENERATION WITH FEEDBACK")
    print("-"*80)
    print("Commands during generation:")
    print("💬 FEEDBACK COMMANDS:")
    print("  POSITIVE: excellent, perfect, good, yes, nice, helpful, correct, 👍")
    print("  NEGATIVE: terrible, bad, wrong, no, poor, not helpful, 👎")
    print("  SPECIFIC: 'that word' / 'exactly that' (rewards last 2 tokens)")
    print("  DIRECT: 'more about fish' / 'avoid politics'")
    print("  STYLE: 'be more formal' / 'sound more casual'")
    print("  STATUS: 'show feedback' / 'reset'")
    print("-"*80)

    print("-"*80)
    
    rl_gen = RLCategoryErrorGenerator(vsa, trans_encoder)
    
    while True:
        user_input = input("\nUSER: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        # Check for feedback commands [web:55][web:59][web:61][web:65]
        
        # STRONG POSITIVE FEEDBACK (+1.5 reward) [web:61][web:66]
        if re.search(r'\b(excellent|perfect|amazing|brilliant|outstanding|superb|fantastic|wonderful|love it|exactly right|spot on|nailed it)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_positive_feedback(recent_tokens, reward=1.5)
            print("  💚 Strong positive feedback applied!")
            continue
        
        # MODERATE POSITIVE FEEDBACK (+1.0 reward) [web:61][web:65]
        if re.search(r'\b(i like that|good|yes|great|nice|helpful|useful|correct|right|better|improved|makes sense|that works|keep going|more like that)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_positive_feedback(recent_tokens, reward=1.0)
            print("  ✓ Positive feedback applied!")
            continue
        
        # MILD POSITIVE FEEDBACK (+0.5 reward) [web:65][web:66]
        if re.search(r'\b(okay|ok|fine|acceptable|decent|not bad|could work|sure|alright)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_positive_feedback(recent_tokens, reward=0.5)
            print("  ✓ Mild positive feedback applied!")
            continue
        
        # MILD NEGATIVE FEEDBACK (-0.5 penalty) [web:61][web:66]
        if re.search(r'\b(not quite|not really|meh|could be better|needs work|try again|not sure about that|off topic|confusing)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_negative_feedback(recent_tokens, penalty=-0.5)
            print("  ⚠ Mild negative feedback applied!")
            continue
        
        # MODERATE NEGATIVE FEEDBACK (-1.0 penalty) [web:61][web:65]
        if re.search(r'\b(bad|no|wrong|incorrect|poor|weak|not helpful|irrelevant|off|stop that|don\'t like|dislike)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_negative_feedback(recent_tokens, penalty=-1.0)
            print("  ✗ Negative feedback applied!")
            continue
        
        # STRONG NEGATIVE FEEDBACK (-1.5 penalty) [web:61][web:66]
        if re.search(r'\b(terrible|awful|horrible|completely wrong|nonsense|garbage|useless|inappropriate|offensive|unacceptable|hate it)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_negative_feedback(recent_tokens, penalty=-1.5)
            print("  ❌ Strong negative feedback applied!")
            continue
        
        # SPECIFIC TOKEN PRAISE (reward last 1-2 tokens more heavily) [web:65]
        if re.search(r'\b(that word|that phrase|exactly that|this|these words)\b', user_input.lower()):
            recent_tokens = list(rl_gen.generation_buffer)[-2:]  # Just last 2 tokens
            rl_gen.feedback_buffer.add_positive_feedback(recent_tokens, reward=2.0)
            print(f"  ⭐ Specific token praise: {' '.join(recent_tokens)}")
            continue
        
        # DIRECTION FEEDBACK (steer toward topic) [web:63][web:65]
        if re.search(r'\b(more about|talk about|focus on|tell me about|elaborate on|explain)\b', user_input.lower()):
            # Extract topic words after the command
            match = re.search(r'(?:more about|talk about|focus on|tell me about|elaborate on|explain)\s+(\w+(?:\s+\w+){0,2})', user_input.lower())
            if match:
                topic = match.group(1)
                topic_tokens = topic.split()
                rl_gen.feedback_buffer.add_positive_feedback(topic_tokens, reward=1.5)
                print(f"  🎯 Directing toward: {topic}")
            continue
        
        # AVOID FEEDBACK (steer away from topic) [web:61][web:65]
        if re.search(r'\b(stop talking about|don\'t mention|avoid|less about|not that|no more)\b', user_input.lower()):
            # Extract topic words after the command
            match = re.search(r'(?:stop talking about|don\'t mention|avoid|less about|no more)\s+(\w+(?:\s+\w+){0,2})', user_input.lower())
            if match:
                topic = match.group(1)
                topic_tokens = topic.split()
                rl_gen.feedback_buffer.add_negative_feedback(topic_tokens, penalty=-1.5)
                print(f"  🚫 Avoiding topic: {topic}")
            continue
        
        # STYLE FEEDBACK [web:63][web:65]
        if re.search(r'\b(be more|use more|write more|sound more)\s+(formal|casual|simple|complex|creative|direct|polite|technical|poetic)\b', user_input.lower()):
            match = re.search(r'(?:be more|use more|write more|sound more)\s+(\w+)', user_input.lower())
            if match:
                style = match.group(1)
                print(f"  🎨 Style preference noted: {style}")
                # Could be extended to track style preferences
            continue
        
        # THUMBS UP/DOWN EMOJI FEEDBACK [web:65][web:68]
        if '👍' in user_input or '👏' in user_input or '❤️' in user_input or '✅' in user_input:
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_positive_feedback(recent_tokens, reward=1.0)
            print("  👍 Positive emoji feedback!")
            continue
        
        if '👎' in user_input or '❌' in user_input or '⛔' in user_input or '🚫' in user_input:
            recent_tokens = list(rl_gen.generation_buffer)[-5:]
            rl_gen.feedback_buffer.add_negative_feedback(recent_tokens, penalty=-1.0)
            print("  👎 Negative emoji feedback!")
            continue
        
        # RESET/CLEAR FEEDBACK [web:61]
        if re.search(r'\b(reset|clear feedback|start over|forget that)\b', user_input.lower()):
            rl_gen.feedback_buffer = FeedbackBuffer(vsa, buffer_size=100)
            print("  🔄 Feedback buffer reset!")
            continue
        
        # SHOW FEEDBACK STATUS [web:65]
        if re.search(r'\b(show feedback|what have you learned|feedback status)\b', user_input.lower()):
            print("\n  📊 Current Feedback Status:")
            top_rewarded = sorted(rl_gen.feedback_buffer.token_rewards.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
            print("    Top rewarded tokens:")
            for token, reward in top_rewarded:
                if reward > 0:
                    print(f"      {token}: +{reward:.2f}")
            print()
            continue

        
        # Normal generation
        print("AI: ", end='', flush=True)
        for token in rl_gen.stream_generation(user_input.split(), 
                                             max_tokens=350, 
                                             temperature=0.7,
                                             error_rate=0.0001,
                                             plausibility_weight=0.9,
                                             rl_weight=2.0):
            print(token, end=' ', flush=True)
        print()

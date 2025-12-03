import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque, Counter
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import pickle
import os
import re
import multiprocessing as mp

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            default = args[0]
            super().__init__(default)
        else:
            super().__init__(int)

# =====================================================================
# 2D POLARIZATION VSA (NO LOCKS - MP SAFE)
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}

    def create_polarized_vector(self, normalize: bool = True) -> np.ndarray:
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
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        dim = len(vec_a) // 2
        
        fft_a_swapped = np.ones_like(fft_a, dtype=complex)
        fft_a_swapped[:dim] = fft_b[dim:]
        fft_a_swapped[dim:] = fft_b[:dim]
        
        result = np.fft.ifft(fft_a + fft_a_swapped)
        return np.real(result)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9)

    def add_to_codebook(self, symbol: str) -> np.ndarray:
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
# DYNAMIC LOW-PROB ENGINE (MP-SAFE)
# =====================================================================
class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        self.unigram_counts = Counter()  # MP-safe
        self.bigram_transitions = defaultdict(Counter)  # MP-safe
        self.trigram_transitions = defaultdict(Counter)  # MP-safe
        
        # Dynamic low-prob activation tracking
        self.low_prob_state = defaultdict(lambda: {'count': 0, 'max_cycles': 3, 'active': False})
        self.low_prob_threshold = 0.05
        self.activation_boost = 4.0
        self.deactivation_penalty = 0.3

    @staticmethod
    def _process_batch_static(batch: List[List[str]]) -> Tuple[Counter, dict, dict]:
        """MP-SAFE: No locks, pure functions."""
        local_unigrams = Counter()
        local_bigrams = defaultdict(Counter)
        local_trigrams = defaultdict(Counter)

        for sequence in batch:
            for i, token in enumerate(sequence):
                local_unigrams[token] += 1
                if i > 0:
                    prev = sequence[i-1]
                    local_bigrams[prev][token] += 1
                if i > 1:
                    prev2 = sequence[i-2]
                    prev1 = sequence[i-1]
                    local_trigrams[(prev2, prev1)][token] += 1
        
        # Convert to regular dicts for pickling
        return (local_unigrams, dict(local_bigrams), dict(local_trigrams))

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = None):
        """MP-SAFE training without locks."""
        if max_workers is None:
            max_workers = min(8, mp.cpu_count())
            
        print(f"Learning transitions (MP: {max_workers} workers)...")
        batches = [corpus[i:i+1000] for i in range(0, len(corpus), 1000)]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self._process_batch_static, batches), 
                               total=len(batches), desc="Processing", ncols=80))

        # Aggregate (thread-safe Counters)
        for uni, bi, tri in results:
            self.unigram_counts.update(uni)
            for prev, trans in bi.items():
                self.bigram_transitions[prev].update(trans)
            for prev_pair, trans in tri.items():
                self.trigram_transitions[prev_pair].update(trans)

        print(f"✓ Learned {len(self.unigram_counts)} tokens")

    def get_low_prob_bias(self, probs: Dict[str, float], context_key: Tuple[str, str], 
                         selected_token: str, selected_prob: float) -> Dict[str, float]:
        """DYNAMIC LOW-PROB CYCLING."""
        state = self.low_prob_state[context_key]
        
        # NEW LOW-PROB ACTIVATION
        if selected_prob < self.low_prob_threshold and not state['active']:
            state['count'] = 0
            state['active'] = True
        
        # BOOST CYCLE (3 times)
        elif state['active'] and state['count'] < state['max_cycles']:
            if selected_token in probs:
                probs[selected_token] *= self.activation_boost
                state['count'] += 1
                cycles_left = state['max_cycles'] - state['count']
        
        # CYCLE END → PENALTY → RESET
        elif state['active'] and state['count'] >= state['max_cycles']:
            state['active'] = False
            state['count'] = 0
            if selected_token in probs:
                probs[selected_token] *= self.deactivation_penalty
        
        # Renormalize
        total = sum(probs.values())
        if total > 0:
            return {k: v/total for k, v in probs.items()}
        return probs

    def get_unigram_probabilities(self) -> Dict[str, float]:
        total = sum(self.unigram_counts.values())
        if total == 0: return {}
        return {t: c/total for t, c in self.unigram_counts.items()}

    def get_bigram_probabilities(self, last_token: str) -> Optional[Dict[str, float]]:
        if last_token not in self.bigram_transitions: return None
        trans = self.bigram_transitions[last_token]
        total = sum(trans.values())
        return {t: c/total for t, c in trans.items()}

    def get_trigram_probabilities(self, last_two: Tuple[str, str]) -> Optional[Dict[str, float]]:
        if last_two not in self.trigram_transitions: return None
        trans = self.trigram_transitions[last_two]
        total = sum(trans.values())
        return {t: c/total for t, c in trans.items()}

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        data = {
            'unigram_counts': dict(self.unigram_counts),
            'bigram_transitions': dict(self.bigram_transitions),
            'trigram_transitions': dict(self.trigram_transitions),
            'low_prob_state': dict(self.low_prob_state)
        }
        for name, d in data.items():
            with open(os.path.join(directory, f"{name}.pkl"), 'wb') as f:
                pickle.dump(d, f)
        print(f"✓ Model saved to {directory}")

    def load_model(self, directory: str):
        for name in ['unigram_counts', 'bigram_transitions', 'trigram_transitions', 'low_prob_state']:
            path = os.path.join(directory, f"{name}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if name == 'unigram_counts':
                        self.unigram_counts = Counter(data)
                    elif name == 'low_prob_state':
                        self.low_prob_state = defaultdict(lambda: {'count': 0, 'max_cycles': 3, 'active': False}, data)
                    else:
                        setattr(self, name, defaultdict(Counter, data))
        print(f"✓ Model loaded from {directory}")

# =====================================================================
# SIMPLIFIED FEEDBACK
# =====================================================================
class FeedbackBuffer:
    def __init__(self, buffer_size: int = 50):
        self.token_rewards = defaultdict(float)
    
    def add_positive_feedback(self, tokens: List[str], reward: float = 1.0):
        for token in tokens:
            self.token_rewards[token] += reward
            print(f"  [✓] +{reward:.1f}: {token}")

    def add_negative_feedback(self, tokens: List[str], penalty: float = -0.5):
        for token in tokens:
            self.token_rewards[token] += penalty
            print(f"  [✗] {penalty:.1f}: {token}")

    def get_token_reward(self, token: str) -> float:
        return self.token_rewards.get(token, 0.0)

# =====================================================================
# DYNAMIC GENERATOR
# =====================================================================
class RLDynamicGenerator:
    def __init__(self, vsa: VectorSymbolicArchitecture, encoder: TransitionEncoder):
        self.vsa = vsa
        self.encoder = encoder
        self.feedback = FeedbackBuffer()
        self.generation_buffer = deque(maxlen=20)

    def stream_generation(self, seed: List[str], max_tokens: int = 100, temperature: float = 0.8,
                         rl_weight: float = 2.0):
        context = seed.copy()
        if not self.vsa.codebook:
            yield "No vocabulary!"
            return
            
        for _ in range(max_tokens):
            ctx_key = tuple(context[-2:]) if len(context) >= 2 else ('<pad>', '<pad>')
            
            # Multi-level prediction
            probs = (self.encoder.get_trigram_probabilities(ctx_key) or 
                    self.encoder.get_bigram_probabilities(context[-1] if context else '<pad>') or
                    self.encoder.get_unigram_probabilities())
            
            if not probs:
                next_token = list(self.vsa.codebook.keys())[0]
                yield next_token
                context.append(next_token)
                continue
            
            # RL feedback
            for token in probs:
                reward = self.feedback.get_token_reward(token)
                probs[token] *= np.exp(reward * rl_weight)
            
            # Sample & track actual prob
            tokens = list(probs.keys())
            prob_vals = np.array(list(probs.values()))
            prob_vals = np.log(prob_vals + 1e-8) / temperature
            prob_vals = np.exp(prob_vals)
            prob_vals /= prob_vals.sum()
            
            next_token = np.random.choice(tokens, p=prob_vals)
            selected_prob = probs[next_token]
            
            # DYNAMIC LOW-PROB CYCLING
            probs = self.encoder.get_low_prob_bias(probs, ctx_key, next_token, selected_prob)
            
            self.generation_buffer.append(next_token)
            yield next_token
            context.append(next_token)

# =====================================================================
# MAIN (MP-FIXED)
# =====================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # MP safety
    
    print("="*80)
    print("🔄 DYNAMIC LOW-PROB CYCLING (MP-FIXED)")
    print("BOOST ×3 → PENALTY → RESET")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=128)
    encoder = TransitionEncoder(vsa)
    
    choice = input("[N]ew/[L]oad? ").strip().lower()
    if choice == 'l':
        dir_path = input("Directory: ")
        vsa.load_codebook(os.path.join(dir_path, "codebook.pkl"))
        encoder.load_model(dir_path)
    else:
        filename = input("Corpus: ")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            sentences = [s.split() for s in text.split('.') if s.strip()]
            encoder.learn_transitions(sentences)
            
            for sentence in tqdm(sentences[:500], desc="Vocab"):
                for token in sentence:
                    vsa.add_to_codebook(token)
            
            dir_path = input("Save to: ")
            vsa.save_codebook(os.path.join(dir_path, "codebook.pkl"))
            encoder.save_model(dir_path)
        except FileNotFoundError:
            print("No corpus found, using minimal vocab")
            vsa.add_to_codebook("the")
            vsa.add_to_codebook("is")
    
    print("\n🎢 DYNAMIC LOW-PROB ACTIVE")
    print("Commands: good/bad | show | quit")
    
    gen = RLDynamicGenerator(vsa, encoder)
    
    while True:
        user_input = input("\nUSER: ").strip()
        if user_input.lower() in ['quit', 'q']:
            break
        
        if re.search(r'\b(good|great|yes|👍)\b', user_input.lower()):
            recent = list(gen.generation_buffer)[-5:]
            gen.feedback.add_positive_feedback(recent, 1.5)
            continue
        elif re.search(r'\b(bad|no|wrong|👎)\b', user_input.lower()):
            recent = list(gen.generation_buffer)[-5:]
            gen.feedback.add_negative_feedback(recent, -1.5)
            continue
        elif 'show' in user_input.lower():
            print("\n📊 LOW-PROB STATES:")
            active = {k:v for k,v in encoder.low_prob_state.items() if v['active'] or v['count']>0}
            for ctx, state in list(active.items())[:5]:
                print(f"  {ctx}: {state['count']}/{state['max_cycles']} ({'ACTIVE' if state['active'] else 'IDLE'})")
            continue
        
        print("AI:", end=' ')
        for token in gen.stream_generation(user_input.split(), max_tokens=600):
            print(token, end=' ')
        print()

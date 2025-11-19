"""
Unified VSA-MoE Architecture: Text Generation & Neuro-Symbolic Reasoning
Supports both unsupervised n-gram learning and supervised reasoning traces,
with interchangeable experts and goal-conditioned generation.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import pickle
import os
from scipy.stats import qmc
import math

class IntDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        if args:
            super().__init__(args[0])
        else:
            super().__init__(int)

# =====================================================================
# UNIFIED VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048, scramble: bool = True):
        self.dimensions = dimensions
        self.token_vectors: Dict[str, np.ndarray] = {}
        self.operator_vectors: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.sobol_engine = qmc.Sobol(d=min(dimensions, 21201), scramble=scramble)
        self._sobol_counter = 0
        self._sobol_buffer = None
        self._buffer_size = 0

    def _refill_sobol_buffer(self, n_points: int):
        power = max(1, math.ceil(math.log2(n_points)))
        self._buffer_size = 2 ** power
        uniform_samples = self.sobol_engine.random(self._buffer_size)
        self._sobol_buffer = qmc.scale(uniform_samples, l_bounds=-3, u_bounds=3)
        self._sobol_counter = 0

    def create_vector(self, normalize: bool = True) -> np.ndarray:
        if self._sobol_buffer is None or self._sobol_counter >= self._buffer_size:
            self._refill_sobol_buffer(1024)
        
        vec = self._sobol_buffer[self._sobol_counter].copy()
        self._sobol_counter += 1
        
        if normalize:
            vec = vec / np.linalg.norm(vec)
        return vec

    def bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        result = np.fft.ifft(fft_a * fft_b)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def add_token(self, token: str, is_operator: bool = False) -> np.ndarray:
        with self.lock:
            if is_operator:
                if token not in self.operator_vectors:
                    self.operator_vectors[token] = self.create_vector()
                return self.operator_vectors[token]
            else:
                if token not in self.token_vectors:
                    self.token_vectors[token] = self.create_vector()
                return self.token_vectors[token]

    def get_vector(self, token: str, is_operator: bool = False) -> Optional[np.ndarray]:
        if is_operator:
            return self.operator_vectors.get(token)
        return self.token_vectors.get(token)

    def save_codebook(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'tokens': self.token_vectors, 'operators': self.operator_vectors}, f)
        print(f"✓ Codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.token_vectors = data['tokens']
            self.operator_vectors = data['operators']
        print(f"✓ Codebook loaded from {filepath}")

# =====================================================================
# HYBRID TRANSITION ENCODER (N-GRAM + SUPERVISED)
# =====================================================================
class HybridTransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        
        # N-gram components (for text generation)
        self.bigram_vectors: Dict[Tuple[str, str], np.ndarray] = {}
        self.trigram_vectors: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.bigram_transitions = defaultdict(IntDefaultDict)
        self.trigram_transitions = defaultdict(IntDefaultDict)
        
        # Supervised components (for reasoning)
        self.valid_transitions = defaultdict(IntDefaultDict)
        self.invalid_transitions = defaultdict(IntDefaultDict)
        self.operator_bindings: Dict[Tuple[str, ...], np.ndarray] = {}
        self.state_vectors: Dict[Tuple[str, ...], np.ndarray] = {}
        
        self._cache_lock = threading.Lock()

    # ==================== N-GRAM METHODS ====================
    def encode_bigram(self, token1: str, token2: str):
        with self._cache_lock:
            self.bigram_transitions[token1][token2] += 1
            if (token1, token2) not in self.bigram_vectors:
                vec1 = -self.vsa.add_token(token1)
                vec2 = self.vsa.add_token(token2)
                self.bigram_vectors[(token1, token2)] = self.vsa.bind(vec1, vec2)

    def encode_trigram(self, token1: str, token2: str, token3: str):
        with self._cache_lock:
            self.trigram_transitions[(token1, token2)][token3] += 1
            if (token1, token2, token3) not in self.trigram_vectors:
                vec1 = self.vsa.add_token(token1)
                vec2 = self.vsa.add_token(token2)
                vec3 = self.vsa.add_token(token3)
                bound12 = self.vsa.bind(vec1, vec2)
                self.trigram_vectors[(token1, token2, token3)] = self.vsa.bind(bound12, vec3)

    def get_bigram_probabilities(self, last_token: str) -> Optional[Dict[str, float]]:
        if last_token not in self.bigram_transitions: return None
        candidates = self.bigram_transitions[last_token]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    def get_trigram_probabilities(self, last_two_tokens: Tuple[str, str]) -> Optional[Dict[str, float]]:
        if last_two_tokens not in self.trigram_transitions: return None
        candidates = self.trigram_transitions[last_two_tokens]
        total = sum(candidates.values())
        return {token: count / total for token, count in candidates.items()}

    # ==================== SUPERVISED METHODS ====================
    def extract_operators(self, sequence: List[str]) -> List[str]:
        operators = set(self.vsa.operator_vectors.keys())
        return [tok for tok in sequence if tok in operators]

    def encode_valid_transition(self, state: List[str], next_state: List[str], 
                               operators: Optional[List[str]] = None):
        """Supervised learning for valid reasoning steps"""
        state_key = tuple(state)
        next_key = tuple(next_state)
        
        with self._cache_lock:
            self.valid_transitions[state_key][next_key] += 1
            
            if state_key not in self.state_vectors:
                vectors = [self.vsa.token_vectors[tok] for tok in state if tok in self.vsa.token_vectors]
                if vectors:
                    self.state_vectors[state_key] = self.vsa.bundle(vectors)
            
            if operators:
                op_vectors = [self.vsa.operator_vectors[op] for op in operators if op in self.vsa.operator_vectors]
                if op_vectors:
                    self.operator_bindings[state_key] = self.vsa.bundle(op_vectors)

    def encode_invalid_transition(self, state: List[str], next_state: List[str]):
        """Track invalid transitions to avoid"""
        state_key = tuple(state)
        next_key = tuple(next_state)
        with self._cache_lock:
            self.invalid_transitions[state_key][next_key] += 1

    def get_valid_transitions(self, state: List[str]) -> Optional[Dict[Tuple[str, ...], float]]:
        """Retrieve valid next states"""
        state_key = tuple(state)
        if state_key not in self.valid_transitions:
            return None
        
        candidates = self.valid_transitions[state_key]
        total = sum(candidates.values())
        return {next_state: count / total for next_state, count in candidates.items()}

    def is_invalid(self, state: List[str], next_state: List[str]) -> bool:
        """Check if transition is known invalid"""
        state_key = tuple(state)
        next_key = tuple(next_state)
        return state_key in self.invalid_transitions and next_key in self.invalid_transitions[state_key]

    # ==================== BATCH PROCESSING ====================
    def _process_ngram_batch(self, sequences: List[List[str]]) -> Tuple[dict, dict, int, int]:
        """Process n-gram batches for text generation"""
        local_bigram = defaultdict(IntDefaultDict)
        local_trigram = defaultdict(IntDefaultDict)
        bigram_count, trigram_count = 0, 0
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                local_bigram[sequence[i]][sequence[i+1]] += 1
                bigram_count += 1
            for i in range(len(sequence) - 2):
                local_trigram[(sequence[i], sequence[i+1])][sequence[i+2]] += 1
                trigram_count += 1
        
        return local_bigram, local_trigram, bigram_count, trigram_count

    def _process_supervised_batch(self, problem_solutions: List[Tuple[List[str], List[List[str]]]]) -> Tuple[dict, dict, int, int]:
        """Process supervised reasoning batches"""
        local_valid = defaultdict(IntDefaultDict)
        local_invalid = defaultdict(IntDefaultDict)
        valid_count, invalid_count = 0, 0
        
        for problem, solution_trace in problem_solutions:
            for i in range(len(solution_trace) - 1):
                current_state = solution_trace[i]
                next_state = solution_trace[i + 1]
                
                operators = self.extract_operators(current_state + next_state)
                state_key = tuple(current_state)
                next_key = tuple(next_state)
                
                local_valid[state_key][next_key] += 1
                valid_count += 1
                
                # Generate negative examples
                if i < len(solution_trace) - 2:
                    wrong_next = solution_trace[i + 2]
                    local_invalid[tuple(current_state)][tuple(wrong_next)] += 1
                    invalid_count += 1
        
        return local_valid, local_invalid, valid_count, invalid_count

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        """Learn n-gram transitions (text generation mode)"""
        print("Learning n-gram transitions...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        
        all_local_bigrams = []
        all_local_trigrams = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(batches), desc="Processing n-gram batches", ncols=80) as pbar:
            for local_bigram, local_trigram, bg_count, tg_count in executor.map(self._process_ngram_batch, batches):
                all_local_bigrams.append(local_bigram)
                all_local_trigrams.append(local_trigram)
                pbar.update(1)
        
        print("Merging n-gram transitions...")
        for local_bigram in tqdm(all_local_bigrams, desc="Merging bigrams", ncols=80):
            for token1, transitions in local_bigram.items():
                for token2, count in transitions.items():
                    self.bigram_transitions[token1][token2] += count
        
        for local_trigram in tqdm(all_local_trigrams, desc="Merging trigrams", ncols=80):
            for token_pair, transitions in local_trigram.items():
                for token3, count in transitions.items():
                    self.trigram_transitions[token_pair][token3] += count
        
        total_bigrams = sum(len(v) for v in self.bigram_transitions.values())
        total_trigrams = sum(len(v) for v in self.trigram_transitions.values())
        print(f"  ✓ Learned {total_bigrams} unique bigram transitions.")
        print(f"  ✓ Learned {total_trigrams} unique trigram transitions.")

    def learn_from_supervised_traces(self, training_data: List[Tuple[List[str], List[List[str]]]], 
                                     max_workers: int = 8, batch_size: int = 50):
        """Learn from labeled problem-solution traces (reasoning mode)"""
        print("Learning supervised transitions...")
        batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
        
        all_local_valid = []
        all_local_invalid = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(batches), desc="Processing supervised batches", ncols=80) as pbar:
            for local_valid, local_invalid, v_count, i_count in executor.map(self._process_supervised_batch, batches):
                all_local_valid.append(local_valid)
                all_local_invalid.append(local_invalid)
                pbar.update(1)
        
        print("Merging valid transitions...")
        for local_valid in tqdm(all_local_valid, desc="Merging valid", ncols=80):
            for state_key, transitions in local_valid.items():
                for next_key, count in transitions.items():
                    self.valid_transitions[state_key][next_key] += count
        
        print("Merging invalid transitions...")
        for local_invalid in tqdm(all_local_invalid, desc="Merging invalid", ncols=80):
            for state_key, transitions in local_invalid.items():
                for next_key, count in transitions.items():
                    self.invalid_transitions[state_key][next_key] += count
        
        total_valid = sum(len(v) for v in self.valid_transitions.values())
        total_invalid = sum(len(v) for v in self.invalid_transitions.values())
        print(f"  ✓ Learned {total_valid} valid transitions.")
        print(f"  ✓ Learned {total_invalid} invalid transitions.")

    def save_model(self, directory: str, mode: str = 'both'):
        os.makedirs(directory, exist_ok=True)
        
        if mode in ['ngram', 'both']:
            with open(os.path.join(directory, "bigram_transitions.pkl"), 'wb') as f:
                pickle.dump(self.bigram_transitions, f)
            with open(os.path.join(directory, "trigram_transitions.pkl"), 'wb') as f:
                pickle.dump(self.trigram_transitions, f)
            with open(os.path.join(directory, "bigram_vectors.pkl"), 'wb') as f:
                pickle.dump(self.bigram_vectors, f)
            with open(os.path.join(directory, "trigram_vectors.pkl"), 'wb') as f:
                pickle.dump(self.trigram_vectors, f)
        
        if mode in ['supervised', 'both']:
            with open(os.path.join(directory, "valid_transitions.pkl"), 'wb') as f:
                pickle.dump(self.valid_transitions, f)
            with open(os.path.join(directory, "invalid_transitions.pkl"), 'wb') as f:
                pickle.dump(self.invalid_transitions, f)
            with open(os.path.join(directory, "state_vectors.pkl"), 'wb') as f:
                pickle.dump(self.state_vectors, f)
            with open(os.path.join(directory, "operator_bindings.pkl"), 'wb') as f:
                pickle.dump(self.operator_bindings, f)
        
        print(f"✓ Model saved in {directory} (mode: {mode})")

    def load_model(self, directory: str, mode: str = 'both'):
        if mode in ['ngram', 'both']:
            with open(os.path.join(directory, "bigram_transitions.pkl"), 'rb') as f:
                self.bigram_transitions = pickle.load(f)
            with open(os.path.join(directory, "trigram_transitions.pkl"), 'rb') as f:
                self.trigram_transitions = pickle.load(f)
            with open(os.path.join(directory, "bigram_vectors.pkl"), 'rb') as f:
                self.bigram_vectors = pickle.load(f)
            with open(os.path.join(directory, "trigram_vectors.pkl"), 'rb') as f:
                self.trigram_vectors = pickle.load(f)
        
        if mode in ['supervised', 'both']:
            with open(os.path.join(directory, "valid_transitions.pkl"), 'rb') as f:
                self.valid_transitions = pickle.load(f)
            with open(os.path.join(directory, "invalid_transitions.pkl"), 'rb') as f:
                self.invalid_transitions = pickle.load(f)
            with open(os.path.join(directory, "state_vectors.pkl"), 'rb') as f:
                self.state_vectors = pickle.load(f)
            with open(os.path.join(directory, "operator_bindings.pkl"), 'rb') as f:
                self.operator_bindings = pickle.load(f)
        
        print(f"✓ Model loaded from {directory} (mode: {mode})")

# =====================================================================
# UNIFIED MoE GENERATOR (TEXT GENERATION + PROBLEM SOLVING)
# =====================================================================
class UnifiedMoEGenerator:
    def __init__(self, vsa: VectorSymbolicArchitecture, encoder: HybridTransitionEncoder, mode: str = 'text'):
        self.vsa = vsa
        self.encoder = encoder
        self.mode = mode  # 'text' or 'reasoning'
        
        # Text generation components
        self.text_router = NGramRouter(vsa, expert_names=['bigram', 'trigram'])
        
        # Reasoning components
        self.reasoning_router = StrategicMoERouter(vsa, strategy_names=['forward_chain', 'analogical', 'constraint_solve'])
        
        self.goal_vector = None
        self.constraint_vectors = []
        self._context_cache = {}

    def set_mode(self, mode: str):
        """Switch between 'text' and 'reasoning' modes"""
        self.mode = mode
        print(f"✓ Mode switched to: {mode}")

    # ==================== TEXT GENERATION ====================
    def _get_context_vector_text(self, context: List[str], length: int) -> np.ndarray:
        cache_key = tuple(context[-length:])
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        vectors = [self.vsa.token_vectors[tok] for tok in cache_key if tok in self.vsa.token_vectors]
        if not vectors:
            vec = self.vsa.create_vector()
        else:
            vec = self.vsa.bundle(vectors)
        
        self._context_cache[cache_key] = vec
        return vec

    def stream_text_generation(self, seed: List[str], max_tokens: int = 50, temperature: float = 1.0):
        """Stream tokens for text generation mode"""
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return

        for _ in range(max_tokens):
            # Get n-gram probabilities
            bigram_probs = self.encoder.get_bigram_probabilities(context[-1]) if len(context) >= 1 else None
            trigram_probs = self.encoder.get_trigram_probabilities(tuple(context[-2:])) if len(context) >= 2 else None

            # Route between n-gram experts
            if trigram_probs:
                context_vec = self._get_context_vector_text(context, 2)
                routing_probs = self.text_router.route(context_vec)
            else:
                routing_probs = {'bigram': 1.0, 'trigram': 0.0}

            # Combine probabilities
            final_probs = defaultdict(float)
            if bigram_probs:
                for token, prob in bigram_probs.items(): 
                    final_probs[token] += routing_probs['bigram'] * prob
            if trigram_probs:
                for token, prob in trigram_probs.items(): 
                    final_probs[token] += routing_probs['trigram'] * prob

            if not final_probs:
                next_token = np.random.choice(list(self.vsa.token_vectors.keys()))
            else:
                tokens, probs = list(final_probs.keys()), np.array(list(final_probs.values()))
                if temperature > 0:
                    probs = np.log(probs + 1e-9) / temperature
                    probs = np.exp(probs)
                probs /= np.sum(probs)
                next_token = np.random.choice(tokens, p=probs)
            
            yield next_token
            context.append(next_token)

    # ==================== PROBLEM SOLVING ====================
    def set_goal(self, goal_tokens: List[str]):
        vectors = [self.vsa.token_vectors[tok] for tok in goal_tokens if tok in self.vsa.token_vectors]
        if vectors:
            self.goal_vector = self.vsa.bundle(vectors)
        else:
            self.goal_vector = self.vsa.create_vector()

    def add_constraint(self, constraint_tokens: List[str]):
        vectors = [self.vsa.token_vectors[tok] for tok in constraint_tokens if tok in self.vsa.token_vectors]
        if vectors:
            self.constraint_vectors.append(self.vsa.bundle(vectors))

    def _get_context_vector_reasoning(self, context: List[str], length: int) -> np.ndarray:
        cache_key = tuple(context[-length:]) + (tuple(self.goal_vector) if self.goal_vector is not None else ())
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        vectors = [self.vsa.token_vectors[tok] for tok in cache_key if tok in self.vsa.token_vectors]
        
        # Add operator bindings
        state_key = tuple(context[-length:])
        if state_key in self.encoder.operator_bindings:
            vectors.append(self.encoder.operator_bindings[state_key])
        
        # Add goal vector
        if self.goal_vector is not None:
            vectors.append(self.goal_vector)
        
        if not vectors:
            vec = self.vsa.create_vector()
        else:
            vec = self.vsa.bundle(vectors)
        
        self._context_cache[cache_key] = vec
        return vec

    def _check_constraints(self, state: List[str], next_state: List[str]) -> bool:
        if not self.constraint_vectors:
            return True
        
        state_vec = self.vsa.bundle([self.vsa.token_vectors[tok] for tok in state + next_state if tok in self.vsa.token_vectors])
        for constraint_vec in self.constraint_vectors:
            if self.vsa.similarity(state_vec, constraint_vec) < 0.3:
                return False
        return True

    def _get_goal_distance(self, state: List[str]) -> float:
        if self.goal_vector is None:
            return 0.0
        
        state_vec = self.vsa.bundle([self.vsa.token_vectors[tok] for tok in state if tok in self.vsa.token_vectors])
        return 1.0 - self.vsa.similarity(state_vec, self.goal_vector)

    def stream_problem_solving(self, seed: List[str], max_tokens: int = 50, temperature: float = 1.0):
        """Stream solution steps for reasoning mode"""
        context = seed.copy()
        if not context:
            print("Error: Seed context cannot be empty.")
            return

        for step in range(max_tokens):
            valid_moves = self.encoder.get_valid_transitions(context)
            
            if not valid_moves:
                print("[No valid moves from current state]")
                break
            
            # Get context vector and route to strategy
            context_vec = self._get_context_vector_reasoning(context, 3)
            strategy_probs = self.reasoning_router.route(context_vec, context)
            
            # Score candidates
            scored_candidates = []
            for next_state, base_prob in valid_moves.items():
                if self.encoder.is_invalid(context, list(next_state)):
                    continue
                
                if not self._check_constraints(context, list(next_state)):
                    continue
                
                # Compute goal distance improvement
                current_dist = self._get_goal_distance(context)
                next_dist = self._get_goal_distance(list(next_state))
                improvement = max(0, current_dist - next_dist)
                
                # Combine with strategy probability
                strategy_score = sum(strategy_probs.values())
                final_score = base_prob * strategy_score * (1 + improvement)
                
                if final_score > 0:
                    scored_candidates.append((next_state, final_score))
            
            if not scored_candidates:
                print("[All candidates failed constraints]")
                break
            
            # Sample next state
            states, scores = zip(*scored_candidates)
            scores = np.array(scores)
            if temperature > 0:
                scores = np.log(scores + 1e-9) / temperature
                scores = np.exp(scores)
            scores /= np.sum(scores)
            
            next_state = list(np.random.choice([s for s in states], p=scores))
            
            # Update router with outcome
            best_strategy = max(strategy_probs.items(), key=lambda x: x[1])[0]
            self.reasoning_router.update_strategy(best_strategy, context, success=True)
            
            yield next_state
            context = next_state

    # ==================== UNIFIED STREAM ====================
    def stream_generate(self, seed: List[str], max_tokens: int = 50, temperature: float = 1.0):
        """Unified generation method that respects mode"""
        if self.mode == 'text':
            yield from self.stream_text_generation(seed, max_tokens, temperature)
        elif self.mode == 'reasoning':
            yield from self.stream_problem_solving(seed, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# =====================================================================
# ROUTER CLASSES
# =====================================================================
class NGramRouter:
    def __init__(self, vsa: VectorSymbolicArchitecture, expert_names: List[str]):
        self.vsa = vsa
        self.expert_vectors = {name: self.vsa.create_vector() for name in expert_names}
        self._routing_cache = {}
        self._cache_lock = threading.Lock()

    def route(self, context_vector: np.ndarray) -> Dict[str, float]:
        context_key = tuple(np.round(context_vector, 4))
        with self._cache_lock:
            if context_key in self._routing_cache:
                return self._routing_cache[context_key]
        
        similarities = []
        expert_names = list(self.expert_vectors.keys())
        for name in expert_names:
            sim = self.vsa.similarity(context_vector, self.expert_vectors[name])
            similarities.append(sim)
            
        exp_sims = np.exp(np.array(similarities) - np.max(similarities))
        probabilities = exp_sims / np.sum(exp_sims)
        result = dict(zip(expert_names, probabilities))
        
        with self._cache_lock:
            self._routing_cache[context_key] = result
        return result

class StrategyExpert:
    def __init__(self, name: str, vsa: VectorSymbolicArchitecture):
        self.name = name
        self.vsa = vsa
        self.strategy_vector = self.vsa.create_vector()
        self.success_cache = defaultdict(float)
        self.attempt_cache = defaultdict(int)

    def update_success_rate(self, context: Tuple[str, ...], success: bool):
        key = context
        self.attempt_cache[key] += 1
        if success:
            self.success_cache[key] += 1

    def get_success_prob(self, context: Tuple[str, ...]) -> float:
        key = context
        attempts = self.attempt_cache[key]
        if attempts == 0:
            return 0.5
        return self.success_cache[key] / attempts

class StrategicMoERouter:
    def __init__(self, vsa: VectorSymbolicArchitecture, strategy_names: List[str]):
        self.vsa = vsa
        self.strategies = {name: StrategyExpert(name, vsa) for name in strategy_names}
        self._routing_cache = {}
        self._cache_lock = threading.Lock()

    def route(self, context_vector: np.ndarray, context: List[str]) -> Dict[str, float]:
        context_key = tuple(context[-3:])
        cache_key = tuple(np.round(context_vector, 4)) + context_key
        
        with self._cache_lock:
            if cache_key in self._routing_cache:
                return self._routing_cache[cache_key]
        
        similarities = []
        strategy_names = list(self.strategies.keys())
        for name in strategy_names:
            strategy_vec = self.strategies[name].strategy_vector
            sim = self.vsa.similarity(context_vector, strategy_vec)
            success_prob = self.strategies[name].get_success_prob(context_key)
            similarities.append(sim * success_prob)
        
        exp_sims = np.exp(np.array(similarities) - np.max(similarities))
        probabilities = exp_sims / np.sum(exp_sims)
        result = dict(zip(strategy_names, probabilities))
        
        with self._cache_lock:
            self._routing_cache[cache_key] = result
        return result

    def update_strategy(self, strategy_name: str, context: List[str], success: bool):
        if strategy_name in self.strategies:
            context_key = tuple(context[-3:])
            self.strategies[strategy_name].update_success_rate(context_key, success)

# =====================================================================
# COMPREHENSIVE EXAMPLES
# =====================================================================
def run_text_generation_examples():
    """Example 1: Creative text generation"""
    print("\n" + "="*80)
    print("EXAMPLE 1: CREATIVE TEXT GENERATION")
    print("="*80)
    
    # Initialize
    vsa = VectorSymbolicArchitecture(dimensions=256, scramble=True)
    encoder = HybridTransitionEncoder(vsa)
    generator = UnifiedMoEGenerator(vsa, encoder, mode='text')
    
    # Corpus: Sample text for training
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning transforms data into insights",
        "neural networks process information in parallel",
        "artificial intelligence augments human creativity",
        "vector symbolic architectures encode meaning algebraically"
    ]
    
    # Build corpus
    corpus = [text.split() for text in sample_texts]
    
    # Learn transitions
    print("Training on text corpus...")
    encoder.learn_transitions(corpus, max_workers=2, batch_size=5)
    
    # Build vocabulary
    for text in corpus:
        for token in text:
            vsa.add_token(token)
    
    # Save model
    encoder.save_model("models/text_model", mode='ngram')
    vsa.save_codebook("models/text_model/codebook.pkl")
    
    # Generate text
    print("\n--- Text Generation Examples ---")
    seeds = [
        ["the", "quick"],
        ["machine", "learning"],
        ["neural", "networks"],
        ["artificial", "intelligence"]
    ]
    
    for seed in seeds:
        print(f"\nSeed: {' '.join(seed)}")
        print("Generated: ", end='')
        for i, token in enumerate(generator.stream_text_generation(seed, max_tokens=10, temperature=0.7)):
            print(token, end=' ')
        print()

def run_mathematical_reasoning_examples():
    """Example 2: Mathematical equation solving"""
    print("\n" + "="*80)
    print("EXAMPLE 2: MATHEMATICAL EQUATION SOLVING")
    print("="*80)
    
    # Initialize
    vsa = VectorSymbolicArchitecture(dimensions=256, scramble=True)
    encoder = HybridTransitionEncoder(vsa)
    generator = UnifiedMoEGenerator(vsa, encoder, mode='reasoning')
    
    # Define operators
    operators = ['add', 'multiply', 'subtract', 'divide', 'equals', '+', '*', '-', '/', '=']
    for op in operators:
        vsa.add_token(op, is_operator=True)
    
    # Training data: (problem, solution_trace)
    training_data = [
        # Simple arithmetic
        (['x', '=', '2', '+', '3'], 
         [['x', '=', '5'], ['x', 'equals', '5']]),
        
        (['y', '=', '4', '*', '6'], 
         [['y', '=', '24'], ['y', 'equals', '24']]),
        
        (['z', '=', '10', '-', '7'], 
         [['z', '=', '3'], ['z', 'equals', '3']]),
        
        # Multi-step problems
        (['a', '=', '2', '+', '3', '*', '4'], 
         [['a', '=', '2', '+', '12'], ['a', '=', '14'], ['a', 'equals', '14']]),
        
        (['b', '=', '(', '5', '+', '3', ')', '*', '2'], 
         [['b', '=', '8', '*', '2'], ['b', '=', '16'], ['b', 'equals', '16']]),
    ]
    
    # Learn from supervised traces
    print("Training on mathematical reasoning traces...")
    encoder.learn_from_supervised_traces(training_data, max_workers=2, batch_size=5)
    
    # Build vocabulary
    for problem, solution in training_data:
        for tok in problem:
            vsa.add_token(tok)
        for step in solution:
            for tok in step:
                vsa.add_token(tok)
    
    # Save model
    encoder.save_model("models/math_model", mode='supervised')
    vsa.save_codebook("models/math_model/codebook.pkl")
    
    # Solve problems
    print("\n--- Mathematical Reasoning Examples ---")
    problems = [
        (['x', '=', '2', '+', '3'], ['x', 'equals', '5']),
        (['y', '=', '4', '*', '6'], ['y', 'equals', '24']),
        (['z', '=', '10', '-', '7'], ['z', 'equals', '3']),
    ]
    
    for problem, goal in problems:
        print(f"\nProblem: {' '.join(problem)}")
        generator.set_goal(goal)
        generator.add_constraint(['equals'])
        
        print("Solution: ")
        for i, step in enumerate(generator.stream_problem_solving(problem, max_tokens=5, temperature=0.5)):
            print(f"  Step {i+1}: {' '.join(step)}")
            if set(step) == set(goal):
                print("  ✓ Goal reached!")
                break

def run_logical_reasoning_examples():
    """Example 3: Logical deduction"""
    print("\n" + "="*80)
    print("EXAMPLE 3: LOGICAL DEDUCTION")
    print("="*80)
    
    # Initialize
    vsa = VectorSymbolicArchitecture(dimensions=256, scramble=True)
    encoder = HybridTransitionEncoder(vsa)
    generator = UnifiedMoEGenerator(vsa, encoder, mode='reasoning')
    
    # Define logical operators
    operators = ['implies', 'and', 'or', 'not', 'therefore', '=>', '&', '|', '~']
    for op in operators:
        vsa.add_token(op, is_operator=True)
    
    # Training data: logical proofs
    training_data = [
        # Modus ponens
        (['p', 'implies', 'q', 'p'], 
         [['p', 'implies', 'q', 'p', 'therefore', 'q'], ['q']]),
        
        # Syllogism
        (['all', 'men', 'are', 'mortal', 'socrates', 'is', 'a', 'man'], 
         [['all', 'men', 'are', 'mortal', 'socrates', 'is', 'a', 'man', 'therefore', 'socrates', 'is', 'mortal'],
          ['socrates', 'is', 'mortal']]),
        
        # Contrapositive
        (['if', 'rains', 'then', 'wet', 'not', 'wet'], 
         [['if', 'rains', 'then', 'wet', 'not', 'wet', 'therefore', 'not', 'rains'],
          ['not', 'rains']]),
    ]
    
    print("Training on logical reasoning traces...")
    encoder.learn_from_supervised_traces(training_data, max_workers=2, batch_size=5)
    
    # Build vocabulary
    for problem, solution in training_data:
        for tok in problem:
            vsa.add_token(tok)
        for step in solution:
            for tok in step:
                vsa.add_token(tok)
    
    # Save model
    encoder.save_model("models/logic_model", mode='supervised')
    vsa.save_codebook("models/logic_model/codebook.pkl")
    
    # Deduction examples
    print("\n--- Logical Reasoning Examples ---")
    deductions = [
        (['p', 'implies', 'q', 'p'], ['q']),
        (['all', 'men', 'are', 'mortal', 'socrates', 'is', 'a', 'man'], ['socrates', 'is', 'mortal']),
    ]
    
    for premise, conclusion in deductions:
        print(f"\nPremises: {' '.join(premise)}")
        generator.set_goal(conclusion)
        generator.add_constraint(['therefore'])
        
        print("Deduction: ")
        for i, step in enumerate(generator.stream_problem_solving(premise, max_tokens=5, temperature=0.5)):
            print(f"  Step {i+1}: {' '.join(step)}")
            if set(conclusion).issubset(set(step)):
                print("  ✓ Conclusion reached!")
                break

def run_code_generation_examples():
    """Example 4: Simple code generation"""
    print("\n" + "="*80)
    print("EXAMPLE 4: CODE GENERATION")
    print("="*80)
    
    # Initialize
    vsa = VectorSymbolicArchitecture(dimensions=256, scramble=True)
    encoder = HybridTransitionEncoder(vsa)
    generator = UnifiedMoEGenerator(vsa, encoder, mode='reasoning')
    
    # Define code operators
    operators = ['def', 'return', 'if', 'else', 'for', 'in', 'range', 'print', '(', ')', ':', 'import']
    for op in operators:
        vsa.add_token(op, is_operator=True)
    
    # Training data: simple Python functions
    training_data = [
        # Factorial function
        (['def', 'factorial', '(', 'n', ')'], 
         [['def', 'factorial', '(', 'n', ')', ':'],
          ['def', 'factorial', '(', 'n', ')', ':', 'if', 'n', '<=', '1', ':'],
          ['def', 'factorial', '(', 'n', ')', ':', 'if', 'n', '<=', '1', ':', 'return', '1'],
          ['def', 'factorial', '(', 'n', ')', ':', 'if', 'n', '<=', '1', ':', 'return', '1', 'else', ':'],
          ['def', 'factorial', '(', 'n', ')', ':', 'if', 'n', '<=', '1', ':', 'return', '1', 'else', ':', 'return', 'n', '*', 'factorial', '(', 'n', '-', '1', ')']]),
        
        # Sum function
        (['def', 'sum_list', '(', 'lst', ')'], 
         [['def', 'sum_list', '(', 'lst', ')', ':'],
          ['def', 'sum_list', '(', 'lst', ')', ':', 'total', '=', '0'],
          ['def', 'sum_list', '(', 'lst', ')', ':', 'total', '=', '0', 'for', 'x', 'in', 'lst', ':'],
          ['def', 'sum_list', '(', 'lst', ')', ':', 'total', '=', '0', 'for', 'x', 'in', 'lst', ':', 'total', '=', 'total', '+', 'x'],
          ['def', 'sum_list', '(', 'lst', ')', ':', 'total', '=', '0', 'for', 'x', 'in', 'lst', ':', 'total', '=', 'total', '+', 'x', 'return', 'total']]),
    ]
    
    print("Training on code generation traces...")
    encoder.learn_from_supervised_traces(training_data, max_workers=2, batch_size=5)
    
    # Build vocabulary
    for problem, solution in training_data:
        for tok in problem:
            vsa.add_token(tok)
        for step in solution:
            for tok in step:
                vsa.add_token(tok)
    
    # Save model
    encoder.save_model("models/code_model", mode='supervised')
    vsa.save_codebook("models/code_model/codebook.pkl")
    
    # Code generation
    print("\n--- Code Generation Examples ---")
    functions = [
        (['def', 'factorial', '(', 'n', ')'], ['return', 'n', '*', 'factorial', '(', 'n', '-', '1', ')']),
        (['def', 'sum_list', '(', 'lst', ')'], ['return', 'total']),
    ]
    
    for func_sig, goal in functions:
        print(f"\nFunction signature: {' '.join(func_sig)}")
        generator.set_goal(goal)
        generator.add_constraint(['def', 'return'])
        
        print("Generated code: ")
        solution = func_sig
        for i, step in enumerate(generator.stream_problem_solving(func_sig, max_tokens=10, temperature=0.5)):
            print(f"  {' '.join(step)}")
            solution = step
            if goal[0] in step:  # Simple completion check
                print("  ✓ Function completed!")
                break

def run_example():
    """Example 5: Switching between modes"""

    # Initialize unified system
    vsa = VectorSymbolicArchitecture(dimensions=256, scramble=True)
    encoder = HybridTransitionEncoder(vsa)
    generator = UnifiedMoEGenerator(vsa, encoder, mode='text')
    
    # Train on mixed data
    # Text data
    with open(input("Filename: "), encoding="utf-8") as f: 
        text_corpus = f.read().split(".")
    text_corpus = [(text + ".").split() for text in text_corpus]
    encoder.learn_transitions(text_corpus, max_workers=8, batch_size=50)
    
    # Reasoning data
    operators = ['add', 'equals']
    for op in operators:
        vsa.add_token(op, is_operator=True)
    
    reasoning_data = [
        (['x', '=', '2', '+', '3'], 
         [['x', '=', '5'], ['x', 'equals', '5']]),
    ]
    encoder.learn_from_supervised_traces(reasoning_data, max_workers=2, batch_size=5)
    
    # Build vocabulary
    for text in text_corpus:
        for tok in text:
            vsa.add_token(tok)
    for problem, solution in reasoning_data:
        for tok in problem:
            vsa.add_token(tok)
        for step in solution:
            for tok in step:
                vsa.add_token(tok)
    
    # Save combined model
    encoder.save_model("models/unified_model", mode='both')
    vsa.save_codebook("models/unified_model/codebook.pkl")
    
    # Demonstrate mode switching
    print("\n--- Mode Switching Demo ---")
    
    # Text generation mode
    generator.set_mode('text')
    print("\nText Generation Mode:")
    print("", end='')
    for token in generator.stream_generate(input("USER: ").split(), max_tokens=700, temperature=0.7):
        print(token, end=' ')
    print()

# =====================================================================
# MAIN ENTRYPOINT WITH EXAMPLES
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("UNIFIED VSA-MoE ARCHITECTURE: TEXT GENERATION & NEURO-SYMBOLIC REASONING")
    print("="*80)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Example 5: Mixed Mode
    try:
        run_example()
    except Exception as e:
        print(f"Mixed mode example failed: {e}")
    

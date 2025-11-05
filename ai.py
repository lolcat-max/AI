import numpy as np
from collections import Counter, defaultdict, deque
import os, zlib, pickle, hashlib
from typing import List, Dict, Tuple, Set, Optional
import time

# ================================================================
# 6NF DATA MODEL - IRREDUCIBLE DECOMPOSITION
# ================================================================

class TemporalAttribute:
    """Base class for temporal attributes in 6NF"""
    def __init__(self, entity_id: int, value: any, valid_from: float, valid_to: float = float('inf')):
        self.entity_id = entity_id
        self.value = value
        self.valid_from = valid_from
        self.valid_to = valid_to
    
    def is_valid_at(self, timestamp: float) -> bool:
        """Check if this attribute is valid at given timestamp"""
        return self.valid_from <= timestamp < self.valid_to
    
    def overlaps(self, start: float, end: float) -> bool:
        """Check if this attribute's validity overlaps given interval"""
        return not (self.valid_to <= start or self.valid_from >= end)


class SixNFDatastore:
    """6NF implementation: one table per independently-varying attribute"""
    
    def __init__(self):
        # Anchor table: just entity IDs
        self.entities = set()  # {entity_id, ...}
        
        # One table per attribute (irreducible decomposition)
        self.word_during = []          # [(entity_id, word, valid_from, valid_to), ...]
        self.frequency_during = []     # [(entity_id, frequency, valid_from, valid_to), ...]
        self.polynomial_coeff_during = [] # [(entity_id, coeff, valid_from, valid_to), ...]
        self.probability_during = []   # [(entity_id, probability, valid_from, valid_to), ...]
        self.context_during = []       # [(entity_id, context_tuple, valid_from, valid_to), ...]
        self.xor_state_during = []     # [(entity_id, xor_state, valid_from, valid_to), ...]
        
        # Indexes for faster temporal queries
        self.entity_word_index = defaultdict(list)  # entity_id -> [indices in word_during]
        self.entity_freq_index = defaultdict(list)
        self.entity_poly_index = defaultdict(list)
        self.entity_prob_index = defaultdict(list)
        self.entity_context_index = defaultdict(list)
        self.entity_xor_index = defaultdict(list)
        
        print("[6NF Datastore initialized]")
        print("  Tables: entities (anchor), word_during, frequency_during,")
        print("          polynomial_coeff_during, probability_during,")
        print("          context_during, xor_state_during")
    
    def create_entity(self) -> int:
        """Create new entity (anchor)"""
        entity_id = len(self.entities)
        self.entities.add(entity_id)
        return entity_id
    
    def insert_word(self, entity_id: int, word: str, timestamp: float):
        """Insert word attribute with temporal validity"""
        # Close previous interval for this entity's word
        for idx in self.entity_word_index[entity_id]:
            if self.word_during[idx][3] == float('inf'):
                # Close the interval
                old_record = self.word_during[idx]
                self.word_during[idx] = (old_record[0], old_record[1], old_record[2], timestamp)
        
        # Insert new interval
        new_idx = len(self.word_during)
        self.word_during.append((entity_id, word, timestamp, float('inf')))
        self.entity_word_index[entity_id].append(new_idx)
    
    def insert_frequency(self, entity_id: int, frequency: int, timestamp: float):
        """Insert frequency attribute with temporal validity"""
        for idx in self.entity_freq_index[entity_id]:
            if self.frequency_during[idx][3] == float('inf'):
                old_record = self.frequency_during[idx]
                self.frequency_during[idx] = (old_record[0], old_record[1], old_record[2], timestamp)
        
        new_idx = len(self.frequency_during)
        self.frequency_during.append((entity_id, frequency, timestamp, float('inf')))
        self.entity_freq_index[entity_id].append(new_idx)
    
    def insert_polynomial_coeff(self, entity_id: int, coeff: float, timestamp: float):
        """Insert polynomial coefficient with temporal validity"""
        for idx in self.entity_poly_index[entity_id]:
            if self.polynomial_coeff_during[idx][3] == float('inf'):
                old_record = self.polynomial_coeff_during[idx]
                self.polynomial_coeff_during[idx] = (old_record[0], old_record[1], old_record[2], timestamp)
        
        new_idx = len(self.polynomial_coeff_during)
        self.polynomial_coeff_during.append((entity_id, coeff, timestamp, float('inf')))
        self.entity_poly_index[entity_id].append(new_idx)
    
    def insert_probability(self, entity_id: int, probability: float, timestamp: float):
        """Insert probability with temporal validity"""
        for idx in self.entity_prob_index[entity_id]:
            if self.probability_during[idx][3] == float('inf'):
                old_record = self.probability_during[idx]
                self.probability_during[idx] = (old_record[0], old_record[1], old_record[2], timestamp)
        
        new_idx = len(self.probability_during)
        self.probability_during.append((entity_id, probability, timestamp, float('inf')))
        self.entity_prob_index[entity_id].append(new_idx)
    
    def insert_context(self, entity_id: int, context: Tuple[str, ...], timestamp: float):
        """Insert context with temporal validity"""
        for idx in self.entity_context_index[entity_id]:
            if self.context_during[idx][3] == float('inf'):
                old_record = self.context_during[idx]
                self.context_during[idx] = (old_record[0], old_record[1], old_record[2], timestamp)
        
        new_idx = len(self.context_during)
        self.context_during.append((entity_id, context, timestamp, float('inf')))
        self.entity_context_index[entity_id].append(new_idx)
    
    def insert_xor_state(self, entity_id: int, xor_state: int, timestamp: float):
        """Insert XOR state with temporal validity"""
        for idx in self.entity_xor_index[entity_id]:
            if self.xor_state_during[idx][3] == float('inf'):
                old_record = self.xor_state_during[idx]
                self.xor_state_during[idx] = (old_record[0], old_record[1], old_record[2], timestamp)
        
        new_idx = len(self.xor_state_during)
        self.xor_state_during.append((entity_id, xor_state, timestamp, float('inf')))
        self.entity_xor_index[entity_id].append(new_idx)
    
    def query_at_time(self, entity_id: int, timestamp: float) -> Dict:
        """Temporal join: reconstruct entity state at specific time"""
        result = {'entity_id': entity_id}
        
        # Query each attribute table independently
        for idx in self.entity_word_index[entity_id]:
            rec = self.word_during[idx]
            if rec[2] <= timestamp < rec[3]:
                result['word'] = rec[1]
                break
        
        for idx in self.entity_freq_index[entity_id]:
            rec = self.frequency_during[idx]
            if rec[2] <= timestamp < rec[3]:
                result['frequency'] = rec[1]
                break
        
        for idx in self.entity_poly_index[entity_id]:
            rec = self.polynomial_coeff_during[idx]
            if rec[2] <= timestamp < rec[3]:
                result['polynomial_coeff'] = rec[1]
                break
        
        for idx in self.entity_prob_index[entity_id]:
            rec = self.probability_during[idx]
            if rec[2] <= timestamp < rec[3]:
                result['probability'] = rec[1]
                break
        
        for idx in self.entity_context_index[entity_id]:
            rec = self.context_during[idx]
            if rec[2] <= timestamp < rec[3]:
                result['context'] = rec[1]
                break
        
        for idx in self.entity_xor_index[entity_id]:
            rec = self.xor_state_during[idx]
            if rec[2] <= timestamp < rec[3]:
                result['xor_state'] = rec[1]
                break
        
        return result
    
    def query_current(self, entity_id: int) -> Dict:
        """Query current (latest) state of entity"""
        return self.query_at_time(entity_id, time.time())
    
    def query_history(self, entity_id: int, attribute: str) -> List[Tuple]:
        """Query full history of a specific attribute"""
        if attribute == 'word':
            indices = self.entity_word_index[entity_id]
            return [self.word_during[i] for i in indices]
        elif attribute == 'frequency':
            indices = self.entity_freq_index[entity_id]
            return [self.frequency_during[i] for i in indices]
        elif attribute == 'polynomial_coeff':
            indices = self.entity_poly_index[entity_id]
            return [self.polynomial_coeff_during[i] for i in indices]
        elif attribute == 'probability':
            indices = self.entity_prob_index[entity_id]
            return [self.probability_during[i] for i in indices]
        elif attribute == 'context':
            indices = self.entity_context_index[entity_id]
            return [self.context_during[i] for i in indices]
        elif attribute == 'xor_state':
            indices = self.entity_xor_index[entity_id]
            return [self.xor_state_during[i] for i in indices]
        return []
    
    def get_statistics(self) -> Dict:
        """Get 6NF datastore statistics"""
        return {
            'entities': len(self.entities),
            'word_records': len(self.word_during),
            'frequency_records': len(self.frequency_during),
            'polynomial_records': len(self.polynomial_coeff_during),
            'probability_records': len(self.probability_during),
            'context_records': len(self.context_during),
            'xor_state_records': len(self.xor_state_during),
            'total_records': (len(self.word_during) + len(self.frequency_during) + 
                            len(self.polynomial_coeff_during) + len(self.probability_during) +
                            len(self.context_during) + len(self.xor_state_during))
        }


# ================================================================
# XOR BRANCHING ENGINE
# ================================================================
class XORBranchingEngine:
    """Uses XOR logic to create contingent paths between low/high prob selections"""
    
    def __init__(self, threshold: float = 0.5, xor_strength: float = 0.3):
        self.threshold = threshold
        self.xor_strength = xor_strength
        self.branch_history = deque(maxlen=50)
        self.xor_state = 0
        
        print(f"\n[XOR Branching Engine]")
        print(f"  Threshold: {self.threshold}")
        print(f"  XOR strength: {self.xor_strength}")
    
    def probability_to_bits(self, prob: float, num_bits: int = 8) -> int:
        """Convert probability to integer bit pattern"""
        scaled = int(prob * ((1 << num_bits) - 1))
        return scaled & ((1 << num_bits) - 1)
    
    def bits_to_probability(self, bits: int, num_bits: int = 8) -> float:
        """Convert bit pattern back to probability"""
        max_val = (1 << num_bits) - 1
        return (bits & max_val) / max_val
    
    def xor_branch(self, probs: np.ndarray, candidates: List[str]) -> np.ndarray:
        """Apply XOR branching to probability distribution"""
        if len(probs) == 0:
            return probs
        
        # Identify low and high probability candidates
        low_mask = probs < self.threshold
        high_mask = probs >= self.threshold
        
        num_low = np.sum(low_mask)
        num_high = np.sum(high_mask)
        
        if num_low == 0 or num_high == 0:
            return probs
        
        # Convert probabilities to bit patterns
        prob_bits = np.array([self.probability_to_bits(p) for p in probs])
        
        # XOR low prob with high prob to create contingent paths
        new_probs = probs.copy()
        
        low_indices = np.where(low_mask)[0]
        high_indices = np.where(high_mask)[0]
        
        # Create XOR pairings between low and high probability items
        for i, low_idx in enumerate(low_indices):
            high_idx = high_indices[i % len(high_indices)]
            
            # XOR the bit patterns
            low_bits = prob_bits[low_idx]
            high_bits = prob_bits[high_idx]
            xor_result = low_bits ^ high_bits
            
            # Update XOR state
            self.xor_state ^= xor_result
            
            # Convert XOR result to probability boost
            xor_prob = self.bits_to_probability(xor_result)
            
            # Boost low probability by XOR result
            boost = xor_prob * self.xor_strength
            new_probs[low_idx] += boost
            
            # Slightly reduce high probability (conservation)
            new_probs[high_idx] -= boost * 0.5
        
        # Ensure no negative probabilities
        new_probs = np.maximum(new_probs, 0.001)
        
        # Record branching event
        self.branch_history.append({
            'num_low': num_low,
            'num_high': num_high,
            'xor_state': self.xor_state,
            'avg_boost': np.mean(new_probs[low_mask] - probs[low_mask]) if num_low > 0 else 0
        })
        
        return new_probs
    
    def get_statistics(self) -> Dict:
        """Get branching statistics"""
        if not self.branch_history:
            return {'branches': 0}
        
        recent = list(self.branch_history)[-10:]
        return {
            'total_branches': len(self.branch_history),
            'xor_state': self.xor_state,
            'avg_low_candidates': np.mean([b['num_low'] for b in recent]),
            'avg_high_candidates': np.mean([b['num_high'] for b in recent]),
            'avg_boost': np.mean([b['avg_boost'] for b in recent])
        }


# ================================================================
# POLYNOMIAL REGRESSION ENGINE
# ================================================================
class InfinitePolynomialRegressor:
    """Performs continuous polynomial regression with iteration cap"""
    
    def __init__(self, initial_poly: np.ndarray, max_iterations: int = 1000):
        self.poly = initial_poly.astype(np.float64)
        self.max_iterations = max_iterations
        self.iteration = 0
        self.history = deque(maxlen=100)
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.velocity = np.zeros_like(self.poly)
        
        print(f"\n[Infinite Polynomial Regressor]")
        print(f"  Initial degree: {len(self.poly)-1}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Learning rate: {self.learning_rate}")
    
    def fit_sample(self, x: float, y_target: float) -> float:
        """Fit one sample using gradient descent with momentum"""
        if self.iteration >= self.max_iterations:
            return 0.0
        
        # Evaluate current polynomial
        y_pred = self.evaluate(x)
        error = y_target - y_pred
        
        # Compute gradients
        gradients = np.zeros_like(self.poly)
        x_powers = np.array([x ** i for i in range(len(self.poly))])
        gradients = -2 * error * x_powers
        
        # Update with momentum
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        self.poly += self.velocity
        
        # Store in history
        self.history.append((x, y_target, y_pred))
        self.iteration += 1
        
        return abs(error)
    
    def fit_batch(self, X: np.ndarray, Y: np.ndarray, epochs: int = 10):
        """Fit multiple samples in batch mode"""
        for epoch in range(epochs):
            if self.iteration >= self.max_iterations:
                break
            
            total_error = 0.0
            for x, y in zip(X, Y):
                error = self.fit_sample(x, y)
                total_error += error
            
            if total_error / len(X) < 1e-6:
                break
    
    def evaluate(self, x: float) -> float:
        """Evaluate polynomial at point x"""
        result = 0.0
        for i, coeff in enumerate(self.poly):
            result += coeff * (x ** i)
        return result
    
    def get_statistics(self) -> Dict:
        """Get regression statistics"""
        return {
            'iterations': self.iteration,
            'max_iterations': self.max_iterations,
            'remaining': self.max_iterations - self.iteration,
            'degree': len(self.poly) - 1,
            'history_size': len(self.history)
        }


# ================================================================
# SUPERPOLYNOMIAL CODEC
# ================================================================
class SuperpolynomialCodec:
    """Encode natural text data into polynomial representation"""
    
    @staticmethod
    def text_to_polynomial(text: str, chunk_size: int = 8) -> np.ndarray:
        """Convert text to polynomial coefficients"""
        print("\n[Encoding text to superpolynomial...]")
        
        text_bytes = text.encode('utf-8')
        compressed = zlib.compress(text_bytes, level=9)
        compressed_len = len(compressed)
        
        print(f"  Original: {len(text_bytes):,} bytes")
        print(f"  Compressed: {compressed_len:,} bytes")
        print(f"  Compression ratio: {len(text_bytes)/compressed_len:.2f}x")
        
        length_bytes = compressed_len.to_bytes(16, byteorder='little')
        full_data = length_bytes + compressed
        
        num_coeffs = (len(full_data) + chunk_size - 1) // chunk_size
        poly = np.zeros(num_coeffs, dtype=np.float64)
        
        for i in range(num_coeffs):
            start = i * chunk_size
            end = min(start + chunk_size, len(full_data))
            chunk = full_data[start:end]
            
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            coeff = int.from_bytes(chunk, byteorder='little', signed=False)
            poly[i] = float(coeff)
        
        print(f"  Polynomial degree: {len(poly)-1}")
        print(f"  Coefficients: {len(poly)}")
        
        return poly
    
    @staticmethod
    def polynomial_to_text(poly: np.ndarray, chunk_size: int = 8) -> str:
        """Decode polynomial back to text"""
        print("\n[Decoding superpolynomial to text...]")
        
        byte_chunks = []
        for coeff in poly:
            int_val = int(coeff)
            chunk = int_val.to_bytes(chunk_size, byteorder='little', signed=False)
            byte_chunks.append(chunk)
        
        full_data = b''.join(byte_chunks)
        
        compressed_len = int.from_bytes(full_data[:16], byteorder='little')
        compressed = full_data[16:16+compressed_len]
        
        try:
            text_bytes = zlib.decompress(compressed)
            text = text_bytes.decode('utf-8')
            print(f"  Decoded {len(text):,} characters")
            return text
        except Exception as e:
            print(f"  Decode error: {e}")
            return ""
    
    @staticmethod
    def dataset_to_polynomial(tokens: List[str], model: Dict, vocab: List[str]) -> np.ndarray:
        """Encode entire dataset (tokens + model) into polynomial"""
        print("\n[Encoding dataset to superpolynomial...]")
        
        dataset = {
            'tokens': tokens,
            'model': dict(model),
            'vocab': vocab,
            'stats': {
                'num_tokens': len(tokens),
                'vocab_size': len(vocab),
                'model_size': len(model)
            }
        }
        
        data_bytes = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(data_bytes, level=9)
        
        print(f"  Dataset size: {len(data_bytes):,} bytes")
        print(f"  Compressed: {len(compressed):,} bytes")
        print(f"  Compression: {len(data_bytes)/len(compressed):.2f}x")
        
        chunk_size = 8
        length_bytes = len(compressed).to_bytes(16, byteorder='little')
        full_data = length_bytes + compressed
        
        num_coeffs = (len(full_data) + chunk_size - 1) // chunk_size
        poly = np.zeros(num_coeffs, dtype=np.float64)
        
        for i in range(num_coeffs):
            start = i * chunk_size
            end = min(start + chunk_size, len(full_data))
            chunk = full_data[start:end]
            
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            coeff = int.from_bytes(chunk, byteorder='little', signed=False)
            poly[i] = float(coeff)
        
        print(f"  Polynomial degree: {len(poly)-1}")
        
        return poly
    
    @staticmethod
    def polynomial_to_dataset(poly: np.ndarray) -> Tuple[List[str], Dict, List[str]]:
        """Decode polynomial back to dataset"""
        print("\n[Decoding superpolynomial to dataset...]")
        
        chunk_size = 8
        byte_chunks = []
        
        for coeff in poly:
            int_val = int(coeff)
            chunk = int_val.to_bytes(chunk_size, byteorder='little', signed=False)
            byte_chunks.append(chunk)
        
        full_data = b''.join(byte_chunks)
        compressed_len = int.from_bytes(full_data[:16], byteorder='little')
        compressed = full_data[16:16+compressed_len]
        
        try:
            data_bytes = zlib.decompress(compressed)
            dataset = pickle.loads(data_bytes)
            
            print(f"  Tokens: {len(dataset['tokens']):,}")
            print(f"  Vocab: {len(dataset['vocab']):,}")
            print(f"  Model entries: {len(dataset['model']):,}")
            
            model = defaultdict(list, dataset['model'])
            return dataset['tokens'], model, dataset['vocab']
        except Exception as e:
            print(f"  Decode error: {e}")
            return [], defaultdict(list), []


# ================================================================
# GENERATOR WITH 6NF BACKEND
# ================================================================
class SixNFPolynomialGenerator:
    """Text generator using 6NF datastore for all state"""
    
    def __init__(self, tokens: List[str], model: Dict, 
                 xor_threshold: float = 0.5, xor_strength: float = 0.3,
                 regression_limit: int = 10000):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.vocab = list(set(tokens))
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        
        # 6NF datastore
        self.datastore = SixNFDatastore()
        
        # XOR branching engine
        self.xor_engine = XORBranchingEngine(xor_threshold, xor_strength)
        
        # Create entities for vocabulary
        self.word_entities = {}
        print(f"\n[Initializing 6NF vocabulary...]")
        
        freq_dist = Counter(tokens)
        timestamp = time.time()
        
        # Initialize with top words for demo
        top_words = self.vocab[:min(1000, len(self.vocab))]
        top_freqs = [freq_dist[word] for word in top_words]
        
        # Create polynomial regressor
        generative_poly = np.array(top_freqs[:min(200, len(top_freqs))], dtype=np.float64)
        if generative_poly.sum() > 0:
            generative_poly = generative_poly / generative_poly.sum()
        self.regressor = InfinitePolynomialRegressor(generative_poly, max_iterations=regression_limit)
        
        for word in top_words:
            entity_id = self.datastore.create_entity()
            self.word_entities[word] = entity_id
            
            # Insert attributes independently
            self.datastore.insert_word(entity_id, word, timestamp)
            self.datastore.insert_frequency(entity_id, freq_dist[word], timestamp)
            
            # Polynomial coefficient based on frequency
            coeff = freq_dist[word] / len(tokens)
            self.datastore.insert_polynomial_coeff(entity_id, coeff, timestamp)
            
            # Initial probability
            self.datastore.insert_probability(entity_id, 0.0, timestamp)
        
        print(f"  Initialized {len(self.word_entities)} word entities in 6NF")
        stats = self.datastore.get_statistics()
        print(f"  Total 6NF records: {stats['total_records']}")
    
    def word_to_seed(self, word: str) -> float:
        """Convert word to polynomial input"""
        if word in self.vocab_map:
            idx = self.vocab_map[word]
            return -1.0 + 2.0 * (idx / len(self.vocab))
        return 0.0
    
    def evaluate_poly(self, word: str, timestamp: float) -> float:
        """Evaluate polynomial using 6NF temporal query"""
        if word not in self.word_entities:
            return 0.0
        
        entity_id = self.word_entities[word]
        state = self.datastore.query_at_time(entity_id, timestamp)
        
        coeff = state.get('polynomial_coeff', 0.0)
        seed = self.word_to_seed(word)
        
        # Use regressor
        return self.regressor.evaluate(seed) * (1.0 + coeff)
    
    def update_probability(self, word: str, new_prob: float, timestamp: float):
        """Update probability attribute independently (6NF principle)"""
        if word in self.word_entities:
            entity_id = self.word_entities[word]
            self.datastore.insert_probability(entity_id, new_prob, timestamp)
    
    def update_polynomial_coeff(self, word: str, new_coeff: float, timestamp: float):
        """Update polynomial coefficient independently"""
        if word in self.word_entities:
            entity_id = self.word_entities[word]
            self.datastore.insert_polynomial_coeff(entity_id, new_coeff, timestamp)
    
    def generate(self, seed: str, length: int = 80, enable_xor: bool = True, 
                 train_every: int = 50) -> str:
        """Generate text with 6NF temporal tracking"""
        words = seed.split()[:2]
        while len(words) < 2:
            words.append(self.vocab[0] if self.vocab else "the")
        
        seed_key = tuple(words)
        if seed_key not in self.model:
            seed_key = self.keys[np.random.randint(len(self.keys))] if self.keys else tuple(words)
        
        output = list(seed_key)
        training_queue = deque(maxlen=1000)
        
        print(f"\n[Generating with 6NF + XOR + Regression...]")
        print(f"  XOR: {'ENABLED' if enable_xor else 'DISABLED'}")
        
        for step in range(length):
            timestamp = time.time()
            context = tuple(output[-2:])
            candidates = list(self.model.get(context, []))
            
            if not candidates:
                if self.keys:
                    context = self.keys[np.random.randint(len(self.keys))]
                    output.extend(list(context))
                continue
            
            # Compute probabilities using 6NF temporal queries
            probs = []
            for cand in candidates:
                poly_val = self.evaluate_poly(cand, timestamp)
                probs.append(max(0.001, poly_val))
            
            probs = np.array(probs)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(candidates)) / len(candidates)
            
            # Apply XOR branching
            if enable_xor:
                probs = self.xor_engine.xor_branch(probs, candidates)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
            
            # Select next word
            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            
            # Update probability in 6NF
            chosen_prob = probs[candidates.index(next_word)]
            self.update_probability(next_word, float(chosen_prob), timestamp)
            
            # Collect training sample
            chosen_seed = self.word_to_seed(next_word)
            training_queue.append((chosen_seed, 1.0))
            
            # Periodic regression training
            if step > 0 and step % train_every == 0 and len(training_queue) >= 32:
                samples = list(training_queue)[-32:]
                X = np.array([s[0] for s in samples])
                Y = np.array([s[1] for s in samples])
                self.regressor.fit_batch(X, Y, epochs=1)
                
                reg_stats = self.regressor.get_statistics()
                xor_stats = self.xor_engine.get_statistics()
                print(f"  Step {step}/{length} | Iter: {reg_stats['iterations']} | XOR: {xor_stats.get('xor_state', 0):08x}")
        
        print(f"[Complete]")
        
        # Show statistics
        stats = self.datastore.get_statistics()
        print(f"\n[6NF Statistics]")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        
        if enable_xor:
            xor_stats = self.xor_engine.get_statistics()
            print(f"\n[XOR Statistics]")
            for key, val in xor_stats.items():
                print(f"  {key}: {val}")
        
        reg_stats = self.regressor.get_statistics()
        print(f"\n[Regression Statistics]")
        for key, val in reg_stats.items():
            print(f"  {key}: {val}")
        
        return " ".join(output)
    
    def show_word_history(self, word: str):
        """Show temporal history of a word's attributes"""
        if word not in self.word_entities:
            print(f"Word '{word}' not in 6NF datastore")
            return
        
        entity_id = self.word_entities[word]
        print(f"\n[6NF History for '{word}' (entity {entity_id})]")
        
        for attr in ['word', 'frequency', 'polynomial_coeff', 'probability']:
            history = self.datastore.query_history(entity_id, attr)
            print(f"\n{attr}_during:")
            for rec in history[-5:]:
                print(f"  {rec}")


# ================================================================
# BUILD MODEL
# ================================================================
def build_ngram(tokens: List[str], n: int = 2) -> Dict[Tuple[str, ...], List[str]]:
    m = defaultdict(list)
    L = len(tokens)
    if n <= 0 or L <= n:
        return dict(m)
    for i in range(L - n):
        key = tuple(tokens[i:i + n])
        m[key].append(tokens[i + n])
    return dict(m)


# ================================================================
# MAIN
# ================================================================
def main():
    print("="*70)
    print("POLYNOMIAL TEXT GENERATOR WITH 6NF + XOR + REGRESSION")
    print("="*70)
    
    poly_file = "model.poly.npy"
    
    # Check if polynomial file exists
    if os.path.exists(poly_file):
        print(f"\n[Found polynomial file: {poly_file}]")
        use_poly = input("Load from polynomial? (y/n, default y): ").strip().lower()
        
        if use_poly != 'n':
            print("\n[Loading from superpolynomial...]")
            poly = np.load(poly_file)
            codec = SuperpolynomialCodec()
            toks, model, vocab = codec.polynomial_to_dataset(poly)
            
            if len(toks) == 0:
                print("Failed to decode polynomial, loading from text instead...")
                poly = None
            else:
                print(f"\n[Successfully decoded from polynomial]")
        else:
            poly = None
    else:
        poly = None
    
    # Load from text file if no polynomial
    if poly is None:
        path = input("\nEnter text file: ").strip()
        if not os.path.exists(path):
            print("File not found")
            return
        
        print("\n[Loading corpus...]")
        text = open(path, encoding="utf-8").read()
        toks = text.lower().split()
        print(f"  Loaded {len(toks):,} tokens")
        
        print("\n[Building model...]")
        model = build_ngram(toks, 2)
        vocab = list(set(toks))
        
        # Offer to save as polynomial
        save_poly = input("\nSave as superpolynomial? (y/n, default y): ").strip().lower()
        if save_poly != 'n':
            codec = SuperpolynomialCodec()
            
            # Save text polynomial
            text_poly = codec.text_to_polynomial(text)
            np.save("text.poly.npy", text_poly)
            print(f"  Saved text polynomial: text.poly.npy")
            
            # Save dataset polynomial
            dataset_poly = codec.dataset_to_polynomial(toks, model, vocab)
            np.save(poly_file, dataset_poly)
            print(f"  Saved dataset polynomial: {poly_file}")
            print(f"\n  Next time, load instantly from polynomial!")
    
    # Get parameters
    try:
        xor_threshold = float(input("\nXOR threshold (0.0-1.0, default 0.5): ").strip() or "0.5")
        xor_strength = float(input("XOR strength (0.0-1.0, default 0.3): ").strip() or "0.3")
        regression_limit = int(input("Regression limit (default 10000): ").strip() or "10000")
    except:
        xor_threshold = 0.5
        xor_strength = 0.3
        regression_limit = 10000
    
    # Initialize 6NF generator
    generator = SixNFPolynomialGenerator(
        toks, model, 
        xor_threshold=xor_threshold,
        xor_strength=xor_strength,
        regression_limit=regression_limit
    )
    
    print("\n" + "="*70)
    print("Ready for generation with 6NF + XOR + Regression!")
    print("Commands: generate, history <word>, stats, decode, exit")
    print("="*70)
    
    while True:
        cmd = input("\n> ").strip().split()
        if not cmd:
            continue
        
        if cmd[0] == "exit":
            break
        elif cmd[0] == "generate":
            seed = input("seed: ")
            try:
                length = int(input("length (default 80): ").strip() or "80")
                enable_xor_input = input("Enable XOR? (y/n, default y): ").strip().lower()
                enable_xor = enable_xor_input != 'n'
            except:
                length = 80
                enable_xor = True
            
            output = generator.generate(seed, length=length, enable_xor=enable_xor)
            print("\n" + "─"*70)
            print(output)
            print("─"*70)
        
        elif cmd[0] == "history" and len(cmd) > 1:
            generator.show_word_history(cmd[1])
        
        elif cmd[0] == "stats":
            stats = generator.datastore.get_statistics()
            print("\n[6NF Datastore Statistics]")
            for key, val in stats.items():
                print(f"  {key}: {val}")
        
        elif cmd[0] == "decode":
            poly_path = input("Polynomial file (.npy): ").strip()
            if os.path.exists(poly_path):
                poly = np.load(poly_path)
                codec = SuperpolynomialCodec()
                
                if "text" in poly_path:
                    decoded_text = codec.polynomial_to_text(poly)
                    print("\n" + "="*70)
                    print(decoded_text[:500])  # First 500 chars
                    print("="*70)
                else:
                    toks_dec, model_dec, vocab_dec = codec.polynomial_to_dataset(poly)
                    print(f"\n  Decoded tokens: {len(toks_dec):,}")
                    print(f"  Decoded vocab: {len(vocab_dec):,}")
                    print(f"  Decoded model: {len(model_dec):,} entries")
            else:
                print("File not found")

if __name__ == "__main__":
    main()

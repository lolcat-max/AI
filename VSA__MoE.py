
# =====================================================================
# COMPLETE INTEGRATED SYSTEM: VSA + TRANSITIONS + ACTIVE MEMORY
# =====================================================================

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import pickle
import os
import re

# =====================================================================
# MEMORY SYSTEM (REASONED INFERENCE)
# =====================================================================

class MemoryType(Enum):
    DATAPOINT = "raw"
    PATTERN = "pattern"
    RULE = "rule"
    ASSOCIATION = "link"
    META_INSIGHT = "insight"

@dataclass
class MemoryNode:
    """Single memory unit with reasoning trace."""
    memory_id: str
    memory_type: MemoryType
    content: Dict
    created_at: int = 0
    activation_count: int = 0
    confidence: float = 1.0
    reasoning_trace: List[str] = field(default_factory=list)
    associations: Set[str] = field(default_factory=set)
    
    def record_activation(self):
        self.activation_count += 1
    
    def add_reasoning(self, reason: str):
        self.reasoning_trace.append(reason)
    
    def link_to(self, node_id: str):
        self.associations.add(node_id)

class ActiveMemory:
    """Inverts datapoints into patterns → rules → meta-insights."""
    
    def __init__(self, max_nodes: int = 5000):
        self.nodes: Dict[str, MemoryNode] = {}
        self.max_nodes = max_nodes
        self.datapoint_registry = defaultdict(list)
        self.pattern_index = defaultdict(set)
        self.rule_index = defaultdict(set)
        self.association_graph = defaultdict(set)
        self.association_strength = defaultdict(float)
        
        self.inversion_depth = 0
        self.pattern_emergence_threshold = 3
        
        self.datapoint_count = 0
        self.pattern_count = 0
        self.rule_count = 0
        self.activations_log = deque(maxlen=100)
    
    def record_datapoint(self, context_key: Tuple, token: str, prob: float,
                        cycle_state: Dict, timestamp: int) -> str:
        """Record observation and invert to patterns/rules."""
        node_id = f"dp_{self.datapoint_count}"
        self.datapoint_count += 1
        
        node = MemoryNode(
            memory_id=node_id,
            memory_type=MemoryType.DATAPOINT,
            content={
                'context': context_key,
                'token': token,
                'probability': prob,
                'cycle_state': cycle_state.copy(),
                'is_low_prob': prob < 0.45,
                'in_boost_phase': cycle_state.get('active', False),
                'boost_cycle': cycle_state.get('count', 0)
            },
            created_at=timestamp
        )
        
        node.add_reasoning(f"Observed: {token} in {context_key} @ p={prob:.3f}")
        self.nodes[node_id] = node
        self._attempt_pattern_inversion(node_id, context_key, prob, cycle_state)
        
        return node_id
    
    def _attempt_pattern_inversion(self, node_id: str, context_key: Tuple,
                                   prob: float, cycle_state: Dict):
        """Invert datapoints into patterns."""
        
        # LOW-PROB TRIGGER PATTERN
        if prob < 0.45 and not cycle_state['active']:
            pattern_sig = ("low_prob_trigger", context_key)
            self.pattern_index[pattern_sig].add(node_id)
            self.datapoint_registry[context_key].append(node_id)
            
            if len(self.pattern_index[pattern_sig]) >= self.pattern_emergence_threshold:
                self._generate_rule_from_pattern(pattern_sig, "LOW_PROB_ACTIVATION_RULE")
        
        # BOOST CYCLING PATTERN
        if cycle_state['active'] and cycle_state['count'] > 0:
            pattern_sig = ("boost_cycle", cycle_state['count'])
            self.pattern_index[pattern_sig].add(node_id)
            
            if len(self.pattern_index[pattern_sig]) >= self.pattern_emergence_threshold:
                self._generate_rule_from_pattern(pattern_sig, "BOOST_CYCLE_RULE")
        
        # TOKEN REPETITION IN CONTEXT
        repeat_sig = ("token_repeat", context_key, self.nodes[node_id].content['token'])
        self.pattern_index[repeat_sig].add(node_id)
        
        if len(self.pattern_index[repeat_sig]) >= self.pattern_emergence_threshold:
            self._generate_rule_from_pattern(repeat_sig, "TOKEN_CONTEXT_AFFINITY_RULE")
    
    def _generate_rule_from_pattern(self, pattern_sig: Tuple, rule_name: str):
        """Convert patterns into actionable rules."""
        
        rule_id = f"rule_{self.rule_count}_{rule_name}"
        self.rule_count += 1
        
        pattern_nodes = self.pattern_index[pattern_sig]
        evidences = [self.nodes[nid].content for nid in pattern_nodes]
        
        avg_prob = np.mean([e['probability'] for e in evidences])
        boost_phases = sum(1 for e in evidences if e['in_boost_phase'])
        low_prob_count = sum(1 for e in evidences if e['is_low_prob'])
        
        rule_node = MemoryNode(
            memory_id=rule_id,
            memory_type=MemoryType.RULE,
            content={
                'rule_name': rule_name,
                'pattern_signature': str(pattern_sig),
                'evidence_count': len(pattern_nodes),
                'avg_probability': float(avg_prob),
                'boost_phase_frequency': float(boost_phases / max(len(evidences), 1)),
                'low_prob_frequency': float(low_prob_count / max(len(evidences), 1)),
                'contexts_affected': list(set(e['context'] for e in evidences))
            },
            confidence=min(1.0, len(pattern_nodes) / 10.0)
        )
        
        rule_node.add_reasoning(f"Induced from {len(pattern_nodes)} obs: {rule_name}")
        
        for nid in pattern_nodes:
            rule_node.link_to(nid)
            self.association_graph[rule_id].add(nid)
            self.association_graph[nid].add(rule_id)
        
        self.nodes[rule_id] = rule_node
        self.rule_index[rule_name].add(rule_id)
    
    def fire_reasoned_inference(self, context_key: Tuple, current_prob: float) -> Dict:
        """Reasoned prediction using active memory."""
        
        matching_rules = []
        reasoning_steps = []
        
        for rule_name, rule_ids in self.rule_index.items():
            for rule_id in rule_ids:
                rule = self.nodes[rule_id]
                rule_contexts = rule.content.get('contexts_affected', [])
                
                if context_key in rule_contexts:
                    matching_rules.append((rule_id, rule))
                    reasoning_steps.append(f"Rule '{rule_name}' matches context")
        
        inference = {
            'should_boost': False,
            'confidence': 0.0,
            'matched_rules': len(matching_rules),
            'reasoning': reasoning_steps,
            'activated_nodes': []
        }
        
        if not matching_rules:
            inference['reasoning'].append("No applicable rules")
            return inference
        
        boost_votes = 0
        total_confidence = 0.0
        
        for rule_id, rule_node in matching_rules:
            rule_node.record_activation()
            inference['activated_nodes'].append(rule_id)
            
            if "LOW_PROB_ACTIVATION" in rule_node.content['rule_name']:
                if current_prob < 0.45:
                    boost_votes += 1
            elif "BOOST_CYCLE" in rule_node.content['rule_name']:
                if rule_node.content['boost_phase_frequency'] > 0.7:
                    boost_votes += 1
            elif "TOKEN_CONTEXT_AFFINITY" in rule_node.content['rule_name']:
                boost_votes += 0.5
            
            total_confidence += rule_node.confidence
        
        inference['confidence'] = min(1.0, total_confidence / max(len(matching_rules), 1))
        inference['should_boost'] = boost_votes > len(matching_rules) / 2
        
        return inference
    
    def summarize_reasoning_state(self) -> Dict:
        return {
            'total_nodes': len(self.nodes),
            'datapoints': self.datapoint_count,
            'rules': self.rule_count,
            'active_rules': {name: len(ids) for name, ids in self.rule_index.items() if ids}
        }

# =====================================================================
# VSA (VECTOR SYMBOLIC ARCHITECTURE)
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

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / ((np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9)

    def add_to_codebook(self, symbol: str) -> np.ndarray:
        if symbol not in self.codebook:
            self.codebook[symbol] = self.create_polarized_vector()
        return self.codebook[symbol]

# =====================================================================
# TRANSITION ENCODER
# =====================================================================

class TransitionEncoder:
    def __init__(self, vsa: VectorSymbolicArchitecture):
        self.vsa = vsa
        self.unigram_counts = Counter()
        self.bigram_transitions = defaultdict(Counter)
        self.trigram_transitions = defaultdict(Counter)
        self.low_prob_state = defaultdict(lambda: {'count': 0, 'max_cycles': 300, 'active': False})
        self.low_prob_threshold = 0.45
        self.activation_boost = 14.0
        self.deactivation_penalty = 0.13

    @staticmethod
    def _process_batch_static(batch: List[List[str]]) -> Tuple[Counter, dict, dict]:
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
        
        return (local_unigrams, dict(local_bigrams), dict(local_trigrams))

    def learn_transitions(self, corpus: List[List[str]], max_workers: int = None):
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        batches = [corpus[i:i+1000] for i in range(0, len(corpus), 1000)]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._process_batch_static, batches))

        for uni, bi, tri in results:
            self.unigram_counts.update(uni)
            for prev, trans in bi.items():
                self.bigram_transitions[prev].update(trans)
            for prev_pair, trans in tri.items():
                self.trigram_transitions[prev_pair].update(trans)

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

    def get_low_prob_bias(self, probs: Dict[str, float], context_key: Tuple[str, str],
                         selected_token: str, selected_prob: float) -> Dict[str, float]:
        state = self.low_prob_state[context_key]
        
        if selected_prob < self.low_prob_threshold and not state['active']:
            state['count'] = 0
            state['active'] = True
        elif state['active'] and state['count'] < state['max_cycles']:
            if selected_token in probs:
                probs[selected_token] *= self.activation_boost
                state['count'] += 1
        elif state['active'] and state['count'] >= state['max_cycles']:
            state['active'] = False
            state['count'] = 0
            if selected_token in probs:
                probs[selected_token] *= self.deactivation_penalty
        
        total = sum(probs.values())
        if total > 0:
            return {k: v/total for k, v in probs.items()}
        return probs

# =====================================================================
# INTEGRATED GENERATOR WITH MEMORY
# =====================================================================

class RLDynamicGeneratorWithMemory:
    def __init__(self, vsa: VectorSymbolicArchitecture, encoder: TransitionEncoder,
                 memory: ActiveMemory):
        self.vsa = vsa
        self.encoder = encoder
        self.memory = memory
        self.generation_buffer = deque(maxlen=20)
        self.inference_trace = deque(maxlen=50)
        
        self.inference_count = 0
        self.memory_guided_decisions = 0

    def stream_generation(self, seed: List[str], max_tokens: int = 100,
                         temperature: float = 0.8, rl_weight: float = 2.0,
                         use_memory_reasoning: bool = True) -> List[str]:
        
        context = seed.copy()
        generated = []
        
        if not self.vsa.codebook:
            return ["[NO_VOCAB]"]
        
        for step in range(max_tokens):
            ctx_key = tuple(context[-2:]) if len(context) >= 2 else ('<pad>', '<pad>')
            
            probs = (self.encoder.get_trigram_probabilities(ctx_key) or
                    self.encoder.get_bigram_probabilities(context[-1] if context else '<pad>') or
                    self.encoder.get_unigram_probabilities())
            
            if not probs:
                next_token = list(self.vsa.codebook.keys())[0]
                generated.append(next_token)
                context.append(next_token)
                continue
            
            # MEMORY REASONING
            memory_inference = None
            if use_memory_reasoning:
                memory_inference = self.memory.fire_reasoned_inference(ctx_key, min(probs.values()))
                self.inference_count += 1
                
                if memory_inference['should_boost'] and memory_inference['confidence'] > 0.2:
                    for token in probs:
                        probs[token] *= 1.5
                    self.memory_guided_decisions += 1
                    trace_msg = f"Memory boost ({memory_inference['matched_rules']} rules)"
                else:
                    trace_msg = f"Memory inference (conf={memory_inference['confidence']:.2f})"
                
                self.inference_trace.append({
                    'step': step,
                    'context': ctx_key,
                    'inference': trace_msg,
                    'rules_fired': memory_inference['matched_rules']
                })
            
            # Sample
            tokens = list(probs.keys())
            prob_vals = np.array(list(probs.values()))
            prob_vals = np.log(prob_vals + 1e-8) / temperature
            prob_vals = np.exp(prob_vals)
            prob_vals /= prob_vals.sum()
            
            next_token = np.random.choice(tokens, p=prob_vals)
            selected_prob = probs.get(next_token, 0.5)
            
            # Record in memory
            cycle_state = self.encoder.low_prob_state[ctx_key]
            self.memory.record_datapoint(ctx_key, next_token, selected_prob, cycle_state, step)
            
            # Apply dynamic cycling
            probs = self.encoder.get_low_prob_bias(probs, ctx_key, next_token, selected_prob)
            
            self.generation_buffer.append(next_token)
            generated.append(next_token)
            context.append(next_token)
        
        return generated


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    print("="*80)
    print("INTEGRATED: VSA + TRANSITIONS + ACTIVE MEMORY REASONING")
    print("="*80)
    
    vsa = VectorSymbolicArchitecture(dimensions=128)
    encoder = TransitionEncoder(vsa)
    memory = ActiveMemory()
    
    # Training corpus
    print("\n📚 Training on synthetic corpus...")
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
                corpus = f.read().split(".")
    corpusB = []
    for sentence in corpus:
        corpusB.append(sentence.split())
    encoder.learn_transitions(corpusB, max_workers=1)
    
    for sent in corpus:
        for token in sent:
            vsa.add_to_codebook(token)
    
    print(f"✓ Vocabulary: {len(vsa.codebook)} tokens")
    print(f"✓ Transitions: {len(encoder.unigram_counts)} unigrams")
    
    # Generation with memory
    gen = RLDynamicGeneratorWithMemory(vsa, encoder, memory)
    
    print("\n" + "="*80)
    print("GENERATION WITH ACTIVE MEMORY GUIDANCE")
    print("="*80)
    while True:
        seed = input("USER: ").split()
        print(f"\n🌱 Seed: {seed}\n")
        
        generated = gen.stream_generation(
            seed,
            max_tokens=800,
            temperature=0.7,
            use_memory_reasoning=True
        )
        
        print(f"Generated: {' '.join(generated)}\n")
    

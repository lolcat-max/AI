# =====================================================================
# NEURAL-AUGMENTED TEXT GENERATION WITH DATASET LOGGING
# Logs synaptic weights to CSV and applies them to generation pipeline
# =====================================================================
KB_len = -1

import numpy as np
import random
import serial
import time
import csv
import os
from datetime import datetime
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------
# NEURAL SERIAL INTERFACE WITH DATASET LOGGING
# ---------------------------------------------------------------------
class NeuralInterface:
    """
    Streams synaptic weights from Arduino and logs to CSV dataset.
    Can replay from existing dataset for reproducible experiments.
    """
    def __init__(self, port='COM3', baud=115200, log_file='neural_dataset.csv', replay_mode=False):
        self.port = port
        self.baud = baud
        self.log_file = log_file
        self.replay_mode = replay_mode
        self.active = False
        self.last_val = 0.5
        self.csv_writer = None
        self.csv_file_handle = None
        self.replay_data = []
        self.replay_index = 0
        
        if replay_mode:
            self._init_replay_mode()
        else:
            self._init_live_mode()

    def _init_live_mode(self):
        """Initialize live serial connection and CSV logging"""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.02)
            self.ser.reset_input_buffer()
            self.active = True
            print(f"✓ NEURAL LINK ESTABLISHED ON {self.port}")
            
            # Initialize CSV logging
            file_exists = os.path.exists(self.log_file)
            self.csv_file_handle = open(self.log_file, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file_handle)
            
            if not file_exists:
                # Write header
                self.csv_writer.writerow(['timestamp', 'unix_time', 'synaptic_weight', 'smoothed_weight'])
                print(f"✓ DATASET LOGGING TO: {self.log_file}")
            else:
                print(f"✓ APPENDING TO EXISTING DATASET: {self.log_file}")
                
        except Exception as e:
            print(f"! NEURAL LINK FAILED (Simulation mode): {e}")
            self.active = False

    def _init_replay_mode(self):
        """Load existing dataset for replay"""
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                self.replay_data = list(reader)
            print(f"✓ REPLAY MODE: Loaded {len(self.replay_data)} samples from {self.log_file}")
            self.active = True
        except Exception as e:
            print(f"! REPLAY MODE FAILED: {e}")
            self.active = False

    def read_synaptic_weight(self):
        """
        Returns normalized float 0.0-1.0 representing neural activation.
        Logs to CSV in live mode, replays from CSV in replay mode.
        """
        if not self.active:
            # Simulation fallback
            sim_val = 0.5 + 0.1 * np.sin(time.time())
            return sim_val
        
        if self.replay_mode:
            # Replay from dataset
            if self.replay_index >= len(self.replay_data):
                self.replay_index = 0  # Loop
            
            row = self.replay_data[self.replay_index]
            self.replay_index += 1
            self.last_val = float(row['smoothed_weight'])
            return self.last_val
        
        else:
            # Live reading with logging
            if self.ser.in_waiting > 0:
                try:
                    lines = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore').splitlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            val = float(last_line)
                            # EMA Smoothing
                            self.last_val = 0.7 * self.last_val + 0.3 * val
                            
                            # Log to CSV
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            unix_time = time.time()
                            self.csv_writer.writerow([timestamp, unix_time, val, self.last_val])
                            self.csv_file_handle.flush()  # Ensure immediate write
                            
                            return self.last_val
                except ValueError:
                    pass
            return self.last_val

    def close(self):
        """Clean shutdown"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        print("✓ Neural interface closed cleanly")


# ---------------------------------------------------------------------
# DATASET PREPROCESSOR
# ---------------------------------------------------------------------
class NeuralDatasetProcessor:
    """
    Analyzes logged neural datasets and computes statistics.
    Used for offline analysis and normalization.
    """
    def __init__(self, dataset_path='neural_dataset.csv'):
        self.dataset_path = dataset_path
        self.data = None
        
    def load_dataset(self):
        """Load and parse CSV dataset"""
        try:
            import pandas as pd
            self.data = pd.read_csv(self.dataset_path)
            print(f"✓ Loaded {len(self.data)} neural samples")
            return True
        except Exception as e:
            print(f"! Dataset load failed: {e}")
            return False
    
    def compute_statistics(self):
        """Compute statistical features for dataset analysis"""
        if self.data is None:
            return None
        
        stats = {
            'mean': self.data['smoothed_weight'].mean(),
            'std': self.data['smoothed_weight'].std(),
            'min': self.data['smoothed_weight'].min(),
            'max': self.data['smoothed_weight'].max(),
            'median': self.data['smoothed_weight'].median(),
            'samples': len(self.data)
        }
        return stats
    
    def normalize_to_range(self, target_min=0.0, target_max=1.0):
        """Z-score normalization with range mapping"""
        if self.data is None:
            return None
        
        weights = self.data['smoothed_weight'].values
        normalized = (weights - weights.min()) / (weights.max() - weights.min())
        scaled = normalized * (target_max - target_min) + target_min
        return scaled


# ---------------------------------------------------------------------
# ENTROPY BLOCKATION
# ---------------------------------------------------------------------
class EntropyBlocker:
    def __init__(self, base_threshold=2.5, aggressive_mode=True, top_k_preserve=20, suppression_factor=0.1):
        self.base_threshold = base_threshold
        self.aggressive_mode = aggressive_mode
        self.top_k_preserve = top_k_preserve
        self.suppression_factor = suppression_factor
        self.entropy_history = deque(maxlen=100)
        self.block_count = 0
        self.total_count = 0

    def compute_entropy(self, probs: dict) -> float:
        if not probs:
            return 0.0
        p_array = np.array(list(probs.values()), dtype=np.float64)
        p_array = p_array[p_array > 0]
        if len(p_array) == 0:
            return 0.0
        entropy = -np.sum(p_array * np.log2(p_array + 1e-12))
        return float(entropy)

    def compute_adaptive_threshold(self) -> float:
        if len(self.entropy_history) < 10:
            return self.base_threshold
        recent = np.array(list(self.entropy_history))
        mean_entropy = recent.mean()
        std_entropy = recent.std()
        adaptive = mean_entropy - 0.5 * std_entropy
        return np.clip(adaptive, self.base_threshold * 0.5, self.base_threshold * 2.0)

    def block_entropy(self, probs: dict, ctx_len: int) -> dict:
        if not probs or len(probs) <= 1:
            return probs
        self.total_count += 1
        entropy = self.compute_entropy(probs)
        self.entropy_history.append(entropy)
        threshold = self.compute_adaptive_threshold()
        
        if ctx_len > 50:
            threshold *= 0.85
        elif ctx_len > 100:
            threshold *= 0.7
        
        if entropy <= threshold:
            return probs
        
        self.block_count += 1
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        
        if self.aggressive_mode:
            blocked = {}
            for i, (tok, p) in enumerate(sorted_items):
                if i < self.top_k_preserve:
                    blocked[tok] = p * (1.5 ** (self.top_k_preserve - i) / self.top_k_preserve)
                else:
                    blocked[tok] = p * self.suppression_factor
        else:
            cutoff_prob = sorted_items[min(self.top_k_preserve, len(sorted_items)-1)][1]
            blocked = {}
            for tok, p in probs.items():
                if p >= cutoff_prob:
                    blocked[tok] = p * 1.2
                else:
                    blocked[tok] = p * self.suppression_factor
        
        Z = sum(blocked.values())
        if Z > 0:
            blocked = {k: v / Z for k, v in blocked.items()}
        else:
            return probs
        
        new_entropy = self.compute_entropy(blocked)
        if new_entropy >= entropy * 0.95:
            return probs
        
        return blocked

    def get_stats(self) -> dict:
        block_rate = self.block_count / max(1, self.total_count)
        avg_entropy = np.mean(list(self.entropy_history)) if self.entropy_history else 0.0
        return {
            "block_rate": block_rate,
            "block_count": self.block_count,
            "total_count": self.total_count,
            "avg_entropy": avg_entropy,
            "current_threshold": self.compute_adaptive_threshold()
        }


# ---------------------------------------------------------------------
# KNOWLEDGE SUBSET CLUSTERING
# ---------------------------------------------------------------------
class KnowledgeSubsets:
    def __init__(self, n_clusters=8, cluster_size=50):
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.clusters = {}
        self.token_to_cluster = {}
        self.cluster_centers = {}

    def build_clusters(self, vocab, vsa):
        if not vocab:
            return
        vecs = {tok: vsa.vec(tok) for tok in vocab}
        clusters = defaultdict(list)
        centers = list(vocab)[:self.n_clusters]
        self.cluster_centers = {i: tok for i, tok in enumerate(centers)}

        for _ in range(3):
            clusters.clear()
            self.token_to_cluster.clear()
            for tok in vocab:
                best_cid, best_sim = 0, -1.0
                v = vecs[tok]
                for cid in range(self.n_clusters):
                    c_tok = self.cluster_centers[cid]
                    sim = float(np.dot(v, vecs[c_tok]))
                    if sim > best_sim:
                        best_sim, best_cid = sim, cid
                clusters[best_cid].append(tok)
                self.token_to_cluster[tok] = best_cid

            for cid in range(self.n_clusters):
                toks = clusters[cid]
                if not toks:
                    continue
                best_center, best_avg = toks[0], -1.0
                for cand in toks:
                    cv = vecs[cand]
                    sims = np.exp(cv/1000)
                    avg_sim = np.mean(sims) if sims[0] else -1.0
                    if avg_sim > best_avg:
                        best_avg, best_center = avg_sim, cand
                self.cluster_centers[cid] = best_center

        self.clusters = {cid: set(toks[:self.cluster_size]) for cid, toks in clusters.items() if toks}
        print(f"✓ Built {len(self.clusters)} knowledge clusters")

    def get_context_clusters(self, ctx):
        if not ctx:
            return set()
        active = set()
        for tok in ctx[-80:]:
            cid = self.token_to_cluster.get(tok)
            if cid is not None:
                active.add(np.exp(cid/1000))
        return active

    def boost_cluster_neighbors(self, probs: dict, active_clusters: set, boost_strength=0.4) -> dict:
        if not active_clusters or not probs:
            return probs
        boosted = dict(probs)
        cluster_bonus = {}
        
        for cid in active_clusters:
            cluster_toks = self.clusters.get(cid, set())
            if not cluster_toks:
                cluster_bonus[cid] = 0.0
                continue
            bonus = sum(probs.get(tok, 0.0) for tok in cluster_toks)
            cluster_bonus[cid] = bonus / len(cluster_toks)
        
        for tok, p in probs.items():
            cid = self.token_to_cluster.get(tok)
            if cid in active_clusters:
                bonus = cluster_bonus.get(np.exp(p/1000), 0.0)
                factor = 1.0 + boost_strength * bonus
                boosted[tok] = p * factor
        
        Z = sum(boosted.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in boosted.items()}


# ---------------------------------------------------------------------
# UNKNOWN CONTEXT PAIR CLUSTERER
# ---------------------------------------------------------------------
class UnknownContextClusterer:
    def __init__(self, min_count=3):
        self.min_count = min_count
        self.pair_counts = Counter()
        self.user_clusters = {}
        self.pair_to_cluster = {}
        self.next_cluster_id = 0

    def observe_pair(self, prev_tok, tok, known: bool):
        if not known and prev_tok is not None and tok is not None:
            self.pair_counts[(prev_tok, tok)] += 1

    def active_cluster_boost(self, probs: dict, ctx) -> dict:
        if not probs or len(ctx) < 1:
            return probs
        prev_tok = ctx[-1]
        boosted = dict(probs)
        active_cids = set()
        
        for tok in probs.keys():
            pair = (prev_tok, tok)
            cid = self.pair_to_cluster.get(pair)
            if cid is not None:
                active_cids.add(cid)
        
        if not active_cids:
            return probs
        
        cluster_activation = {cid: 0.0 for cid in active_cids}
        for pair, cid in self.pair_to_cluster.items():
            if cid not in active_cids:
                continue
            ptok, ntok = pair
            if ptok == prev_tok and ntok in probs:
                cluster_activation[cid] += np.mean(probs[ntok])
        
        max_act = max(cluster_activation.values()) if cluster_activation else 0.0
        if max_act <= 0:
            return probs
        
        base_strength = 0.3
        for tok, p in probs.items():
            pair = (prev_tok, tok)
            cid = self.pair_to_cluster.get(pair)
            if cid in active_cids:
                act = cluster_activation.get(cid, 0.0) / max_act
                factor = 1.0 + base_strength * act
                boosted[tok] = p * factor
        
        Z = sum(boosted.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in boosted.items()}


# ---------------------------------------------------------------------
# DOME SPIRAL PROBABILITY MODULATION
# ---------------------------------------------------------------------
class DomeSpiral:
    def __init__(self, n_spirals=70, dome_height=2.0, decay=0.15):
        self.n_spirals = n_spirals
        self.dome_height = dome_height
        self.decay = decay
        self.token_angles = {}
        self.spiral_phase = 0.0

    def assign_angle(self, tok: str) -> float:
        if tok not in self.token_angles:
            h = (hash(tok) & 0xFFFFFFFF) / 0xFFFFFFFF
            self.token_angles[tok] = h * 2 * np.pi
        return self.token_angles[tok]

    def dome_z(self, r: float) -> float:
        return self.dome_height * (1 - r**2)

    def spiral_weight(self, theta: float, r: float, spiral_idx: int) -> float:
        spiral_base = 2 * np.pi * spiral_idx / self.n_spirals
        spiral_theta = spiral_base + 3.0 * r + self.spiral_phase
        delta = np.arctan2(np.sin(theta - spiral_theta), np.cos(theta - spiral_theta))
        sigma2 = 0.32
        weight = np.exp(-delta**2 / sigma2)
        weight *= (0.5 + 0.5 * self.dome_z(r) / self.dome_height)
        return weight

    def compute_spiral_weights(self, probs: dict, ctx_len: int) -> dict:
        if not probs:
            return {}
        sorted_toks = sorted(probs.items(), key=lambda x: -x[1])
        n = len(sorted_toks)
        weights = {}
        for rank, (tok, p) in enumerate(sorted_toks):
            r = (rank + 1) / (n + 1)
            theta = self.assign_angle(tok)
            spiral_sum = 0.0
            for si in range(self.n_spirals):
                spiral_sum += self.spiral_weight(theta, r, si)
            weights[tok] = (spiral_sum / self.n_spirals) * (self.decay ** r)
        return weights

    def modulate_probs(self, probs: dict, ctx_len: int, blend: float = 0.3) -> dict:
        if not probs:
            return {}
        self.spiral_phase += (0.1 * np.pi)
        spiral_w = self.compute_spiral_weights(probs, ctx_len)
        sw_total = sum(spiral_w.values())
        if sw_total > 0:
            spiral_w = {k: v / sw_total for k, v in spiral_w.items()}
        result = {tok: (1 - blend) * p + blend * spiral_w.get(tok, 0.0)
                  for tok, p in probs.items()}
        Z = sum(result.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in result.items()}


# ---------------------------------------------------------------------
# STOCHASTIC CARDINAL ORDERING
# ---------------------------------------------------------------------
class StochasticCardinalOrder:
    def __init__(self, inversion_strength=1.7, noise_scale=0.05):
        self.inv_str = inversion_strength
        self.noise_scale = noise_scale
        self.attribution_history = deque(maxlen=500)
        self.cardinal_memory = {}

    def cardinal_rank(self, probs: dict) -> dict:
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        return {
            tok: {
                "original_prob": p,
                "cardinal": i + 1,
                "inverted_weight": (i + 1) ** self.inv_str,
            }
            for i, (tok, p) in enumerate(sorted_items)
        }

    def stochastic_perturb(self, ranks: dict) -> dict:
        perturbed = {}
        for tok, data in ranks.items():
            g = -np.log(-np.log(np.random.uniform(0.001, 0.999)))
            perturbed[tok] = data["inverted_weight"] * (1 + self.noise_scale * g)
        return perturbed

    def attribution_transform(self, probs: dict, ctx_attribution: float = 0.5) -> dict:
        if not probs:
            return {}
        ranks = self.cardinal_rank(probs)
        perturbed = self.stochastic_perturb(ranks)
        inv_factor = 0.3 + 0.7 * ctx_attribution
        pert_sum = sum(perturbed.values()) or 1.0

        final_scores = {}
        for tok, pert_weight in perturbed.items():
            orig_p = ranks[tok]["original_prob"]
            blended = (1 - inv_factor) * orig_p + inv_factor * (pert_weight / pert_sum)
            final_scores[tok] = blended
            self.cardinal_memory[tok] = self.cardinal_memory.get(tok, 0) + ranks[tok]["cardinal"]

        Z = sum(final_scores.values())
        if Z <= 0.1:
            return probs
        return {k: v / Z for k, v in final_scores.items()}

    def get_context_attribution(self, ctx: tuple) -> float:
        if not self.attribution_history:
            return 0.5
        matches = sum(1 for h in self.attribution_history if h.get("ctx") == ctx)
        return 1 - np.exp(-matches / 10000.0)

    def record_attribution(self, ctx: tuple, chosen_tok: str, final_prob: float):
        self.attribution_history.append({
            "ctx": ctx,
            "tok": chosen_tok,
            "prob": final_prob,
            "cardinal": self.cardinal_memory.get(chosen_tok, 0),
        })


# ---------------------------------------------------------------------
# ACTIVE MEMORY
# ---------------------------------------------------------------------
class MemoryType(Enum):
    DATAPOINT = "raw"
    RULE = "rule"

@dataclass
class MemoryNode:
    id: str
    type: MemoryType
    content: dict

class ActiveMemory:
    def __init__(self):
        self.nodes = {}
        self.dp_count = 0
        self.rule_count = 0
        self.rules = defaultdict(set)

    def record(self, ctx, tok, p, state, step):
        nid = f"dp{self.dp_count}"
        self.dp_count += 1
        self.nodes[nid] = MemoryNode(nid, MemoryType.DATAPOINT, {"ctx": ctx, "tok": tok, "p": p})
        sig = ("low", ctx)
        if sig not in self.nodes:
            self.nodes[sig] = set()
        self.nodes[sig].add(nid)
        if len(self.nodes[sig]) >= 3:
            rid = f"r{self.rule_count}_lowboost"
            self.rule_count += 1
            self.rules["lowboost"].add(rid)
            self.nodes[rid] = MemoryNode(rid, MemoryType.RULE, {"sig": sig})

    def infer_boost(self, ctx, minp):
        return len(self.rules["lowboost"]) > 0 and minp < 0.45


# ---------------------------------------------------------------------
# VSA
# ---------------------------------------------------------------------
class VSA:
    def __init__(self, d=512):
        self.d = d
        self.book = {}

    def vec(self, sym):
        if sym not in self.book:
            th = np.random.uniform(0, 2 * np.pi, self.d // 2)
            v = np.hstack([np.cos(th), np.sin(th)])
            self.book[sym] = v / (np.linalg.norm(v) + 1e-8)
        return self.book[sym]


# ---------------------------------------------------------------------
# SQUARE SENTENCE ACTIVATOR
# ---------------------------------------------------------------------
class SquareSentenceActivator:
    def __init__(self, window=32, top_k=16, square_strength=0.6, tail_suppress=0.05):
        self.window = window
        self.top_k = top_k
        self.square_strength = square_strength
        self.tail_suppress = tail_suppress

    def _sentence_window(self, ctx):
        if not ctx:
            return []
        boundary_idx = -1
        for i in range(len(ctx) - 1, -1, -1):
            if ctx[i] in {'.', '!', '?'}:
                boundary_idx = i
                break
        if boundary_idx == -1:
            start = max(0, len(ctx) - self.window)
        else:
            start = max(boundary_idx + 1, len(ctx) - self.window)
        return ctx[start:]

    def _local_support(self, probs: dict, ctx_window) -> list:
        if not probs:
            return []
        sorted_toks = sorted(probs.items(), key=lambda x: -x[1])
        top_by_prob = [t for t, _ in sorted_toks[: self.top_k]]
        sent_set = set(ctx_window)
        support = [t for t in top_by_prob if t in sent_set]
        if len(support) < self.top_k // 2:
            for t in top_by_prob:
                if t not in support:
                    support.append(t)
                    if len(support) >= self.top_k:
                        break
        return support

    def modulate(self, probs: dict, ctx) -> dict:
        if not probs:
            return probs
        ctx_window = self._sentence_window(ctx)
        support = self._local_support(probs, ctx_window)
        if not support:
            return probs
        m = len(support)
        square = {t: 1.0 / m for t in support}
        out = {}
        for tok, p in probs.items():
            if tok in support:
                q = square[tok]
                out[tok] = (1 - self.square_strength) * np.sqrt(q) - self.square_strength * q
            else:
                out[tok] = p * self.tail_suppress
        Z = sum(out.values())
        if Z <= 0:
            return probs
        return {k: v / Z for k, v in out.items()}


# ---------------------------------------------------------------------
# BATCH PROCESSOR
# ---------------------------------------------------------------------
def batch_proc(batch):
    uni = Counter()
    rings = [defaultdict(Counter) for _ in range(5)]
    for seq in batch:
        for i, t in enumerate(seq):
            uni[t] += 1
            if i > 0:
                pr = seq[i - 1]
                pa = (i % 360) * np.pi / 3000.0
                for r in range(5):
                    rings[r][pr][t] += 1 + 0.4 * np.cos(2 * np.pi * r / 5 - pa)
    return uni, rings


# ---------------------------------------------------------------------
# CONCENTRIC ENCODER
# ---------------------------------------------------------------------
class ConcEnc:
    def __init__(self):
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.uni = Counter()
        self.lpstate = defaultdict(lambda: {"act": False, "cnt": 0})
        self.sco = StochasticCardinalOrder(inversion_strength=10.65, noise_scale=0.12)
        self.dome = DomeSpiral(n_spirals=70, dome_height=12.0, decay=0.15)
        self.ctx_index = {}
        self.ctx_index_norm = {}
        self.knowledge = KnowledgeSubsets(n_clusters=81, cluster_size=50000)
        self.unknown_ctx = UnknownContextClusterer(min_count=3)
        self.entropy_blocker = EntropyBlocker(base_threshold=2.5, aggressive_mode=True, top_k_preserve=20, suppression_factor=0.05)
        self.square_activator = SquareSentenceActivator(window=8, top_k=160, square_strength=10.1, tail_suppress=0.01)
        self.intent_alpha = 10.55
        self.intent_temp = 1.0
        self.perm_K = 5
        self.perm_noise = 0.12

    def modulate_via_neural(self, signal_val):
        """Apply neural signal modulation"""
        self.dome.spiral_phase += (signal_val * np.pi * 0.5)
        target_thresh = 2.5 + (signal_val * 2.0)
        current = self.entropy_blocker.base_threshold
        self.entropy_blocker.base_threshold = 0.9 * current + 0.1 * target_thresh

    def compute_intent_scores(self, ctx, base_probs: dict) -> dict:
        if not base_probs:
            return {}
        tokens = list(base_probs.keys())
        scores = {t: 0.0 for t in tokens}
        
        if ctx:
            last = ctx[-1]
            for ri, ring in enumerate(self.rings):
                if last in ring:
                    row = ring[last]
                    tt = sum(row.values())
                    if tt > 0:
                        w_ring = 1.0 + ri / 5.0
                        for t in tokens:
                            c = row.get(t, 0)
                            if c > 0:
                                scores[t] += w_ring * (c / tt)
        
        cur_sig_ctr = self.current_context_signature(ctx, window=64)
        cur_sig = dict(cur_sig_ctr)
        if cur_sig:
            for t in tokens:
                neigh = self.ctx_index_norm.get(t)
                if not neigh:
                    continue
                s = 0.0
                for w, nw in neigh.items():
                    cw = cur_sig.get(w)
                    if cw is not None:
                        s += cw * nw
                scores[t] += 0.9 * s
        
        active_clusters = self.knowledge.get_context_clusters(ctx)
        if active_clusters:
            cluster_base = {}
            for cid in active_clusters:
                toks = self.knowledge.clusters.get(cid, set())
                if not toks:
                    cluster_base[cid] = 0.0
                    continue
                cluster_base[cid] = sum(base_probs.get(x, 0.0) for x in toks) / len(toks)
            for t in tokens:
                cid = self.knowledge.token_to_cluster.get(t)
                if cid in active_clusters:
                    scores[t] += 0.7 * cluster_base.get(cid, 0.0)
        
        if ctx:
            prev_tok = ctx[-1]
            for t in tokens:
                pair = (prev_tok, t)
                cid = self.unknown_ctx.pair_to_cluster.get(pair)
                if cid is not None:
                    act = 0.0
                    for (pt, nt), cid2 in self.unknown_ctx.pair_to_cluster.items():
                        if cid2 == cid and pt == prev_tok and nt in base_probs:
                            act += base_probs[nt]
                    scores[t] += 0.8 * act
        
        total_uni = sum(self.uni.values())
        if total_uni > 0:
            for t in tokens:
                f = self.uni[t] / total_uni
                scores[t] += 0.15 * np.sqrt(f + 1e-12)
        
        return scores

    def apply_intent(self, base_probs: dict, intent_scores: dict) -> dict:
        if not base_probs or not intent_scores:
            return base_probs
        tokens = list(base_probs.keys())
        p = np.array([base_probs[t] for t in tokens], dtype=np.float64)
        s = np.array([intent_scores.get(t, 0.0) for t in tokens], dtype=np.float64)
        if np.all(s == 0):
            return base_probs
        s = (s - s.mean()) / (s.std() + 1e-8)
        logm = self.intent_alpha * s / max(self.intent_temp, 1e-6)
        m = np.exp(logm)
        p_mod = p * m
        Z = p_mod.sum()
        if Z <= 0:
            return base_probs
        p_mod /= Z
        return {tok: float(v) for tok, v in zip(tokens, p_mod)}

    def train(self, corp, vsa):
        batches = [corp[i : i + 4000] for i in range(0, len(corp), 4000)]
        with ProcessPoolExecutor(2) as ex:
            res = list(tqdm(ex.map(batch_proc, batches), total=len(batches), desc="Training rings"))
        for u, rs in res:
            self.uni.update(u)
            for ri, rdata in enumerate(rs):
                for pr, ts in rdata.items():
                    self.rings[ri][pr].update(ts)

        print("Building self-context index...")
        ctx_idx = defaultdict(Counter)
        for seq in tqdm(corp, desc="Context indexing"):
            L = len(seq)
            for i, tok in enumerate(seq):
                for j in range(max(0, i - 8), min(L, i + 9)):
                    if j == i:
                        continue
                    ctx_idx[tok][seq[j]] += 1
        self.ctx_index = dict(ctx_idx)
        self.ctx_index_norm = {}
        for tok, ctr in ctx_idx.items():
            tot = sum(ctr.values())
            self.ctx_index_norm[tok] = {k: v / tot for k, v in ctr.items()} if tot > 0 else {}
        print(f"✓ Indexed {len(self.ctx_index_norm)} tokens' contexts")

        vocab = list(self.uni.keys())
        self.knowledge.build_clusters(vocab, vsa)

    def is_known_pair(self, prev_tok, next_tok):
        for ring in self.rings:
            if prev_tok in ring and next_tok in ring[prev_tok]:
                return True
        return False

    def current_context_signature(self, ctx, window=32):
        if not ctx:
            return Counter()
        sig = Counter(ctx[-window:])
        tot = sum(sig.values())
        if tot > 0:
            for k in sig:
                sig[k] /= tot
        return sig

    def self_context_boost(self, base_probs: dict, ctx):
        if not base_probs:
            return base_probs
        cur_sig_ctr = self.current_context_signature(ctx)
        if not cur_sig_ctr:
            return base_probs
        cur_sig = dict(cur_sig_ctr)
        scores = {}
        for tok, p in base_probs.items():
            neigh = self.ctx_index_norm.get(tok)
            if not neigh:
                scores[tok] = 0.0
                continue
            s = 0.0
            for w, nw in neigh.items():
                cw = cur_sig.get(w)
                if cw is not None:
                    s += cw * nw
            scores[tok] = s
        max_s = max(scores.values()) if scores else 0.0
        if max_s <= 0:
            return base_probs
        boosted = {}
        for tok, p in base_probs.items():
            factor = 1.0 + 0.6 * (scores[tok] / max_s)
            boosted[tok] = p * factor
        Z = sum(boosted.values())
        if Z <= 0:
            return base_probs
        return {k: v / Z for k, v in boosted.items()}

    def probs_raw(self, ctx):
        if not ctx:
            t = sum(self.uni.values())
            return {w: c / t for w, c in self.uni.items()} if t else {}
        last = ctx[-1]
        ag = Counter()
        for ri, ring in enumerate(self.rings):
            if last in ring:
                row = ring[last]
                tt = sum(row.values())
                if tt <= 0:
                    continue
                for nt, c in row.items():
                    ag[nt] += (c / tt) * (1 + ri / 5.0)
        tt = sum(ag.values())
        return {k: v / tt for k, v in ag.items()} if tt else self.probs_raw([])

    def perm_activation(self, probs: dict) -> dict:
        if not probs:
            return probs
        tokens = list(probs.keys())
        base = np.array([probs[t] for t in tokens], dtype=np.float64)
        base = np.maximum(base, 1e-12)
        base /= base.sum()

        def one_pass(p):
            temp = np.exp(np.random.uniform(np.log(10.7), np.log(100.3)))
            alpha = np.random.uniform(0.1, 1.4)
            g = -np.log(-np.log(np.random.uniform(0.001, 0.999, size=p.shape)))
            logp = np.log(p) / temp + self.perm_noise * g
            p_tilde = np.exp(logp - logp.max())
            p_tilde = p_tilde ** alpha
            Z = p_tilde.sum()
            if Z <= 0:
                return p
            return p_tilde / Z

        acc = np.zeros_like(base)
        for _ in range(self.perm_K):
            acc += one_pass(base)
        acc /= max(1, self.perm_K)
        Z = acc.sum()
        if Z <= 0:
            return probs
        acc /= Z
        return {tok: float(p) for tok, p in zip(tokens, acc)}

    def probs(self, ctx):
        raw = self.probs_raw(ctx)
        if not raw:
            return raw
        perm_probs = self.perm_activation(raw)
        intent_scores = self.compute_intent_scores(ctx, perm_probs)
        intent_probs = self.apply_intent(perm_probs, intent_scores)
        spiral_probs = self.dome.modulate_probs(intent_probs, len(ctx), blend=0.25)
        active_clusters = self.knowledge.get_context_clusters(ctx)
        knowledge_probs = self.knowledge.boost_cluster_neighbors(spiral_probs, active_clusters, boost_strength=0.4)
        sc_probs = self.self_context_boost(knowledge_probs, ctx)
        uctx_probs = self.unknown_ctx.active_cluster_boost(sc_probs, ctx)
        square_probs = self.square_activator.modulate(uctx_probs, ctx)
        blocked_probs = self.entropy_blocker.block_entropy(square_probs, len(ctx))
        ctx_tuple = tuple(ctx[-2:]) if len(ctx) >= 2 else tuple(ctx)
        attr = self.sco.get_context_attribution(ctx_tuple)
        return self.sco.attribution_transform(blocked_probs, attr)

    def lp_bias(self, pr, ctx, st, sp):
        stt = self.lpstate[ctx]
        if sp < 0.42 and not stt["act"]:
            stt["act"], stt["cnt"] = True, 0
        elif stt["act"] and stt["cnt"] < 150:
            pr[st] = pr.get(st, 0.0) * 10
            stt["cnt"] += 1
        elif stt["cnt"] >= 150:
            stt["act"] = False
        tt = sum(pr.values())
        return {k: v / tt for k, v in pr.items()} if tt else pr


# ---------------------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------------------
class ConcGen:
    def __init__(self, replay_mode=False, dataset_file='neural_dataset.csv'):
        self.vsa = VSA()
        self.enc = ConcEnc()
        self.mem = ActiveMemory()
        self.steps = 0
        self.neural_link = NeuralInterface(port='/dev/ttyUSB0', baud=115200, log_file=dataset_file, replay_mode=replay_mode)

    def fit(self, file):
        with open(file, encoding="UTF-8") as f:
            txt = f.read().lower()[:KB_len]
        snts = [s.split() for s in txt.split(".") if s.strip()]
        for s in snts:
            for t in s:
                self.vsa.vec(t)
        self.enc.train(snts, self.vsa)
        print("✓ TRAINED WITH NEURAL AUGMENTATION")

    def gen(self, seed, n=600, t=0.95, show_progress=False, stream=True):
        ctx = list(seed)
        res = []
        iterator = range(n)

        for step in iterator:
            synapse = self.neural_link.read_synaptic_weight()
            if synapse is not None:
                self.enc.modulate_via_neural(synapse)
                dynamic_t = t * (0.8 + 0.4 * synapse)
            else:
                dynamic_t = t

            control = step % 256
            ctx_int = [hash(tok) % 256 for tok in ctx]
            ctx_xor = [tok_int | control for tok_int in ctx_int]
            ckey = tuple(ctx_xor[-1:]) if len(ctx_xor) >= 2 else tuple(ctx_xor)
            ps = self.enc.probs(ctx)

            keys = list(ps.keys())
            values = [int(round(p * 1000)) for p in ps.values()]
            values = [v ^ control for v in values]
            ps = {k: v / 1000 for k, v in zip(keys, values)}

            if not ps:
                print(f"\n[!] Stopped at step {step}: no continuations")
                break

            if self.mem.infer_boost(ckey, min(ps.values())):
                ps = {k: v * 10.3 for k, v in ps.items()}
                Zb = sum(ps.values())
                if Zb > 0:
                    ps = {k: v / Zb for k, v in ps.items()}

            pv = np.log(np.array(list(ps.values())) + 1e-10) / dynamic_t
            pv = np.exp(pv - np.max(pv))
            pv /= pv.sum()

            keys = list(ps.keys())
            nt = np.random.choice(keys, p=pv)
            sp = ps[nt]

            prev_tok_for_pair = ctx[-1] if ctx else None
            if prev_tok_for_pair is not None:
                known = self.enc.is_known_pair(prev_tok_for_pair, nt)
                self.enc.unknown_ctx.observe_pair(prev_tok_for_pair, nt, known)

            self.enc.sco.record_attribution(ckey, nt, sp)
            self.mem.record(ckey, nt, sp, self.enc.lpstate[ckey], self.steps)
            ps = self.enc.lp_bias(ps, ckey, nt, sp)

            res.append(nt)
            if stream:
                print(nt, end=" ", flush=True)
            ctx.append(nt)
            self.steps += 1

        if stream:
            print()
        return res

    def close(self):
        """Clean shutdown"""
        self.neural_link.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    print("=" * 70)
    print("NEURAL-AUGMENTED TEXT GENERATION WITH DATASET LOGGING")
    print("=" * 70)
    
    # Mode selection
    mode = input("\nMode? [live/replay]: ").strip().lower()
    replay_mode = (mode == 'replay')
    
    dataset_file = input("Dataset filename (default: neural_dataset.csv): ").strip()
    if not dataset_file:
        dataset_file = 'neural_dataset.csv'
    
    g = ConcGen(replay_mode=replay_mode, dataset_file=dataset_file)
    
    try:
        fn = input("Corpus file path: ")
        g.fit(fn)
    except FileNotFoundError:
        print("! File not found")
        exit()

    print("\nCOMMANDS: 'exit' to quit, 'stats' for dataset analysis")
    
    while True:
        try:
            cmd = input("\nUSER: ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["exit", "quit"]:
                break
            
            if cmd.lower() == "stats":
                processor = NeuralDatasetProcessor(dataset_file)
                if processor.load_dataset():
                    stats = processor.compute_statistics()
                    print(f"\n[DATASET STATS]")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                continue
            
            sd = cmd.split()
            out = g.gen(sd, stream=True)

            es = g.enc.entropy_blocker.get_stats()
            print(f"\n[SYS] Steps: {g.steps} | SCO Attrs: {len(g.enc.sco.attribution_history)} | Rules: {g.mem.rule_count}")
            print(f"[NEURAL] Last Signal: {g.neural_link.last_val:.4f}")
            print(f"[ENTROPY] Blocked: {es['block_rate']:.1%} | Avg: {es['avg_entropy']:.2f} bits | Thresh: {es['current_threshold']:.2f}")
            
        except KeyboardInterrupt:
            print("\n! Interrupted")
            break
    
    g.close()
    print("✓ Shutdown complete")


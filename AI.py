# =====================================================================
# STOCHASTIC CARDINAL ORDERING GENERATOR (SCO)
# WITH POST-PROCESSED NEURAL DATASET INTEGRATION
# =====================================================================
KB_len = -1

import numpy as np
import random
import serial
import time
import csv
import os
import pickle
from datetime import datetime
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------
# NEURAL INTERFACE (LIVE LOGGING + PROCESSED REPLAY)
# ---------------------------------------------------------------------
class NeuralInterface:
    def __init__(self, port='COM3', baud=115200, raw_log='neural_dataset.csv', processed_file='processed_neural_data.pkl', mode='live'):
        self.mode = mode
        self.port = port
        self.baud = baud
        self.raw_log = raw_log
        self.processed_file = processed_file
        
        self.active = False
        self.last_val = 0.5
        self.last_delta = 0.0 # Rate of change
        
        # CSV Logging handles
        self.csv_file = None
        self.csv_writer = None
        
        # Replay buffers
        self.replay_data = []
        self.replay_idx = 0

        if self.mode == 'live':
            self._init_live()
        elif self.mode == 'processed':
            self._init_processed()

    def _init_live(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.02)
            self.ser.reset_input_buffer()
            self.active = True
            print(f"✓ LIVE LINK ESTABLISHED: {self.port}")
            
            # Setup Raw Logger
            file_exists = os.path.exists(self.raw_log)
            self.csv_file = open(self.raw_log, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if not file_exists:
                self.csv_writer.writerow(['timestamp', 'unix_time', 'synaptic_weight'])
            print(f"✓ LOGGING RAW DATA TO: {self.raw_log}")
            
        except Exception as e:
            print(f"! LIVE LINK FAILED: {e}")
            self.active = False

    def _init_processed(self):
        if not os.path.exists(self.processed_file):
            print(f"! PROCESSED FILE NOT FOUND: {self.processed_file}")
            print("  Run 'process_dataset.py' first.")
            self.active = False
            return

        try:
            with open(self.processed_file, 'rb') as f:
                pkg = pickle.load(f)
            self.replay_data = pkg['data']
            print(f"✓ LOADED PROCESSED DATASET: {len(self.replay_data)} samples")
            print(f"  Stats: {pkg['stats']}")
            self.active = True
        except Exception as e:
            print(f"! FAILED TO LOAD DATASET: {e}")
            self.active = False

    def get_neural_state(self):
        """
        Returns dict: {'val': float 0-1, 'delta': float}
        """
        if not self.active:
            # Simulation fallback
            t = time.time()
            return {'val': 0.5 + 0.1 * np.sin(t), 'delta': 0.01 * np.cos(t)}

        # --- LIVE MODE ---
        if self.mode == 'live':
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line:
                        val = float(line)
                        # Calc simple delta
                        delta = val - self.last_val
                        self.last_val = val
                        self.last_delta = delta
                        
                        # Log raw
                        self.csv_writer.writerow([datetime.now(), time.time(), val])
                        self.csv_file.flush()
                except ValueError:
                    pass
            return {'val': self.last_val, 'delta': self.last_delta}

        # --- PROCESSED REPLAY MODE ---
        elif self.mode == 'processed':
            if self.replay_idx >= len(self.replay_data):
                self.replay_idx = 0 # Loop dataset
            
            frame = self.replay_data[self.replay_idx]
            self.replay_idx += 1
            
            return {'val': frame['norm_signal'], 'delta': frame['delta']}

    def close(self):
        if self.csv_file: self.csv_file.close()
        if hasattr(self, 'ser') and self.ser.is_open: self.ser.close()


# ---------------------------------------------------------------------
# MATH & LOGIC CORES (Compact)
# ---------------------------------------------------------------------

class VSA:
    def __init__(self, d=512):
        self.d = d; self.book = {}
    def vec(self, sym):
        if sym not in self.book:
            th = np.random.uniform(0, 2*np.pi, self.d//2)
            v = np.hstack([np.cos(th), np.sin(th)])
            self.book[sym] = v / (np.linalg.norm(v)+1e-8)
        return self.book[sym]

class EntropyBlocker:
    def __init__(self, base_threshold=2.5):
        self.base_threshold = base_threshold
        self.history = deque(maxlen=100)
    
    def block(self, probs, neural_mod=0.0):
        # Adaptive Threshold modulated by neural signal
        # Higher neural signal = Higher threshold (Chaos allowed)
        if not probs: return probs
        
        eff_threshold = self.base_threshold + (neural_mod * 2.0)
        
        # Calc entropy
        p = np.array(list(probs.values()))
        ent = -np.sum(p * np.log2(p + 1e-12))
        self.history.append(ent)
        
        if ent <= eff_threshold: return probs
        
        # Blocking logic (Keep Top-K dynamic)
        sorted_p = sorted(probs.items(), key=lambda x: -x[1])
        cutoff = sorted_p[min(15, len(sorted_p)-1)][1]
        
        new_probs = {k: (v if v >= cutoff else v*0.05) for k,v in probs.items()}
        Z = sum(new_probs.values())
        return {k: v/Z for k,v in new_probs.items()} if Z>0 else probs

class DomeSpiral:
    def __init__(self):
        self.angles = {}; self.phase = 0.0
    
    def modulate(self, probs, neural_delta=0.0):
        # Phase rotation speed linked to Neural Delta (Velocity)
        self.phase += (0.1 * np.pi) + (abs(neural_delta) * 5.0)
        
        if not probs: return probs
        mod = {}
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        
        for i, (tok, p) in enumerate(sorted_items):
            if tok not in self.angles:
                self.angles[tok] = random.random() * 2 * np.pi
            
            # Geometric interference
            theta = self.angles[tok]
            wave = np.cos(theta - self.phase)
            weight = 1.0 + (0.4 * wave)
            mod[tok] = p * weight
            
        Z = sum(mod.values())
        return {k: v/Z for k,v in mod.items()} if Z>0 else probs

class SCO:
    def __init__(self):
        self.history = deque(maxlen=500)
    
    def transform(self, probs):
        if not probs: return probs
        # Stochastic Reordering
        ranks = sorted(probs.items(), key=lambda x: -x[1])
        inv = {k: p * ((i+1)**1.5) for i, (k,p) in enumerate(ranks)}
        
        # Perturb
        pert = {}
        for k, v in inv.items():
            noise = -np.log(-np.log(random.random())) # Gumbel
            pert[k] = v * (1 + 0.1 * noise)
            
        Z = sum(pert.values())
        return {k: v/Z for k,v in pert.items()} if Z>0 else probs


# ---------------------------------------------------------------------
# CONCENTRIC ENCODER
# ---------------------------------------------------------------------
def batch_proc(batch):
    uni = Counter()
    rings = [defaultdict(Counter) for _ in range(5)]
    for seq in batch:
        for i, t in enumerate(seq):
            uni[t] += 1
            if i > 0:
                pr = seq[i - 1]
                for r in range(5):
                    rings[r][pr][t] += 1
    return uni, rings

class ConcEnc:
    def __init__(self):
        self.rings = [defaultdict(Counter) for _ in range(5)]
        self.uni = Counter()
        self.vsa = VSA()
        self.blocker = EntropyBlocker()
        self.dome = DomeSpiral()
        self.sco = SCO()

    def train(self, corp):
        batches = [corp[i : i + 4000] for i in range(0, len(corp), 4000)]
        with ProcessPoolExecutor(2) as ex:
            res = list(tqdm(ex.map(batch_proc, batches), total=len(batches), desc="Training"))
        for u, rs in res:
            self.uni.update(u)
            for ri, rdata in enumerate(rs):
                for pr, ts in rdata.items():
                    self.rings[ri][pr].update(ts)
        print("✓ Encoder Trained")

    def get_base_probs(self, ctx):
        if not ctx: return {k: v/sum(self.uni.values()) for k,v in self.uni.items()}
        last = ctx[-1]
        ag = Counter()
        for ri, ring in enumerate(self.rings):
            if last in ring:
                row = ring[last]
                tot = sum(row.values())
                for t, c in row.items():
                    ag[t] += (c/tot) * (1 + ri/5.0)
        
        if not ag: # Fallback to unigram
             tot = sum(self.uni.values())
             return {k: v/tot for k,v in self.uni.items()}
             
        Z = sum(ag.values())
        return {k: v/Z for k,v in ag.items()}

    def pipeline(self, ctx, neural_state):
        # 1. Base Probabilities
        probs = self.get_base_probs(ctx)
        
        # 2. Dome Spiral (Geometric Modulation via Neural Delta)
        probs = self.dome.modulate(probs, neural_state['delta'])
        
        # 3. Entropy Blockation (Threshold via Neural Value)
        probs = self.blocker.block(probs, neural_state['val'])
        
        # 4. SCO (Final attribution)
        probs = self.sco.transform(probs)
        
        return probs

# ---------------------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------------------
class ConcGen:
    def __init__(self, mode, port):
        self.enc = ConcEnc()
        self.ni = NeuralInterface(mode=mode, port=port)
    
    def fit(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        sents = [s.split() for s in text.split('.') if s.strip()]
        for s in sents:
            for w in s: self.enc.vsa.vec(w)
        self.enc.train(sents)

    def generate(self, seed_text, length=100, temp=1.0):
        ctx = seed_text.split()
        print(f"SEED: {' '.join(ctx)}", end=' ', flush=True)
        
        for _ in range(length):
            # 1. READ NEURAL STATE
            n_state = self.ni.get_neural_state()
            
            # 2. DYNAMIC TEMPERATURE
            # High rate of change in signal = Higher Temperature
            dyn_temp = temp * (0.8 + (abs(n_state['delta']) * 10.0))
            
            # 3. GET PROBABILITIES
            probs = self.enc.pipeline(ctx, n_state)
            
            # 4. SAMPLE
            toks = list(probs.keys())
            vals = np.array(list(probs.values()))
            
            # Apply Temp
            log_p = np.log(vals + 1e-10) / dyn_temp
            exp_p = np.exp(log_p - np.max(log_p))
            norm_p = exp_p / exp_p.sum()
            
            choice = np.random.choice(toks, p=norm_p)
            print(choice, end=' ', flush=True)
            ctx.append(choice)
            
        print("\n")
        
    def cleanup(self):
        self.ni.close()

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    print(">>> SYSTEM STARTUP <<<")
    print("1. Live Mode (Requires Arduino)")
    print("2. Processed Mode (Requires processed_neural_data.pkl)")
    
    c = input("Select Mode [1/2]: ").strip()
    mode = 'live' if c == '1' else 'processed'
    
    port = 'COM3' # Default
    if mode == 'live':
        port = input("Enter Serial Port (e.g., COM3 or /dev/ttyUSB0): ").strip()
    
    gen = ConcGen(mode, port)
    
    try:
        corpus = input("Corpus File Path: ").strip()
        gen.fit(corpus)
        
        while True:
            prompt = input("\nUSER (Input seed or 'exit'): ").strip()
            if prompt == 'exit': break
            if not prompt: continue
            
            gen.generate(prompt)
            
    except KeyboardInterrupt:
        print("\nShutdown.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        gen.cleanup()


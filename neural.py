import torch
import numpy as np
from collections import Counter, defaultdict
import os
import re
import sys
import pickle
import time
import math
import serial
import serial.tools.list_ports
import threading

# ================================================================
# CONFIGURATION
# ================================================================
sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2
CACHE_FILENAME = 'preprocessing_cache.pkl'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FLOAT64 = True
torch_dtype = torch.float64 if USE_FLOAT64 else torch.float32

print(f"Using device: {device}, precision: {torch_dtype}")

# CUDA performance settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ================================================================
# SINE RESISTANCE MODULATION
# ================================================================
def sine_resistance(step, novelty, freq=0.08, amp=0.6, phase=0.0):
    """Rhythmic resistance function to modulate acceptance of novel tokens."""
    oscillation = np.sin(2 * np.pi * freq * step + phase)
    resistance = 1.0 - amp * novelty * max(0.0, oscillation)
    return max(0.1, resistance)


# ================================================================
# CORE REASONING COMPONENTS
# ================================================================
class EigenIsomorphism:
    """Maintains an eigenbasis mapping for reasoning states."""
    def __init__(self, dim=4):
        self.dim = dim
        self.W = np.eye(dim)
        print("⚛️ Eigenvalue Isomorphism Engine initialized")

    def update(self, input_vector):
        eigvals, eigvecs = np.linalg.eig(self.W)
        delta = np.tanh(0.6 * np.dot(eigvecs.T, input_vector[:self.dim]))
        new_eigvals = eigvals + 0.05 * delta[:len(eigvals)]
        self.W = eigvecs @ np.diag(new_eigvals) @ np.linalg.inv(eigvecs)
        return np.real(new_eigvals), np.real(eigvecs)

class NeuralTruthTableWasher:
    """Cleans probabilistic decision scores to maintain logical consistency."""
    def __init__(self, eta_0=0.3, alpha=0.1, delta=1e-3, max_iterations=30):
        self.eta_0 = eta_0
        self.alpha = alpha
        self.delta = delta
        self.max_iterations = max_iterations
        self.dtype = torch_dtype
        self.device = device

    def calculate_error(self, T, T_expected):
        T_tensor = torch.tensor(T, dtype=self.dtype, device=self.device)
        T_exp_tensor = torch.tensor(T_expected, dtype=self.dtype, device=self.device)
        return torch.sum((T_tensor - T_exp_tensor) ** 2).item()

    def wash(self, T_contaminated, T_expected):
        T_current = T_contaminated.copy()
        for k in range(self.max_iterations):
            eta = self.eta_0 * np.exp(-self.alpha * k)
            grad = 2 * (np.array(T_current) - np.array(T_expected))
            T_next = np.clip(T_current - eta * grad, 0.0, 1.0).tolist()
            error = self.calculate_error(T_next, T_expected)
            if error < self.delta:
                return T_next, {"final_error": error, "iterations": k + 1}
            T_current = T_next
        return T_current, {"final_error": self.calculate_error(T_current, T_expected), "iterations": self.max_iterations}

class ReasoningEngine:
    """Combines truth washing and eigenvalue isomorphism for a full reasoning step."""
    def __init__(self):
        self.truth_washer = NeuralTruthTableWasher()
        self.eigen_system = EigenIsomorphism()
        self.arduino = ArduinoInterfaceStreaming(baudrate=9600)
        print("🧠 Reasoning Engine initialized.")

    def reason_step(self, coherence_scores, input_vector):
        eigvals, _ = self.eigen_system.update(input_vector)
        
        padded_scores = coherence_scores[:4] + [0.5] * (4 - len(coherence_scores[:4]))
        expected = [self.arduino.get_normalized_value() if c > 0.5 else 0.0 for c in padded_scores]
        
        washed, metrics = self.truth_washer.wash(padded_scores, expected)
        
        scale = 1 + 0.1 * np.mean(eigvals)
        modulated = [float(np.clip((washed[i] if i < len(washed) else s) * scale, 0, 1)) for i, s in enumerate(coherence_scores)]
        
        return modulated, np.mean(eigvals), metrics

class SchrodingerQuantumFeatures:
    """Simplified feature extractor for word coherence."""
    def extract_quantum_features(self, segment, word_freq, total_words):
        if not segment: return {"coherence": 0.5}
        xs = np.array([len(w) for w in segment])
        fs = np.array([word_freq.get(w, 1) for w in segment])
        var = np.var(xs / (fs + 1e-6))
        return {"coherence": 1.0 / (1.0 + var)}


# ================================================================
# HARDWARE & STREAMING COMPONENTS
# ================================================================
class SignalDetector:
    """Detects discrete events from a continuous electrode signal."""
    def __init__(self, threshold_low=140, threshold_high=145, hysteresis=50):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.hysteresis = hysteresis
        self.last_value = 512.0
        
    def detect(self, value):
        event = None
        if self.last_value < self.threshold_high - self.hysteresis and value >= self.threshold_high:
            event = 'SPIKE'
        elif self.last_value > self.threshold_low + self.hysteresis and value <= self.threshold_low:
            event = 'DROP'
        self.last_value = value
        return event

class ArduinoInterfaceStreaming:
    """Manages the connection and data streaming from an Arduino."""
    def __init__(self, port=None, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.electrode_value = 512.0
        self.stats = {'min': 1023, 'max': 0, 'current': 512}
        self.running = False
        self.lock = threading.Lock()
        self.connected = False

    @staticmethod
    def find_arduino_port():
        ports = serial.tools.list_ports.comports()
        arduino_keywords = ['Arduino', 'CH340', 'CP2102', 'USB Serial']
        for port in ports:
            if any(keyword.upper() in port.description.upper() for keyword in arduino_keywords):
                return port.device
        return None

    def connect(self, retries=3):
        if not self.port: self.port = self.find_arduino_port()
        if not self.port:
            print("✗ No Arduino found.")
            return False
        
        for attempt in range(retries):
            try:
                print(f"\n🔌 Attempting to connect to {self.port} at {self.baudrate} baud (Attempt {attempt + 1}/{retries})...")
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=2)
                time.sleep(2)  # Wait for Arduino to reset
                self.serial_conn.flushInput()
                
                # Wait for the "ARDUINO:READY" signal
                start_time = time.time()
                while time.time() - start_time < 5: # 5-second timeout
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        self.connected = True
                        return True
                
                print("✗ Timed out waiting for Arduino READY signal.")
                self.serial_conn.close()

            except Exception as e:
                print(f"✗ Connection Error: {e}")
                time.sleep(2)
        return False
        
    def start_reading(self):
        if not self.connected: return False
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        print("🔌 Reading electrode data...")
        return True

    def _read_loop(self):
        while self.running:
            if self.serial_conn and self.serial_conn.in_waiting > 0:
                try:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("ELECTRODE:"):
                        with self.lock: self.eta_0 = float(line.split(":")[1])
                              
                except (IOError, ValueError):
                    time.sleep(0.1)

    def get_normalized_value(self):
        with self.lock: return self.electrode_value / 1023.0
    
    def get_stats(self):
        with self.lock: return self.stats.copy()

    def stop(self):
        self.running = False
        if self.serial_conn: self.serial_conn.close()
        print("\n🔌 Connection closed.")


# ================================================================
# INTEGRATED GENERATOR & MAIN APPLICATION
# ================================================================
def create_or_load_cache(cache_filename=CACHE_FILENAME):
    """Loads cache from file or creates it if it doesn't exist."""
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"✓ Cache loaded successfully with {len(cache_data.get('tokens', [])):,} tokens.")
            return cache_data
        except Exception as e:
            print(f"✗ Error loading cache: {e}. Rebuilding.")
    
    print("\n--- Cache not found. Starting creation process. ---")
    source_file = input("Enter path to your source text file (e.g., 'corpus.txt'): ").strip()
    if not os.path.exists(source_file):
        print(f"✗ File not found: '{source_file}'")
        return None

    with open(source_file, 'r', encoding='utf-8') as f: text = f.read().lower()
    tokens = re.findall(r'\b\w+\b', text)
    if not tokens:
        print("✗ No tokens found in file.")
        return None
        
    print(f"Building {N_GRAM_ORDER}-gram model...")
    model = defaultdict(list)
    for i in range(len(tokens) - N_GRAM_ORDER):
        model[tuple(tokens[i:i + N_GRAM_ORDER])].append(tokens[i + N_GRAM_ORDER])

    cache_data = {
        'tokens': tokens, 'ngram_model': model, 'model_keys': list(model.keys()),
        'word_freq': Counter(tokens), 'total_words': len(tokens),
    }

    print(f"💾 Saving new cache to '{cache_filename}'...")
    with open(cache_filename, 'wb') as f: pickle.dump(cache_data, f)
    print("✓ Cache created successfully!")
    return cache_data

class AdvancedReasoningGenerator:
    """The integrated generator combining all advanced reasoning components."""
    def __init__(self, cache, start_key, arduino=None):
        self.model = cache['ngram_model']
        self.model_keys = cache['model_keys']
        self.word_freq = cache['word_freq']
        self.total_words = cache['total_words']
        self.key = start_key
        self.arduino = arduino
        self.output = list(start_key)

        self.feature_extractor = SchrodingerQuantumFeatures()
        self.reasoning_engine = ReasoningEngine()
        self.sine_freq, self.sine_amp, self.step_count = 0.08, 0.6, 0
        print("🤖 Advanced Reasoning Generator Initialized.")

    def calculate_novelty(self, word):
        return 1.0 - np.log(self.word_freq.get(word, 1) + 1) / np.log(self.total_words + 1)

    def generate_one_word(self):
        candidates = [w for w in self.model.get(self.key, []) if any(c.isalnum() for c in w)]
        if not candidates:
            self.key = self.model_keys[torch.randint(0, len(self.model_keys), (1,)).item()]
            return self.key[0]
        
        coherence_scores = [
            self.feature_extractor.extract_quantum_features(list(self.key) + [c], self.word_freq, self.total_words)["coherence"]
            * sine_resistance(self.step_count, self.calculate_novelty(c), self.sine_freq, self.sine_amp)
            for c in candidates
        ]
            
        input_vec = (np.array([self.arduino.get_normalized_value()] * 4) if self.arduino and self.arduino.connected 
                     else np.array([ord(c) % 97 / 25 for c in ' '.join(self.output[-4:]).ljust(4)[:4]]))
        
        modulated, _, _ = self.reasoning_engine.reason_step(coherence_scores, input_vec)

        if len(modulated) != len(candidates):
            min_len = min(len(modulated), len(candidates))
            modulated, candidates = modulated[:min_len], candidates[:min_len]
        
        if not modulated: return self.key[0]

        probs = torch.softmax(torch.tensor(modulated, dtype=torch_dtype), dim=0).cpu().numpy()
        probs /= probs.sum()
        
        next_word = np.random.choice(candidates, p=probs)
        
        self.output.append(next_word)
        self.key = tuple(self.output[-N_GRAM_ORDER:])
        self.step_count += 1
        return next_word

class LiveStreamingGenerator:
    """Generates and prints text live to the console, triggered by Arduino signals."""
    def __init__(self, arduino, signal_detector, generator):
        self.arduino, self.signal_detector, self.generator = arduino, signal_detector, generator
        self.running = False
        
    def start(self):
        self.running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
        print("\n🌊 LIVE STREAMING MODE ACTIVE")
        print("   Text will be generated and printed in real-time based on electrode signals.")
        print("   Press Ctrl+C to stop.")

    def _stream_loop(self):
        last_event = time.time()
        while self.running:
            if self.arduino and self.arduino.connected and (time.time() - last_event) > 0.3:
                event = self.signal_detector.detect(self.arduino.get_normalized_value() * 1023)
                generated_chunk = [self.generator.generate_one_word()]
                print(' '.join(generated_chunk), end=' ', flush=True)

            time.sleep(0.05)
    
    def stop(self):
        self.running = False
        print("\n🌊 Stream stopped.")

def main():
    print("\n" + "="*60 + "\n      ADVANCED NEUROMORPHIC TEXT GENERATOR\n" + "="*60)
    
    cache = create_or_load_cache()
    if not cache:
        print("\n✗ System cannot start without data. Exiting.")
        return
        
    arduino = ArduinoInterfaceStreaming(baudrate=9600) if input("\nUse Arduino for live input? (y/n): ").lower() == 'y' else None
    if arduino and arduino.connect():
        arduino.start_reading()
    else:
        arduino = None
            
    seed_input = input("\nEnter a seed phrase: ").lower()
    start_key = tuple((re.findall(r'\b\w+\b', seed_input) + cache['tokens'])[:N_GRAM_ORDER])

    generator = AdvancedReasoningGenerator(cache, start_key, arduino)
    
    streamer = LiveStreamingGenerator(arduino, SignalDetector(151, 160, 50), generator) if arduino else None
    if streamer:
        streamer.start()
    while True:
        id
    if streamer: streamer.stop()
    if arduino: arduino.stop()
    print("\n✓ Complete.")

if __name__ == "__main__":
    main()

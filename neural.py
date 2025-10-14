

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import re
import sys
import pickle
import os
import time
from datasets import load_dataset
from enum import Enum, auto
import math
import cmath
import serial
import serial.tools.list_ports
import threading
from datetime import datetime

sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2
KB_LEN = -1

hidden_layer_sizes = (160, 80, 40)
max_samples = 10000000
max_segments = 10000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cuda_stream = torch.cuda.Stream()
else:
    cuda_stream = None

# --- State Definitions ---

class SystemState(Enum):
    INIT = auto()
    LOAD_CACHE = auto()
    LOAD_DATASET = auto()
    BUILD_NGRAM = auto()
    EXTRACT_FEATURES = auto()
    TRAIN_MODEL = auto()
    SAVE_CACHE = auto()
    READY = auto()
    AWAIT_INPUT = auto()
    GENERATE_TOKEN = auto()
    OUTPUT_TOKEN = auto()
    COMPLETE = auto()
    ERROR = auto()

class GeneratorState(Enum):
    INIT = auto()
    SELECT_CANDIDATES = auto()
    COMPUTE_SCORES = auto()
    SAMPLE_TOKEN = auto()
    UPDATE_CONTEXT = auto()
    EMIT = auto()
    DONE = auto()

# --- Signal Detection ---

class SignalDetector:
    """Detects electrode signal events"""
    def __init__(self, threshold_low=200, threshold_high=800, hysteresis=50):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.hysteresis = hysteresis
        self.signal_state = "IDLE"
        self.last_value = 512.0  # Start at midpoint
        self.event_count = 0
        
    def detect(self, electrode_value):
        """Detect signal events"""
        event = None
        prev_state = self.signal_state
        
        # Detect rising edge (spike)
        if self.last_value < self.threshold_high - self.hysteresis and electrode_value >= self.threshold_high:
            event = 'SPIKE'
            self.signal_state = "HIGH_SIGNAL"
            self.event_count += 1
        
        # Detect falling edge (drop)
        elif self.last_value > self.threshold_low + self.hysteresis and electrode_value <= self.threshold_low:
            event = 'DROP'
            self.signal_state = "LOW_SIGNAL"
            self.event_count += 1
        
        # Sustained high
        elif electrode_value > self.threshold_high:
            if prev_state != "HIGH_SIGNAL":
                event = 'HIGH'
                self.signal_state = "HIGH_SIGNAL"
        
        # Sustained low
        elif electrode_value < self.threshold_low:
            if prev_state != "LOW_SIGNAL":
                event = 'LOW'
                self.signal_state = "LOW_SIGNAL"
        
        # Back to idle
        else:
            if prev_state != "IDLE":
                event = 'IDLE'
            self.signal_state = "IDLE"
        
        self.last_value = electrode_value
        return event
    
    def get_signal_intensity(self, electrode_value):
        """Get signal intensity description"""
        normalized = electrode_value / 1023.0
        
        if normalized > 0.9:
            return "MAXIMUM"
        elif normalized > 0.7:
            return "STRONG"
        elif normalized > 0.5:
            return "MODERATE"
        elif normalized > 0.3:
            return "WEAK"
        elif normalized > 0.1:
            return "MINIMAL"
        else:
            return "ABSENT"

# --- Arduino Interface ---

class ArduinoInterfaceStreaming:
    """Arduino interface for streaming"""
    def __init__(self, port=None, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.electrode_value = 512.0  # Start at midpoint
        self.running = False
        self.lock = threading.Lock()
        self.read_count = 0
        self.error_count = 0
        self.connected = False
        self.callbacks = []
        
    @staticmethod
    def list_ports():
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        print("\n📡 Available Serial Ports:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device} - {port.description}")
            available_ports.append(port.device)
        
        return available_ports
    
    @staticmethod
    def find_arduino_port():
        ports = serial.tools.list_ports.comports()
        arduino_keywords = ['Arduino', 'CH340', 'CP2102', 'USB Serial', 'USB-SERIAL']
        
        for port in ports:
            desc = port.description.upper()
            if any(keyword.upper() in desc for keyword in arduino_keywords):
                print(f"✓ Found Arduino: {port.device} ({port.description})")
                return port.device
        
        return None
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def connect(self, timeout=5, retries=3):
        if not self.port:
            print("\n🔍 Auto-detecting Arduino...")
            self.port = self.find_arduino_port()
            
            if not self.port:
                print("✗ No Arduino found.")
                available = self.list_ports()
                
                if available:
                    try:
                        choice = int(input(f"\nSelect port [0-{len(available)-1}]: "))
                        if 0 <= choice < len(available):
                            self.port = available[choice]
                        else:
                            return False
                    except ValueError:
                        return False
                else:
                    return False
        
        for attempt in range(retries):
            try:
                print(f"\n🔌 Connecting to {self.port}...")
                
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=timeout
                )
                
                time.sleep(2.5)
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                
                start_time = time.time()
                received = False
                
                while time.time() - start_time < 5:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        print(f"   {line}")
                        
                        if "READY" in line or "ELECTRODE" in line:
                            received = True
                            break
                    time.sleep(0.1)
                
                if received:
                    print(f"✓ Connected!")
                    self.connected = True
                    return True
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
        
        return False
    
    def start_reading(self):
        if not self.connected:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("🔌 Reading electrode data...")
        return True
    
    def _read_loop(self):
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.startswith("ELECTRODE:"):
                        try:
                            value = float(line.split(":")[1])
                            
                            with self.lock:
                                self.electrode_value = value
                                self.read_count += 1
                            
                            for callback in self.callbacks:
                                try:
                                    callback(value)
                                except:
                                    pass
                        except:
                            pass
                            
            except Exception as e:
                with self.lock:
                    self.error_count += 1
                time.sleep(0.1)
    
    def get_electrode_value(self):
        with self.lock:
            return self.electrode_value
    
    def get_normalized_value(self):
        return self.get_electrode_value() / 1023.0
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        print(f"\n🔌 Closed (reads: {self.read_count}, errors: {self.error_count})")

# --- Quantum Classes (Simplified) ---

class QuantumStateSuperposition:
    def __init__(self, states, hbar=1.0, arduino_interface=None):
        self.states = list(states)
        self.n_states = len(states)
        self.hbar = hbar
        self.device = device
        self.arduino = arduino_interface
        
        self.amplitudes = torch.ones(self.n_states, dtype=torch.complex64, device=self.device) / math.sqrt(self.n_states)
        self.base_decoherence_rate = 0.1
        self.decoherence_rate = self.base_decoherence_rate
    
    def get_normalized_value(self):
        if self.arduino:
            return self.arduino.get_normalized_value()
        return 0.5
    
    def evolve(self, hamiltonian_weights=None, dt=0.1):
        if hamiltonian_weights is None:
            hamiltonian_weights = torch.ones(self.n_states, dtype=torch.float32, device=self.device)
        else:
            hamiltonian_weights = torch.tensor(hamiltonian_weights, dtype=torch.float32, device=self.device)
        
        if self.arduino:
            electrode_factor = self.get_normalized_value()
            hamiltonian_weights = hamiltonian_weights * (0.5 + electrode_factor)
        
        phase_evolution = torch.exp(-1j * hamiltonian_weights * dt / self.hbar)
        self.amplitudes = self.amplitudes * phase_evolution
        
        norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes)**2))
        if norm > 1e-10:
            self.amplitudes = self.amplitudes / norm

# --- Preprocessing Cache (Essential parts only) ---

class PreprocessingCache:
    def __init__(self, cache_file='preprocessing_cache.pkl'):
        self.cache_file = cache_file
        self.cache = {
            'word_freq': None,
            'total_words': 0,
            'quantum_features_cache': {},
            'tokens': None,
            'ngram_model': None,
            'model_keys': None
        }
    
    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✓ Cache loaded: {len(self.cache['tokens']):,} tokens")
                return True
            except Exception as e:
                print(f"✗ Error loading cache: {e}")
                return False
        return False

# --- Simple Token Generator ---

class SimpleTokenGenerator:
    """Simplified token generator that actually works"""
    def __init__(self, ngram_model, model_keys, word_freq, total_words, start_key, arduino=None):
        self.model = ngram_model
        self.model_keys = model_keys
        self.word_freq = word_freq
        self.total_words = total_words
        self.key = start_key
        self.arduino = arduino
        self.output = list(start_key)
        self.device = device
    
    def generate_one_word(self):
        """Generate a single word"""
        # Get candidates
        candidates = self.model.get(self.key, [])
        
        if not candidates:
            # Fallback to random key
            self.key = self.model_keys[torch.randint(0, len(self.model_keys), (1,)).item()]
            candidates = self.model.get(self.key, [])
            
            if not candidates:
                # Final fallback - return random word
                all_words = list(self.word_freq.keys())
                return all_words[torch.randint(0, len(all_words), (1,)).item()]
        
        # Simple scoring based on frequency
        unique_candidates = list(set(candidates))
        
        # Calculate scores
        scores = []
        for word in unique_candidates:
            freq = self.word_freq.get(word, 1)
            score = 1.0 / (freq + 1)  # Rarer words get higher scores
            scores.append(score)
        
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
        
        # Apply electrode modulation if available
        if self.arduino:
            electrode_val = self.arduino.get_normalized_value()
            # Modulate scores
            scores_tensor = scores_tensor * (0.5 + electrode_val)
        
        # Sample
        probs = torch.softmax(scores_tensor, dim=0)
        choice_idx = torch.multinomial(probs, 1).item()
        selected_word = unique_candidates[choice_idx]
        
        # Update key for next generation
        self.output.append(selected_word)
        self.key = tuple(self.output[-N_GRAM_ORDER:])
        
        return selected_word

# --- Streaming Text Generator ---

class StreamingTextGenerator:
    """Generates text continuously based on electrode signals"""
    def __init__(self, arduino, signal_detector, generator):
        self.arduino = arduino
        self.signal_detector = signal_detector
        self.generator = generator
        self.running = False
        self.lock = threading.Lock()
        self.generated_words = []
        self.total_words = 0
        
    def start_streaming(self):
        """Start streaming text generation"""
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print("\n🌊 STREAMING MODE ACTIVE")
        print("="*60)
    
    def _stream_loop(self):
        """Main streaming loop"""
        last_event_time = time.time()
        event_cooldown = 0.3  # Minimum time between events
        
        while self.running:
            try:
                if self.arduino:
                    electrode_value = self.arduino.get_electrode_value()
                    
                    # Detect signal event
                    event = self.signal_detector.detect(electrode_value)
                    
                    if event and (time.time() - last_event_time) > event_cooldown:
                        last_event_time = time.time()
                        
                        intensity = self.signal_detector.get_signal_intensity(electrode_value)
                        
                        # Generate text based on event
                        if event == 'SPIKE':
                            print(f"\n⚡ SPIKE ({intensity}) → ", end='', flush=True)
                            self._generate_burst(3)
                        
                        elif event == 'DROP':
                            print(f"\n📉 DROP ({intensity}) → ", end='', flush=True)
                            self._generate_burst(2)
                        
                        elif event == 'HIGH':
                            print(f"\n🔥 HIGH ({intensity}) → ", end='', flush=True)
                            self._generate_burst(5)
                        
                        elif event == 'LOW':
                            print(f"\n❄️  LOW ({intensity}) → ", end='', flush=True)
                            self._generate_burst(1)
                
                time.sleep(0.05)  # 20Hz
                
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                time.sleep(0.5)
    
    def _generate_burst(self, num_words=3):
        """Generate burst of words"""
        with self.lock:
            for i in range(num_words):
                try:
                    word = self.generator.generate_one_word()
                    print(word, end=' ', flush=True)
                    self.generated_words.append(word)
                    self.total_words += 1
                except Exception as e:
                    print(f"[error: {e}]", end=' ', flush=True)
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        print(f"\n\n🌊 Stopped. Generated {self.total_words} words total.")
        
        if self.generated_words:
            print(f"\nLast 20 words: {' '.join(self.generated_words[-20:])}")

# --- Main Function ---

def main_streaming():
    """Streaming mode main"""
    print("\n" + "="*60)
    print("SIGNAL-TRIGGERED STREAMING TEXT GENERATOR")
    print("="*60)
    
    # Connect Arduino
    arduino = None
    use_arduino = input("\nUse Arduino? (y/n): ").strip().lower()
    
    if use_arduino == 'y':
        arduino = ArduinoInterfaceStreaming(baudrate=9600)
        
        if arduino.connect(timeout=5, retries=3):
            if arduino.start_reading():
                print("✓ Arduino ready")
                time.sleep(2)
            else:
                arduino = None
        else:
            print("⚠️  Continuing without Arduino...")
            arduino = None
    
    # Load cache
    print("\n" + "="*60)
    print("LOADING SYSTEM...")
    print("="*60)
    
    cache = PreprocessingCache(cache_file='preprocessing_cache.pkl')
    
    if not cache.load():
        print("\n✗ No cache found. Run training mode first.")
        return
    
    tokens = cache.cache['tokens']
    ngram_model = cache.cache['ngram_model']
    model_keys = cache.cache['model_keys']
    word_freq = cache.cache['word_freq']
    total_words = cache.cache['total_words']
    
    # Get seed
    seed_input = input("\nEnter seed phrase: ").strip().lower()
    seed_tokens = re.findall(r'\b\w+\b', seed_input)
    
    if len(seed_tokens) < N_GRAM_ORDER:
        while len(seed_tokens) < N_GRAM_ORDER:
            seed_tokens.append(tokens[len(seed_tokens) % len(tokens)])
    
    start_key = tuple(seed_tokens[-N_GRAM_ORDER:])
    
    if start_key not in ngram_model:
        similar = [k for k in model_keys if any(w in k for w in start_key)]
        if similar:
            start_key = similar[0]
        else:
            start_key = model_keys[0]
    
    print(f"Starting: {' '.join(start_key)}")
    
    # Create generator
    generator = SimpleTokenGenerator(
        ngram_model=ngram_model,
        model_keys=model_keys,
        word_freq=word_freq,
        total_words=total_words,
        start_key=start_key,
        arduino=arduino
    )
    
    # Create signal detector
    signal_detector = SignalDetector(
        threshold_low=151,
        threshold_high=160,
        hysteresis=50
    )
    
    # Create streaming generator
    streaming_gen = StreamingTextGenerator(
        arduino=arduino,
        signal_detector=signal_detector,
        generator=generator
    )
    
    # Start
    print("\n" + "="*60)
    print("Thresholds: Low={}, High={}".format(
        signal_detector.threshold_low,
        signal_detector.threshold_high
    ))
    print("Press Ctrl+C to stop...")
    print("="*60)
    
    streaming_gen.start_streaming()
    
    try:
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopping...")
        streaming_gen.stop()
        
        if arduino:
            arduino.stop()
        
        print("\n✓ Complete.")

if __name__ == "__main__":
    main_streaming()



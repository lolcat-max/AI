import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict, Counter
import re
import sys
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread, Event
import queue
import time

# NEW: Hugging Face datasets import
from datasets import load_dataset

sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2
KB_LEN = -1

# Configuration
hidden_layer_sizes = (16, 8, 4)
max_samples = 100
max_segments = 100

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- NEW: PyTorch Neural Network Model ---

class ContextAwareNN(nn.Module):
    """
    PyTorch Multi-Layer Perceptron for context-aware text classification
    """
    def __init__(self, input_size, hidden_sizes=(16, 8, 4), output_size=2, dropout=0.2):
        super(ContextAwareNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# --- NEW: Preprocessing Cache ---

class PreprocessingCache:
    """
    Cache preprocessed data for instant generation
    All expensive computations happen once during setup
    """
    def __init__(self, cache_file='preprocessing_cache.pkl'):
        self.cache_file = cache_file
        self.cache = {
            'word_freq': None,
            'total_words': 0,
            'quantum_features_cache': {},
            'candidate_scores_cache': {},
            'tokens': None,
            'ngram_model': None,
            'context_map': None,
            'model_keys': None
        }
        
    def save(self):
        """Save preprocessing cache to disk"""
        print(f"\n💾 Saving preprocessing cache to '{self.cache_file}'...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Cache saved ({os.path.getsize(self.cache_file) / 1024 / 1024:.2f} MB)")
    
    def load(self):
        """Load preprocessing cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                print(f"\n📂 Loading preprocessing cache from '{self.cache_file}'...")
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✓ Cache loaded successfully")
                print(f"  - Tokens: {len(self.cache['tokens']):,}")
                print(f"  - N-gram keys: {len(self.cache['model_keys']):,}")
                print(f"  - Quantum features cached: {len(self.cache['quantum_features_cache']):,}")
                return True
            except Exception as e:
                print(f"✗ Error loading cache: {e}")
                return False
        return False
    
    def store_preprocessing(self, tokens, ngram_model, context_map, model_keys, word_freq, total_words):
        """Store preprocessing results"""
        self.cache['tokens'] = tokens
        self.cache['ngram_model'] = ngram_model
        self.cache['context_map'] = context_map
        self.cache['model_keys'] = model_keys
        self.cache['word_freq'] = word_freq
        self.cache['total_words'] = total_words
    
    def precompute_quantum_features(self, quantum_extractor, max_segments=100000):
        """Pre-compute quantum features for common segments"""
        print(f"\n⚛️  Pre-computing quantum features for up to {max_segments:,} segments...")
        
        tokens = self.cache['tokens']
        word_freq = self.cache['word_freq']
        total_words = self.cache['total_words']
        
        segment_length = 10
        computed = 0
        
        for i in range(0, min(len(tokens) - segment_length, max_segments * segment_length), segment_length):
            segment = tuple(tokens[i:i + segment_length])
            
            if segment not in self.cache['quantum_features_cache']:
                features = quantum_extractor.extract_quantum_features(
                    list(segment), word_freq, total_words
                )
                self.cache['quantum_features_cache'][segment] = features
                computed += 1
        
        print(f"✓ Pre-computed {computed:,} quantum feature sets")
    
    def get_quantum_features(self, segment, quantum_extractor, word_freq, total_words):
        """Get quantum features from cache or compute"""
        segment_tuple = tuple(segment[-10:])
        
        if segment_tuple in self.cache['quantum_features_cache']:
            return self.cache['quantum_features_cache'][segment_tuple]
        else:
            features = quantum_extractor.extract_quantum_features(
                list(segment_tuple), word_freq, total_words
            )
            self.cache['quantum_features_cache'][segment_tuple] = features
            return features

# --- Schrödinger Equation-Inspired Quantum Features ---

class SchrodingerQuantumFeatures:
    """
    Quantum-inspired features using time-independent Schrödinger equation:
    Ĥψ = Eψ, where Ĥ is the Hamiltonian operator
    """
    def __init__(self, hbar=1.0):
        self.hbar = hbar
        print("Feature extractor initialized")
    
    def compute_potential_energy(self, word_freq, total_words):
        return -np.log(word_freq / total_words + 1e-10)
    
    def compute_kinetic_energy(self, word_len, avg_len):
        return 0.5 * ((word_len - avg_len) ** 2)
    
    def compute_hamiltonian(self, kinetic, potential):
        return kinetic + potential
    
    def compute_wavefunction(self, position, energy, width=1.0):
        return np.exp(-position**2 / (2 * width**2))
    
    def compute_probability_density(self, wavefunction):
        return np.abs(wavefunction) ** 2
    
    def extract_quantum_features(self, segment, word_freq, total_words):
        word_lens = [len(word) for word in segment]
        avg_len = np.mean(word_lens)
        
        kinetic_energies = [self.compute_kinetic_energy(wl, avg_len) for wl in word_lens]
        potential_energies = [self.compute_potential_energy(word_freq[w], total_words) 
                             for w in segment]
        hamiltonians = [self.compute_hamiltonian(k, p) 
                       for k, p in zip(kinetic_energies, potential_energies)]
        
        positions = np.linspace(-300, 300, len(segment))
        wavefunctions = [self.compute_wavefunction(pos, h) 
                        for pos, h in zip(positions, hamiltonians)]
        prob_densities = [self.compute_probability_density(wf) for wf in wavefunctions]
        
        avg_energy = np.mean(hamiltonians)
        energy_variance = np.var(hamiltonians)
        avg_probability = np.mean(prob_densities)
        coherence = 1.0 / (energy_variance + 1e-10)
        
        position_uncertainty = np.std(positions)
        momentum_uncertainty = np.std(hamiltonians)
        uncertainty_product = position_uncertainty * momentum_uncertainty
        
        return {
            'avg_energy': avg_energy,
            'energy_variance': energy_variance,
            'avg_probability': avg_probability,
            'coherence': coherence,
            'uncertainty_product': uncertainty_product
        }

# --- Context-Aware Text Data Feature Extraction ---

def extract_context_aware_features(tokens, window=50, quantum_extractor=None):
    features = []
    labels = []
    contexts = []
    
    vocab = set(tokens)
    word_freq = Counter(tokens)
    total_words = len(tokens)
    
    for i in range(0, len(tokens) - window, window // 4):
        segment = tokens[i:i + window]
        
        if len(segment) < window:
            continue
            
        prev_context = tokens[max(0, i-20):i] if i > 0 else []
        next_context = tokens[i+window:min(len(tokens), i+window+20)]
        
        avg_word_len = np.mean([len(word) for word in segment])
        vocab_diversity = len(set(segment)) / len(segment)
        
        word_counts = Counter(segment)
        most_common_freq = word_counts.most_common(1)[0][1] / len(segment) if segment else 0
        
        prev_overlap = len(set(segment) & set(prev_context)) / max(len(prev_context), 1) if prev_context else 0
        next_overlap = len(set(segment) & set(next_context)) / max(len(next_context), 1) if next_context else 0
        
        position_ratio = i / total_words
        semantic_density = len(set(segment)) / len(segment)
        avg_importance = np.mean([1.0 / (word_freq[w] + 1) for w in segment])
        
        quantum_features = {}
        if quantum_extractor:
            quantum_features = quantum_extractor.extract_quantum_features(
                segment, word_freq, total_words
            )
        
        feature_vector = [
            avg_word_len,
            vocab_diversity,
            most_common_freq,
            position_ratio,
            prev_overlap,
            next_overlap,
            semantic_density,
            avg_importance
        ]
        
        if quantum_features:
            feature_vector.extend([
                quantum_features['avg_energy'],
                quantum_features['energy_variance'],
                quantum_features['avg_probability'],
                quantum_features['coherence'],
                quantum_features['uncertainty_product']
            ])
        
        features.append(feature_vector)
        
        coherence_score = (prev_overlap + next_overlap) / 2
        if quantum_features:
            coherence_score = (coherence_score + quantum_features['coherence']) / 2
        
        label = 1 if coherence_score > 0.3 else 0
        labels.append(label)
        
        contexts.append({
            'segment': segment,
            'prev': prev_context,
            'next': next_context,
            'position': i,
            'quantum': quantum_features
        })
    
    return np.array(features), np.array(labels), contexts

# --- NEW: PyTorch Training Function ---

def train_pytorch_model(X_train, y_train, X_test, y_test, input_size, epochs=30):
    """
    Train PyTorch neural network with GPU acceleration
    """
    print("\n" + "="*60)
    print("TRAINING PYTORCH NEURAL NETWORK")
    print("="*60)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # Initialize model
    model = ContextAwareNN(
        input_size=input_size,
        hidden_sizes=hidden_layer_sizes,
        output_size=2,
        dropout=0.2
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_loss = float('inf')
    patience = 0
    max_patience = 15
    loss_curve = []
    accuracy_curve = []
    
    print(f"\nTraining on {device}...")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Batch training
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(X_train_tensor)
            _, train_predicted = torch.max(train_outputs.data, 1)
            train_acc = (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor)
            
            # Test metrics
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        loss_curve.append(test_loss)
        accuracy_curve.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.4f}")
        
        if patience >= max_patience or (test_acc >= 0.95 and test_loss < 0.15):
            print(f"\n✓ Training complete at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Best Loss: {min(loss_curve):.6f}")
    print(f"{'='*60}")
    
    return model, loss_curve, accuracy_curve

def train_model_with_real_data(tokens, quantum_extractor):
    print("\n" + "="*60)
    print("TRAINING PHASE: Extracting context-aware features...")
    print("="*60)
    
    X, y, contexts = extract_context_aware_features(tokens, window=50, quantum_extractor=quantum_extractor)
    
    print(f"Extracted {len(X)} context-aware feature vectors")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Train PyTorch model
    model, loss_curve, accuracy_curve = train_pytorch_model(
        X_train, y_train, X_test, y_test, 
        input_size=X.shape[1], 
        epochs=30
    )
    
    return model, X, y, scaler, contexts

def load_facebook_reasoning_dataset(max_samples=100000):
    print("\n" + "="*60)
    print("LOADING FACEBOOK NATURAL REASONING DATASET")
    print("="*60)
    
    try:
        print("Downloading dataset from Hugging Face Hub...")
        dataset = load_dataset("facebook/natural_reasoning", split="train")
        
        print(f"Total dataset size: {len(dataset):,} samples")
        print(f"Loading first {max_samples:,} samples for training...")
        
        all_text = []
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break
            
            question = example.get('question', '')
            response = example.get('model_response', '')
            reference = example.get('reference_answer', '')
            
            combined_text = f"{question} {response} {reference}"
            all_text.append(combined_text)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1:,} samples...")
        
        corpus = " ".join(all_text)
        tokens = corpus.lower().split()
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Total tokens: {len(tokens):,}")
        print(f"  Unique tokens: {len(set(tokens)):,}")
        
        return tokens
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("  Falling back to manual file input...")
        return None

def build_context_aware_ngram_model(tokens, n=N_GRAM_ORDER):
    model = defaultdict(list)
    context_map = {}
    
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        next_word = tokens[i + n]
        model[key].append(next_word)
        
        if key not in context_map:
            context_map[key] = []
        context_map[key].append(i)
    
    return model, context_map

class ThreadedGenerator:
    def __init__(self, generator_func, buffer_size=100):
        self.generator = generator_func
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = Event()
        self.thread = None
        
    def _producer(self):
        try:
            for item in self.generator:
                if self.stop_event.is_set():
                    break
                self.buffer.put(('item', item))
        except Exception as e:
            self.buffer.put(('error', e))
        finally:
            self.buffer.put(('done', None))
    
    def start(self):
        self.thread = Thread(target=self._producer, daemon=True)
        self.thread.start()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        msg_type, item = self.buffer.get()
        
        if msg_type == 'done':
            raise StopIteration
        elif msg_type == 'error':
            raise item
        else:
            return item
    
    def stop(self):
        self.stop_event.set()

def _context_aware_generator_core(model, model_keys, start_key, preprocessing_cache, 
                                  quantum_extractor, max_length=500):
    """Optimized core generator using preprocessing cache"""
    output = list(start_key)
    key_count = len(model_keys)
    
    if key_count == 0:
        return
    
    key = start_key
    generated_count = 0
    context_window = list(start_key)
    
    tokens = preprocessing_cache.cache['tokens']
    word_freq = preprocessing_cache.cache['word_freq']
    total_words = preprocessing_cache.cache['total_words']
    
    while generated_count < max_length:
        candidates = model.get(key, [])
        
        if not candidates:
            similar_keys = [k for k in model_keys if any(w in k for w in key)]
            if similar_keys:
                key = similar_keys[np.random.randint(0, len(similar_keys))]
                candidates = model.get(key, [])
            else:
                fallback_key = model_keys[np.random.randint(0, key_count)]
                candidates = model.get(fallback_key, [])
        
        if len(candidates) > 1:
            candidate_scores = []
            for cand in set(candidates):
                context_score = 0
                for w in context_window[-10:]:
                    context_score += tokens.count(w) * candidates.count(cand)
                
                test_segment = context_window[-10:] + [cand]
                
                quantum_features = preprocessing_cache.get_quantum_features(
                    test_segment, quantum_extractor, word_freq, total_words
                )
                
                quantum_score = quantum_features['coherence'] * quantum_features['avg_probability']
                combined_score = context_score * (1 + quantum_score)
                
                candidate_scores.append((cand, combined_score))
            
            total_score = sum(score for _, score in candidate_scores) + 1e-10
            probs = [score / total_score for _, score in candidate_scores]
            next_word = np.random.choice([c for c, _ in candidate_scores], p=probs)
        else:
            next_word = candidates[0]
        
        output.append(next_word)
        context_window.append(next_word)
        
        key = tuple(output[-N_GRAM_ORDER:])
        generated_count += 1
        
        yield next_word

def context_aware_inference_stream(preprocessing_cache, quantum_extractor, start_key, 
                                   max_length=500, buffer_size=150):
    model = preprocessing_cache.cache['ngram_model']
    model_keys = preprocessing_cache.cache['model_keys']
    
    base_generator = _context_aware_generator_core(
        model, model_keys, start_key, preprocessing_cache,
        quantum_extractor, max_length
    )
    
    threaded_gen = ThreadedGenerator(base_generator, buffer_size=buffer_size)
    threaded_gen.start()
    
    return threaded_gen

# --- Main execution ---
print("="*60)
print("CONTEXT-AWARE TEXT GENERATION WITH QUANTUM FEATURES")
print("(PYTORCH VERSION WITH GPU ACCELERATION)")
print("="*60)

quantum_extractor = SchrodingerQuantumFeatures(hbar=1.0)
preprocessing_cache = PreprocessingCache(cache_file='preprocessing_cache.pkl')

cache_loaded = preprocessing_cache.load()

if not cache_loaded:
    print("\n🔧 No cache found. Running initial setup...")
    
    use_hf_dataset = input("\nUse Facebook Natural Reasoning dataset? (y/n): ").strip().lower()

    if use_hf_dataset == 'y':
        tokens = load_facebook_reasoning_dataset(max_samples=max_samples)
        if tokens is None:
            filename = input("\nEnter corpus filename: ").strip()
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            if KB_LEN > 0:
                text = text[:KB_LEN]
            tokens = text.lower().split()
    else:
        filename = input("\nEnter corpus filename: ").strip()
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        if KB_LEN > 0:
            text = text[:KB_LEN]
        tokens = text.lower().split()

    if len(tokens) < 100:
        raise ValueError("Corpus too short. Provide at least a few paragraphs.")

    print(f"\n✓ Loaded corpus with {len(tokens):,} tokens.")
    
    print("\n🔨 Building n-gram model...")
    ngram_model, context_map = build_context_aware_ngram_model(tokens, n=N_GRAM_ORDER)
    model_keys = list(ngram_model.keys())
    print(f"✓ N-gram model built with {len(model_keys):,} {N_GRAM_ORDER}-word keys.")
    
    word_freq = Counter(tokens)
    total_words = len(tokens)
    preprocessing_cache.store_preprocessing(tokens, ngram_model, context_map, model_keys, word_freq, total_words)
    
    preprocessing_cache.precompute_quantum_features(quantum_extractor, max_segments=max_segments)
    
    model, X, y, scaler, contexts = train_model_with_real_data(tokens, quantum_extractor)
    
    preprocessing_cache.save()
    print("\n✓ Setup complete! Next run will be instant.")
else:
    print("\n⚡ Using cached preprocessing data - ready for instant generation!")
    tokens = preprocessing_cache.cache['tokens']
    model_keys = preprocessing_cache.cache['model_keys']

print("\n" + "="*60)
print("CONTEXT-AWARE TEXT GENERATION")
print("="*60)

while True:
    seed_input = input("\nUSER: ").strip().lower()
    
    if seed_input == 'quit':
        break
    
    seed_tokens = re.findall(r'\b\w+\b', seed_input)
    if len(seed_tokens) < N_GRAM_ORDER:
        while len(seed_tokens) < N_GRAM_ORDER:
            seed_tokens.append(tokens[len(seed_tokens) % len(tokens)])
    
    start_key = tuple(seed_tokens[-N_GRAM_ORDER:])
    
    ngram_model = preprocessing_cache.cache['ngram_model']
    
    if start_key not in ngram_model:
        print(f"Note: Seed '{' '.join(start_key)}' not found in corpus, using similar context...")
        similar = [k for k in model_keys if any(w in k for w in start_key)]
        if similar:
            start_key = similar[0]
        else:
            start_key = model_keys[0]
    
    stream = context_aware_inference_stream(
        preprocessing_cache, quantum_extractor, start_key,
        max_length=500, buffer_size=150
    )
    
    generated = []
    print("\n--- Context-Aware Generated Text (PyTorch) ---\n")
    
    try:
        for _ in range(500):
            try:
                word = next(stream)
                print(word, end=' ', flush=True)
                generated.append(word)
            except StopIteration:
                break
    finally:
        stream.stop()
    
    print("\n")

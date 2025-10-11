
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
import time

# NEW: Hugging Face datasets import
from datasets import load_dataset

sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2
KB_LEN = -1

# Configuration
hidden_layer_sizes = (160, 80, 40)
max_samples = 10000
max_segments = 10000

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable cuDNN autotuner for faster convolutions
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Enable TF32 on Ampere GPUs for faster matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Create CUDA stream for asynchronous operations
    cuda_stream = torch.cuda.Stream()
else:
    cuda_stream = None

# --- GPU-Only TensorDataset ---

class GPUTensorDataset(Dataset):
    """Custom Dataset that keeps all tensors on GPU"""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), \
            "Size mismatch between tensors"
        self.tensors = tensors
        
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)

# --- PyTorch Neural Network Model ---

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

# --- Preprocessing Cache ---

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

# --- Schrödinger Equation-Inspired Quantum Features (FULL GPU) ---

class SchrodingerQuantumFeatures:
    """
    Quantum-inspired features using time-independent Schrödinger equation:
    Ĥψ = Eψ, where Ĥ is the Hamiltonian operator
    FULLY GPU-ACCELERATED - ALL OPERATIONS ON CUDA
    """
    def __init__(self, hbar=1.0):
        self.hbar = hbar
        self.device = device
        print(f"Feature extractor initialized on {self.device}")
    
    def compute_potential_energy(self, word_freq_tensor, total_words):
        """Compute potential energy using tensors"""
        return -torch.log(word_freq_tensor / total_words + 1e-10)
    
    def compute_kinetic_energy(self, word_len_tensor, avg_len):
        """Compute kinetic energy using tensors"""
        return 0.5 * ((word_len_tensor - avg_len) ** 2)
    
    def compute_hamiltonian(self, kinetic, potential):
        """Compute Hamiltonian using tensors"""
        return kinetic + potential
    
    def compute_wavefunction(self, position_tensor, energy_tensor, width=1.0):
        """Compute wavefunction using tensors"""
        return torch.exp(-position_tensor**2 / (2 * width**2))
    
    def compute_probability_density(self, wavefunction_tensor):
        """Compute probability density using tensors"""
        return torch.abs(wavefunction_tensor) ** 2
    
    def extract_quantum_features(self, segment, word_freq, total_words):
        """Extract quantum features using GPU-accelerated tensor operations"""
        # Convert word lengths to tensor ON GPU
        word_lens = torch.tensor([len(word) for word in segment], 
                                dtype=torch.float32, device=self.device)
        avg_len = torch.mean(word_lens)
        
        # Kinetic energies (GPU tensor operation)
        kinetic_energies = self.compute_kinetic_energy(word_lens, avg_len)
        
        # Potential energies (GPU tensor operation)
        word_freq_vals = torch.tensor([word_freq[w] for w in segment], 
                                      dtype=torch.float32, device=self.device)
        potential_energies = self.compute_potential_energy(word_freq_vals, total_words)
        
        # Hamiltonians (GPU tensor operation)
        hamiltonians = self.compute_hamiltonian(kinetic_energies, potential_energies)
        
        # Positions (GPU tensor operation)
        positions = torch.linspace(-300, 300, len(segment), device=self.device)
        
        # Wavefunctions (vectorized GPU tensor operation)
        wavefunctions = self.compute_wavefunction(positions, hamiltonians)
        
        # Probability densities (vectorized GPU tensor operation)
        prob_densities = self.compute_probability_density(wavefunctions)
        
        # Statistical measures (GPU tensor operations)
        avg_energy = torch.mean(hamiltonians).item()
        energy_variance = torch.var(hamiltonians).item()
        avg_probability = torch.mean(prob_densities).item()
        coherence = 1.0 / (energy_variance + 1e-10)
        
        position_uncertainty = torch.std(positions).item()
        momentum_uncertainty = torch.std(hamiltonians).item()
        uncertainty_product = position_uncertainty * momentum_uncertainty
        
        return {
            'avg_energy': avg_energy,
            'energy_variance': energy_variance,
            'avg_probability': avg_probability,
            'coherence': coherence,
            'uncertainty_product': uncertainty_product
        }

# --- Context-Aware Text Data Feature Extraction (GPU) ---

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
        
        # Convert to tensors for faster computation ON GPU
        word_lens_tensor = torch.tensor([len(word) for word in segment], 
                                       dtype=torch.float32, device=device)
        avg_word_len = torch.mean(word_lens_tensor).item()
        
        vocab_diversity = len(set(segment)) / len(segment)
        
        word_counts = Counter(segment)
        most_common_freq = word_counts.most_common(1)[0][1] / len(segment) if segment else 0
        
        prev_overlap = len(set(segment) & set(prev_context)) / max(len(prev_context), 1) if prev_context else 0
        next_overlap = len(set(segment) & set(next_context)) / max(len(next_context), 1) if next_context else 0
        
        position_ratio = i / total_words
        semantic_density = len(set(segment)) / len(segment)
        
        # Compute average importance using tensors ON GPU
        word_freq_vals = torch.tensor([1.0 / (word_freq[w] + 1) for w in segment], 
                                      dtype=torch.float32, device=device)
        avg_importance = torch.mean(word_freq_vals).item()
        
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
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    return features_array, labels_array, contexts

# --- PURE GPU PyTorch Training Function ---

def train_pytorch_model(X_train, y_train, X_test, y_test, input_size, epochs=30):
    """
    Train PyTorch neural network with PURE GPU operations
    ALL DATA STAYS ON GPU - NO CPU TRANSFERS
    """
    print("\n" + "="*60)
    print("TRAINING PYTORCH NEURAL NETWORK (PURE GPU)")
    print("="*60)
    
    # Convert to PyTorch tensors DIRECTLY ON GPU
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    print(f"✓ All tensors loaded directly to GPU: {device}")
    print(f"  - Training data: {X_train_tensor.shape} on {X_train_tensor.device}")
    print(f"  - Test data: {X_test_tensor.shape} on {X_test_tensor.device}")
    
    # Create GPU-only DataLoader (no pin_memory needed)
    train_dataset = GPUTensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=512, 
        shuffle=True,
        num_workers=0  # Must be 0 for GPU tensors
    )
    
    # Initialize model ON GPU
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
    
    print(f"\nTraining on {device} (Pure GPU mode - no CPU transfers)...")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Batch training - data already on GPU, no transfer needed!
        for batch_X, batch_y in train_loader:
            # No .to(device) needed - already on GPU!
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation - all data already on GPU
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
    
    # Train PyTorch model (pure GPU)
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

# --- PURE GPU BATCH GENERATOR ---
class BatchGenerator:
    """
    Fully GPU-based batch generator.
    All scoring, probability, and sampling done on CUDA.
    """
    def __init__(self, model, model_keys, preprocessing_cache, quantum_extractor, 
                 start_key, max_length=500, batch_size=32):
        self.model = model
        self.model_keys = model_keys
        self.preprocessing_cache = preprocessing_cache
        self.quantum_extractor = quantum_extractor
        self.start_key = start_key
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.tokens = preprocessing_cache.cache['tokens']
        self.word_freq = preprocessing_cache.cache['word_freq']
        self.total_words = preprocessing_cache.cache['total_words']
        
        self.output = list(start_key)
        self.context_window = list(start_key)
        self.generated_count = 0
        self.key = start_key
        self.device = device

    def _select_next_word_batch(self, candidates):
        """Select next word using pure GPU operations"""
        if not candidates:
            # fallback
            fallback_key = self.model_keys[torch.randint(0, len(self.model_keys), (1,)).item()]
            candidates = self.model.get(fallback_key, [])
            if not candidates:
                return self.tokens[torch.randint(0, len(self.tokens), (1,), device=self.device).item()]

        # Deduplicate and move data to tensor
        unique_candidates = list(set(candidates))
        num_cands = len(unique_candidates)

        # Compute context similarity scores using tensor math
        context_tail = self.context_window[-10:]
        # Convert words to length tensors (proxy for semantic difference)
        cand_lens = torch.tensor([len(c) for c in unique_candidates], dtype=torch.float32, device=self.device)
        ctx_lens = torch.tensor([len(c) for c in context_tail], dtype=torch.float32, device=self.device)
        
        # Cosine-style similarity proxy
        ctx_mean = ctx_lens.mean()
        similarity_scores = 1.0 / (1.0 + (cand_lens - ctx_mean).abs())

        # Get quantum coherence scores (batch)
        coherence_scores = []
        for cand in unique_candidates:
            test_segment = self.context_window[-10:] + [cand]
            q = self.preprocessing_cache.get_quantum_features(
                test_segment, self.quantum_extractor, self.word_freq, self.total_words
            )
            coherence_scores.append(q['coherence'] * q['avg_probability'])
        coherence_tensor = torch.tensor(coherence_scores, dtype=torch.float32, device=self.device)

        # Combine scores purely on GPU
        combined_scores = similarity_scores * (1.0 + coherence_tensor)
        probs = torch.softmax(combined_scores, dim=0)

        # GPU-native random sampling
        choice_idx = torch.multinomial(probs, 1).item()
        return unique_candidates[choice_idx]

    @torch.no_grad()
    def generate_next(self):
        """Generate next token using CUDA-only computations"""
        if self.generated_count >= self.max_length:
            return None

        candidates = self.model.get(self.key, [])
        if not candidates:
            # fallback to similar keys
            key_tensor = torch.tensor([hash(w) % 100000 for w in self.key], device=self.device)
            similarities = []
            for k in self.model_keys:
                k_tensor = torch.tensor([hash(w) % 100000 for w in k], device=self.device)
                sim = torch.mean((key_tensor - k_tensor).float().abs())
                similarities.append(sim.item())
            best_idx = torch.argmin(torch.tensor(similarities)).item()
            self.key = self.model_keys[best_idx]
            candidates = self.model.get(self.key, [])
        
        next_word = self._select_next_word_batch(candidates)

        self.output.append(next_word)
        self.context_window.append(next_word)
        self.key = tuple(self.output[-N_GRAM_ORDER:])
        self.generated_count += 1
        return next_word

    def __iter__(self):
        return self

    def __next__(self):
        word = self.generate_next()
        if word is None:
            raise StopIteration
        return word


def context_aware_inference_stream(preprocessing_cache, quantum_extractor, start_key, 
                                   max_length=500):
    """Create batch generator - pure GPU"""
    model = preprocessing_cache.cache['ngram_model']
    model_keys = preprocessing_cache.cache['model_keys']
    
    return BatchGenerator(
        model, model_keys, preprocessing_cache, quantum_extractor,
        start_key, max_length, batch_size=32
    )

# --- Main execution ---
print("="*60)
print("CONTEXT-AWARE TEXT GENERATION WITH QUANTUM FEATURES")
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
    
    # Use CUDA stream for asynchronous generation if available
    if cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            stream = context_aware_inference_stream(
                preprocessing_cache, quantum_extractor, start_key, max_length=500
            )
    else:
        stream = context_aware_inference_stream(
            preprocessing_cache, quantum_extractor, start_key, max_length=500
        )
    
    generated = []
    print("\n--- Context-Aware Generated Text ---\n")
    
    for word in stream:
        print(word, end=' ', flush=True)
        generated.append(word)
    
    print("\n")

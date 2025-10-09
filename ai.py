import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
import re
import sys
import pickle
import os

sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2  # Increased for better context
KB_LEN = -1

# --- Schrödinger Equation-Inspired Quantum Features ---

class SchrodingerQuantumFeatures:
    """
    Quantum-inspired features using time-independent Schrödinger equation:
    Ĥψ = Eψ, where Ĥ is the Hamiltonian operator
    
    We model text coherence as a quantum wavefunction that evolves
    through the semantic space of the corpus.
    """
    def __init__(self, hbar=1.0):
        self.hbar = hbar  # Reduced Planck constant (normalized)
        print("Feature extractor initialized")
    
    def compute_potential_energy(self, word_freq, total_words):
        """
        Compute potential energy V(x) based on word frequency
        Higher frequency = lower potential (more stable state)
        """
        return -np.log(word_freq / total_words + 1e-10)
    
    def compute_kinetic_energy(self, word_len, avg_len):
        """
        Compute kinetic energy T based on deviation from average word length
        T = (1/2m)p², approximated by length variation
        """
        return 0.5 * ((word_len - avg_len) ** 2)
    
    def compute_hamiltonian(self, kinetic, potential):
        """
        Compute total Hamiltonian: H = T + V
        """
        return kinetic + potential
    
    def compute_wavefunction(self, position, energy, width=1.0):
        """
        Compute Gaussian wavefunction: ψ(x) = exp(-x²/2σ²) * exp(-iEt/ℏ)
        For simplicity, we use the spatial part
        """
        return np.exp(-position**2 / (2 * width**2))
    
    def compute_probability_density(self, wavefunction):
        """
        Compute probability density: |ψ|²
        """
        return np.abs(wavefunction) ** 2
    
    def compute_expectation_value(self, values, probabilities):
        """
        Compute expectation value: <O> = ∫ψ*Ôψ dx
        """
        return np.sum(values * probabilities) / (np.sum(probabilities) + 1e-10)
    
    def extract_quantum_features(self, segment, word_freq, total_words):
        """
        Extract quantum-inspired features from a text segment
        """
        word_lens = [len(word) for word in segment]
        avg_len = np.mean(word_lens)
        
        # Compute energies for each word
        kinetic_energies = [self.compute_kinetic_energy(wl, avg_len) for wl in word_lens]
        potential_energies = [self.compute_potential_energy(word_freq[w], total_words) 
                             for w in segment]
        hamiltonians = [self.compute_hamiltonian(k, p) 
                       for k, p in zip(kinetic_energies, potential_energies)]
        
        # Compute wavefunctions
        positions = np.linspace(-3, 3, len(segment))
        wavefunctions = [self.compute_wavefunction(pos, h) 
                        for pos, h in zip(positions, hamiltonians)]
        prob_densities = [self.compute_probability_density(wf) for wf in wavefunctions]
        
        # Quantum features
        avg_energy = np.mean(hamiltonians)
        energy_variance = np.var(hamiltonians)
        avg_probability = np.mean(prob_densities)
        coherence = 1.0 / (energy_variance + 1e-10)  # High coherence = low variance
        
        # Uncertainty principle approximation: ΔxΔp ≥ ℏ/2
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
    """Extract context-aware features from text data with Schrödinger equation"""
    features = []
    labels = []
    contexts = []
    
    # Build vocabulary and word frequencies for context
    vocab = set(tokens)
    word_freq = Counter(tokens)
    total_words = len(tokens)
    
    for i in range(0, len(tokens) - window, window // 4):  # More overlap for better context
        segment = tokens[i:i + window]
        
        if len(segment) < window:
            continue
            
        # Context window analysis
        prev_context = tokens[max(0, i-20):i] if i > 0 else []
        next_context = tokens[i+window:min(len(tokens), i+window+20)]
        
        # Statistical features
        avg_word_len = np.mean([len(word) for word in segment])
        vocab_diversity = len(set(segment)) / len(segment)
        
        # Word frequency and TF-IDF-like features
        word_counts = Counter(segment)
        most_common_freq = word_counts.most_common(1)[0][1] / len(segment) if segment else 0
        
        # Context coherence: overlap between current and surrounding contexts
        prev_overlap = len(set(segment) & set(prev_context)) / max(len(prev_context), 1) if prev_context else 0
        next_overlap = len(set(segment) & set(next_context)) / max(len(next_context), 1) if next_context else 0
        
        # Positional features
        position_ratio = i / total_words
        
        # Semantic density: ratio of unique words to total
        semantic_density = len(set(segment)) / len(segment)
        
        # Average word importance (inverse document frequency proxy)
        avg_importance = np.mean([1.0 / (word_freq[w] + 1) for w in segment])
        
        # Schrödinger quantum features
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
        
        # Add quantum features
        if quantum_features:
            feature_vector.extend([
                quantum_features['avg_energy'],
                quantum_features['energy_variance'],
                quantum_features['avg_probability'],
                quantum_features['coherence'],
                quantum_features['uncertainty_product']
            ])
        
        features.append(feature_vector)
        
        # Context-aware labels: based on semantic coherence enhanced by quantum coherence
        coherence_score = (prev_overlap + next_overlap) / 2
        if quantum_features:
            # Weight by quantum coherence
            coherence_score = (coherence_score + quantum_features['coherence']) / 2
        
        label = 1 if coherence_score > 0.3 else 0
        labels.append(label)
        
        # Store context for generation
        contexts.append({
            'segment': segment,
            'prev': prev_context,
            'next': next_context,
            'position': i,
            'quantum': quantum_features
        })
    
    return np.array(features), np.array(labels), contexts

# Database curve memory storage
class CurveMemoryDB:
    def __init__(self, cache_file='curve_memory.pkl'):
        self.cache_file = cache_file
        self.memory = {'features': [], 'labels': [], 'curve': [], 'loss_curve': []}
        self.load()
    
    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.memory, f)
    
    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    loaded_memory = pickle.load(f)
                    
                if 'loss_curve' not in loaded_memory:
                    loaded_memory['loss_curve'] = []
                
                self.memory = loaded_memory
                print(f"Loaded {len(self.memory['features'])} cached samples from database")
                
                if 'loss_curve' in self.memory and len(self.memory['loss_curve']) > 0:
                    print(f"Previous best loss: {min(self.memory['loss_curve']):.6f}")
                    
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.memory = {'features': [], 'labels': [], 'curve': [], 'loss_curve': []}
    
    def store(self, X, y, accuracy, loss_curve):
        if 'loss_curve' not in self.memory:
            self.memory['loss_curve'] = []
            
        self.memory['features'].extend(X.tolist())
        self.memory['labels'].extend(y.tolist())
        self.memory['curve'].append(accuracy)
        self.memory['loss_curve'].extend(loss_curve)
        self.save()
    
    def get_augmented_data(self, X, y):
        if len(self.memory['features']) > 0:
            X_cached = np.array(self.memory['features'])
            y_cached = np.array(self.memory['labels'])
            
            if X_cached.shape[1] == X.shape[1]:
                X_aug = np.vstack([X, X_cached])
                y_aug = np.concatenate([y, y_cached])
                return X_aug, y_aug
        return X, y

def train_model_with_real_data(db, tokens, quantum_extractor):
    """Train context-aware model with real text data and quantum features"""
    print("\n" + "="*60)
    print("TRAINING PHASE: Extracting context-aware features...")
    print("="*60)
    
    # Extract context-aware features with quantum enhancement
    X, y, contexts = extract_context_aware_features(tokens, window=50, quantum_extractor=quantum_extractor)
    
    print(f"Extracted {len(X)} context-aware feature vectors")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Augment with database memory
    X, y = db.get_augmented_data(X, y)
    print(f"Total samples after augmentation: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("\nTraining context-aware neural network...")
    
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=16,
        learning_rate='adaptive',
        learning_rate_init=0.01,
        max_iter=1,
        shuffle=True,
        random_state=42,
        early_stopping=False,
        validation_fraction=0.1,
        warm_start=True,
        verbose=False
    )

    epochs = 30
    loss_curve = []
    accuracy_curve = []
    
    clf.fit(X_train, y_train)
    
    best_loss = float('inf')
    patience = 0
    max_patience = 15
    
    for epoch in range(epochs):
        if epoch > 0 and epoch % 10 == 0:
            clf.alpha = min(clf.alpha * 1.2, 0.01)
            clf.learning_rate_init *= 0.9
        
        clf.partial_fit(X_train, y_train, classes=np.unique(y))
        
        train_pred_proba = clf.predict_proba(X_train)
        test_pred_proba = clf.predict_proba(X_test)
        
        train_loss = log_loss(y_train, train_pred_proba)
        test_loss = log_loss(y_test, test_pred_proba)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        loss_curve.append(test_loss)
        accuracy_curve.append(test_acc)
        
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

    y_pred = clf.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Final Context-Aware Model Accuracy: {final_acc:.4f}")
    print(f"Best Loss: {min(loss_curve):.6f}")
    print(f"{'='*60}")

    db.store(X_train, y_train, final_acc, loss_curve)
    
    return clf, X, y, scaler, contexts

# --- Load corpus ---

print("="*60)
print("CONTEXT-AWARE TEXT GENERATION")
print("="*60)

# Initialize quantum feature extractor
quantum_extractor = SchrodingerQuantumFeatures(hbar=1.0)

filename = input("\nEnter corpus filename: ").strip()
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
if KB_LEN > 0:
    text = text[:KB_LEN]

tokens = text.lower().split()
if len(tokens) < 100:
    raise ValueError("Corpus too short. Provide at least a few paragraphs.")

print(f"Loaded corpus with {len(tokens):,} tokens from '{filename}'.")

# --- Build context-aware N-gram model ---

def build_context_aware_ngram_model(tokens, n=N_GRAM_ORDER):
    """Build n-gram model with context tracking"""
    model = defaultdict(list)
    context_map = {}
    
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        next_word = tokens[i + n]
        model[key].append(next_word)
        
        # Track context position for coherence
        if key not in context_map:
            context_map[key] = []
        context_map[key].append(i)
    
    return model, context_map

ngram_model, context_map = build_context_aware_ngram_model(tokens, n=N_GRAM_ORDER)
model_keys = list(ngram_model.keys())
print(f"Context-aware N-gram model built with {len(model_keys):,} {N_GRAM_ORDER}-word keys.")

# --- Train the model ---
db = CurveMemoryDB()
clf, X, y, scaler, contexts = train_model_with_real_data(db, tokens, quantum_extractor)

# --- Context-aware generation with quantum coherence ---

def context_aware_inference_stream(model, context_map, model_keys, X_data, clf, 
                                   start_key, tokens, quantum_extractor, max_length=500):
    """Generate text with context awareness, coherence checking, and quantum features"""
    output = list(start_key)
    key_count = len(model_keys)
    
    if key_count == 0:
        return
    
    key = start_key
    generated_count = 0
    context_window = list(start_key)
    word_freq = Counter(tokens)
    total_words = len(tokens)
    
    while generated_count < max_length:
        candidates = model.get(key, [])
        
        if not candidates:
            # Find contextually similar n-grams
            similar_keys = [k for k in model_keys if any(w in k for w in key)]
            if similar_keys:
                key = similar_keys[np.random.randint(0, len(similar_keys))]
                candidates = model.get(key, [])
            else:
                fallback_key = model_keys[np.random.randint(0, key_count)]
                candidates = model.get(fallback_key, [])
        
        # Context-aware selection with quantum coherence
        if len(candidates) > 1:
            # Score candidates based on context coherence and quantum features
            candidate_scores = []
            for cand in set(candidates):
                # Traditional context score
                context_score = 0
                for w in context_window[-10:]:
                    context_score += tokens.count(w) * candidates.count(cand)
                
                # Quantum-enhanced scoring
                test_segment = context_window[-10:] + [cand]
                quantum_features = quantum_extractor.extract_quantum_features(
                    test_segment, word_freq, total_words
                )
                
                # Combine classical and quantum scores
                quantum_score = quantum_features['coherence'] * quantum_features['avg_probability']
                combined_score = context_score * (1 + quantum_score)
                
                candidate_scores.append((cand, combined_score))
            
            # Weighted random selection based on combined scores
            total_score = sum(score for _, score in candidate_scores) + 1e-10
            probs = [score / total_score for _, score in candidate_scores]
            next_word = np.random.choice([c for c, _ in candidate_scores], p=probs)
        else:
            next_word = candidates[0]
        
        output.append(next_word)
        context_window.append(next_word)
        
        # Update key for next iteration
        key = tuple(output[-N_GRAM_ORDER:])
        generated_count += 1
        
        yield next_word

# --- Main interactive generation ---
print("\n" + "="*60)
print("CONTEXT-AWARE TEXT GENERATION")
print("="*60)

while True:
    seed_input = input("USER: ").strip().lower()
    
    if seed_input == 'quit':
        break
    
    seed_tokens = re.findall(r'\b\w+\b', seed_input)
    if len(seed_tokens) < N_GRAM_ORDER:
        while len(seed_tokens) < N_GRAM_ORDER:
            seed_tokens.append(tokens[len(seed_tokens) % len(tokens)])
    
    start_key = tuple(seed_tokens[-N_GRAM_ORDER:])
    
    # Check if start_key exists in model
    if start_key not in ngram_model:
        print(f"Note: Seed '{' '.join(start_key)}' not found in corpus, using similar context...")
        # Find similar key
        similar = [k for k in model_keys if any(w in k for w in start_key)]
        if similar:
            start_key = similar[0]
        else:
            start_key = model_keys[0]
    
    stream = context_aware_inference_stream(
        ngram_model, context_map, model_keys, X, clf, start_key, tokens, 
        quantum_extractor, max_length=500
    )
    generated = []
    print("\n--- Context-Aware Generated Text ---\n")
    for _ in range(500):
        try:
            word = next(stream)
            print(word, end=' ', flush=True)
            generated.append(word)
        except StopIteration:
            break
    print("\n")
    
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
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cuda_stream = torch.cuda.Stream()
else:
    cuda_stream = None


# --- Answer-Based Q&A System ---
class AnswerBasedQASystem:
    """
    Q&A system that generates DIRECTLY FROM ANSWERS.
    Questions are ONLY for retrieval - generation uses answer content.
    """
    def __init__(self, device=device):
        self.device = device
        self.qa_pairs = []
        self.qa_index = {}
        self.answer_corpus = []
        self.answer_tokens = []
        self.answer_ngram_model = {}
        self.qa_cache_file = 'qa_answer_cache.pkl'
        print("📚 Answer-Based Q&A System initialized")

    def load_qa_dataset(self, dataset_name='squad', max_samples=5000):
        print(f"\n📖 Loading {dataset_name} (answer-focused)...")
        try:
            dataset = load_dataset('squad', split='train')
            print(f"Loading {max_samples:,} Q&A pairs, extracting ANSWERS...")

            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break

                question = example['question']
                answers = example['answers']
                context = example['context']

                if answers and 'text' in answers and len(answers['text']) > 0:
                    answer = answers['text'][0]
                else:
                    continue

                if question and answer:
                    self.qa_pairs.append({
                        'question': question.lower(),
                        'answer': answer.lower(),
                        'context': context.lower() if context else "",
                        'q_tokens': question.lower().split(),
                        'a_tokens': answer.lower().split()
                    })
                    self.answer_corpus.append(answer.lower())
                    self.answer_tokens.extend(answer.lower().split())

                if (i + 1) % 500 == 0:
                    print(f"  Loaded {i+1:,}...")

            print(f"\n✓ Loaded {len(self.qa_pairs):,} Q&A pairs")
            print(f"✓ Answer corpus: {len(self.answer_tokens):,} tokens")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def build_answer_ngram_model(self):
        print("\n🔨 Building ANSWER-ONLY n-gram model...")
        self.answer_ngram_model = defaultdict(list)

        for qa in self.qa_pairs:
            answer_tokens = qa['a_tokens']
            for i in range(len(answer_tokens) - N_GRAM_ORDER):
                key = tuple(answer_tokens[i:i+N_GRAM_ORDER])
                next_token = answer_tokens[i+N_GRAM_ORDER]
                self.answer_ngram_model[key].append(next_token)

        print(f"✓ Answer n-gram model: {len(self.answer_ngram_model):,} keys")
        return self.answer_ngram_model

    def build_qa_index(self):
        print("\n🔨 Building Q&A index...")
        self.qa_index = defaultdict(list)
        for idx, qa in enumerate(self.qa_pairs):
            for token in qa['q_tokens']:
                if len(token) > 3:
                    self.qa_index[token].append(idx)
            for token in qa['a_tokens']:
                if len(token) > 3:
                    self.qa_index[token].append(idx)
        print(f"✓ Indexed {len(self.qa_index):,} keywords")

    def comprehend_question(self, user_input):
        tokens = user_input.lower().split()
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose'}
        is_question = any(w in question_words for w in tokens)

        question_type = None
        for qw in question_words:
            if qw in tokens:
                question_type = qw
                break

        stop_words = {'the', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should', 'have', 'has', 'had'}
        key_entities = [w for w in tokens if len(w) > 3 and w not in stop_words]

        return {
            'is_question': is_question,
            'question_type': question_type,
            'key_entities': key_entities,
            'tokens': tokens
        }

    def retrieve_relevant_qa_pairs(self, comprehension_result, top_k=5):
        if not self.qa_pairs:
            return []

        key_entities = comprehension_result['key_entities']
        tokens = comprehension_result['tokens']
        qa_scores = []

        for qa in self.qa_pairs:
            score = 0.0
            q_tokens_set = set(qa['q_tokens'])
            a_tokens_set = set(qa['a_tokens'])

            for entity in key_entities:
                if entity in q_tokens_set:
                    score += 2.0
                if entity in a_tokens_set:
                    score += 1.5

            for token in tokens:
                if token in q_tokens_set:
                    score += 0.5
                if token in a_tokens_set:
                    score += 0.3

            if score > 0:
                qa_scores.append((qa, score))

        qa_scores.sort(key=lambda x: x[1], reverse=True)
        return qa_scores[:top_k]

    def get_answer_seed(self, relevant_qa_pairs):
        if not relevant_qa_pairs:
            return []
        top_qa, score = relevant_qa_pairs[0]
        answer_tokens = top_qa['a_tokens']
        return answer_tokens[:max(N_GRAM_ORDER, len(answer_tokens))]

    def extract_answer_patterns(self, relevant_qa_pairs):
        answer_tokens = []
        for qa, score in relevant_qa_pairs:
            for token in qa['a_tokens']:
                answer_tokens.extend([token] * int(score + 1))
        answer_freq = Counter(answer_tokens)
        patterns = [token for token, count in answer_freq.most_common(20)]
        return {'answer_tokens': answer_tokens, 'answer_patterns': patterns}

    def save_cache(self):
        cache_data = {
            'qa_pairs': self.qa_pairs,
            'qa_index': dict(self.qa_index),
            'answer_ngram_model': dict(self.answer_ngram_model),
            'answer_tokens': self.answer_tokens,
            'timestamp': time.time()
        }
        with open(self.qa_cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n✓ Q&A cache saved")

    def load_cache(self):
        if not os.path.exists(self.qa_cache_file):
            return False
        try:
            with open(self.qa_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.qa_pairs = cache_data['qa_pairs']
            self.qa_index = defaultdict(list, cache_data['qa_index'])
            self.answer_ngram_model = defaultdict(list, cache_data.get('answer_ngram_model', {}))
            self.answer_tokens = cache_data.get('answer_tokens', [])
            print(f"\n✓ Q&A cache loaded: {len(self.qa_pairs):,} pairs")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
# --- State Definitions ---
class SystemState(Enum):
    """FSM states for the text generation system"""
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
    """FSM states for token generation"""
    INIT = auto()
    SELECT_CANDIDATES = auto()
    COMPUTE_SCORES = auto()
    SAMPLE_TOKEN = auto()
    UPDATE_CONTEXT = auto()
    EMIT = auto()
    DONE = auto()

# --- Hamiltonian Training System ---
class HamiltonianTrainingSystem:
    """
    Separate training system that evolves Hamiltonian weights over time
    and saves cache when accuracy threshold is reached.
    """
    def __init__(self, accuracy_threshold=0.99, cache_file='hamiltonian_cache.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accuracy_threshold = accuracy_threshold
        self.cache_file = cache_file

        # Initialize Hamiltonian weights for system states
        self.system_hamiltonian = {
            'INIT': 1.0,
            'LOAD_CACHE': 2.0,
            'LOAD_DATASET': 3.0,
            'BUILD_NGRAM': 5.0,
            'EXTRACT_FEATURES': 6.0,
            'TRAIN_MODEL': 4.0,
            'SAVE_CACHE': 2.0,
            'READY': 1.5,
            'AWAIT_INPUT': 3.0,
            'GENERATE_TOKEN': 2.0,
            'OUTPUT_TOKEN': 0.0,
            'COMPLETE': 0.0,
            'ERROR': 10.0
        }

        # Initialize Hamiltonian weights for generator states
        self.generator_hamiltonian = {
            'INIT': 1.0,
            'SELECT_CANDIDATES': 2.0,
            'COMPUTE_SCORES': 3.0,
            'SAMPLE_TOKEN': 2.5,
            'UPDATE_CONTEXT': 2.0,
            'EMIT': 1.5,
            'DONE': 0.0
        }

        # Training history
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'system_hamiltonian_history': [],
            'generator_hamiltonian_history': [],
            'quantum_coherence': []
        }

        self.best_accuracy = 0.0
        self.epoch_count = 0

    def update_hamiltonian_from_loss(self, state_name, hamiltonian_dict, loss_value, accuracy):
        """
        Update Hamiltonian weights based on loss and accuracy.
        Lower loss and higher accuracy -> lower energy (more stable state)
        """
        # Adaptive learning rate for Hamiltonian
        learning_rate = 0.05

        # Energy update based on performance
        # Good performance (low loss, high acc) -> reduce energy
        # Bad performance -> increase energy
        performance_factor = loss_value / (accuracy + 1e-6)

        current_energy = hamiltonian_dict[state_name]
        new_energy = current_energy * (1.0 - learning_rate) + performance_factor * learning_rate

        # Clamp energy to reasonable bounds
        hamiltonian_dict[state_name] = np.clip(new_energy, 0.1, 10.0)

    def evolve_hamiltonians(self, train_loss, train_acc, test_loss, test_acc):
        """
        Evolve both system and generator Hamiltonians based on training metrics
        """
        # Update system Hamiltonian - focus on training pipeline states
        self.update_hamiltonian_from_loss('TRAIN_MODEL', self.system_hamiltonian, 
                                         train_loss, train_acc)
        self.update_hamiltonian_from_loss('EXTRACT_FEATURES', self.system_hamiltonian,
                                         train_loss, train_acc)

        # Update generator Hamiltonian - focus on generation quality
        self.update_hamiltonian_from_loss('COMPUTE_SCORES', self.generator_hamiltonian,
                                         test_loss, test_acc)
        self.update_hamiltonian_from_loss('SAMPLE_TOKEN', self.generator_hamiltonian,
                                         test_loss, test_acc)

        # Store history
        self.training_history['system_hamiltonian_history'].append(
            dict(self.system_hamiltonian)
        )
        self.training_history['generator_hamiltonian_history'].append(
            dict(self.generator_hamiltonian)
        )

    def train_with_hamiltonian_evolution(self, model, train_loader, test_loader, 
                                        criterion, optimizer, epochs=50):
        """
        Training loop that evolves Hamiltonians over time and saves when accuracy threshold is met
        """
        print("\n" + "="*80)
        print("HAMILTONIAN EVOLUTION TRAINING SYSTEM")
        print("="*80)
        print(f"Cache file: {self.cache_file}")
        print("="*80 + "\n")

        for epoch in range(epochs):
            self.epoch_count = epoch + 1

            # ============ TRAINING PHASE ============
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total

            # ============ VALIDATION PHASE ============
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()

            test_loss /= len(test_loader)
            test_accuracy = test_correct / test_total

            # ============ HAMILTONIAN EVOLUTION ============
            self.evolve_hamiltonians(train_loss, train_accuracy, test_loss, test_accuracy)

            # Compute quantum coherence metric
            coherence = self._compute_system_coherence()

            # Store metrics
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['test_accuracy'].append(test_accuracy)
            self.training_history['quantum_coherence'].append(coherence)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}]  "
                      f"Train Loss: {train_loss:.4f}  Train Acc: {train_accuracy:.4f}  "
                      f"Test Loss: {test_loss:.4f}  Test Acc: {test_accuracy:.4f}  "
                      f"Coherence: {coherence:.4f}")

                # Show key Hamiltonian energies
                print(f"  → System H: TRAIN_MODEL={self.system_hamiltonian['TRAIN_MODEL']:.3f}, "
                      f"EXTRACT_FEATURES={self.system_hamiltonian['EXTRACT_FEATURES']:.3f}")
                print(f"  → Generator H: COMPUTE_SCORES={self.generator_hamiltonian['COMPUTE_SCORES']:.3f}, "
                      f"SAMPLE_TOKEN={self.generator_hamiltonian['SAMPLE_TOKEN']:.3f}")


        print(f"Best accuracy achieved: {self.best_accuracy:.2%}")
        self.save_cache(model, optimizer)
        return False

    def _compute_system_coherence(self):
        """
        Compute system coherence from Hamiltonian variance
        Lower variance = higher coherence
        """
        sys_energies = np.array(list(self.system_hamiltonian.values()))
        gen_energies = np.array(list(self.generator_hamiltonian.values()))

        combined_variance = np.var(sys_energies) + np.var(gen_energies)
        coherence = 1.0 / (1.0 + combined_variance)

        return coherence

    def save_cache(self, model, optimizer):
        """
        Save complete cache including model, Hamiltonians, and training history
        """
        cache_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'system_hamiltonian': self.system_hamiltonian,
            'generator_hamiltonian': self.generator_hamiltonian,
            'training_history': self.training_history,
            'best_accuracy': self.best_accuracy,
            'epoch_count': self.epoch_count,
            'timestamp': time.time()
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        cache_size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
        print(f"\n✓ Hamiltonian cache saved: {cache_size_mb:.2f} MB")
        print(f"  - Model state: ✓")
        print(f"  - System Hamiltonian: {len(self.system_hamiltonian)} states")
        print(f"  - Generator Hamiltonian: {len(self.generator_hamiltonian)} states")
        print(f"  - Training history: {len(self.training_history['epochs'])} epochs")

    def load_cache(self):
        """
        Load cache and restore training state
        """
        if not os.path.exists(self.cache_file):
            print(f"Cache file not found: {self.cache_file}")
            return None

        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.system_hamiltonian = cache_data['system_hamiltonian']
        self.generator_hamiltonian = cache_data['generator_hamiltonian']
        self.training_history = cache_data['training_history']
        self.best_accuracy = cache_data['best_accuracy']
        self.epoch_count = cache_data['epoch_count']

        print(f"\n✓ Hamiltonian cache loaded from {self.cache_file}")
        print(f"  - Best accuracy: {self.best_accuracy:.2%}")
        print(f"  - Trained epochs: {self.epoch_count}")
        print(f"  - System coherence: {self.training_history['quantum_coherence'][-1]:.4f}")

        return cache_data


# --- Question-Answer Comprehension System ---

class QuantumStateSuperposition:
    """
    Implements quantum-inspired state superposition for FSMs.
    States exist in superposition: |ψ⟩ = Σ c_n|n⟩ where c_n are complex amplitudes.
    Measurement collapses to single state with probability |c_n|².
    """
    def __init__(self, states, hbar=1.0):
        """
        Initialize quantum state superposition
        Args:
            states: List of possible states (Enum values)
            hbar: Reduced Planck constant (controls decoherence)
        """
        self.states = list(states)
        self.n_states = len(states)
        self.hbar = hbar
        self.device = device

        # Initialize state amplitudes (complex-valued on GPU)
        # Start in equal superposition: |ψ⟩ = (1/√N) Σ|n⟩
        self.amplitudes = torch.ones(self.n_states, dtype=torch.complex64, device=self.device) / math.sqrt(self.n_states)

        # Phase angles for each state
        self.phases = torch.zeros(self.n_states, dtype=torch.float32, device=self.device)

        # Decoherence time constant
        self.decoherence_rate = 0.1

        print(f"⚛️ Quantum superposition initialized: {self.n_states} states in coherent superposition")

    def set_state_amplitude(self, state, amplitude, phase=0.0):
        """Set amplitude and phase for a specific state"""
        idx = self.states.index(state)
        self.amplitudes[idx] = amplitude * cmath.exp(1j * phase)
        self.phases[idx] = phase

    def mix_states(self, state_weights):
        """
        Create superposition by mixing states with given weights.
        Args:
            state_weights: dict mapping state -> weight
        """
        # Reset amplitudes
        self.amplitudes = torch.zeros(self.n_states, dtype=torch.complex64, device=self.device)

        # Normalize weights
        total_weight = sum(state_weights.values())
        for state, weight in state_weights.items():
            idx = self.states.index(state)
            normalized_weight = weight / total_weight

            # Add random phase for quantum interference
            phase = torch.rand(1, device=self.device).item() * 2 * math.pi
            self.amplitudes[idx] = math.sqrt(normalized_weight) * cmath.exp(1j * phase)
            self.phases[idx] = phase

        # Renormalize to ensure Σ|c_n|² = 1
        norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes)**2))
        self.amplitudes = self.amplitudes / norm

    def evolve(self, hamiltonian_weights=None, dt=0.1):
        """
        Time evolution under Schrödinger equation: i·ħ·∂|ψ⟩/∂t = Ĥ|ψ⟩
        Args:
            hamiltonian_weights: Energy weights for each state (Hamiltonian diagonal)
            dt: Time step
        """
        if hamiltonian_weights is None:
            # Default: equal energy states
            hamiltonian_weights = torch.ones(self.n_states, dtype=torch.float32, device=self.device)
        else:
            hamiltonian_weights = torch.tensor(hamiltonian_weights, dtype=torch.float32, device=self.device)

        # Time evolution: |ψ(t+dt)⟩ = exp(-i·Ĥ·dt/ħ)|ψ(t)⟩
        phase_evolution = torch.exp(-1j * hamiltonian_weights * dt / self.hbar)
        self.amplitudes = self.amplitudes * phase_evolution

        # Apply decoherence (gradual collapse to classical mixture)
        decoherence = torch.exp(torch.tensor(-self.decoherence_rate * dt, device=self.device))
        self.amplitudes = self.amplitudes * decoherence

        # Renormalize
        norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes)**2))
        if norm > 1e-10:
            self.amplitudes = self.amplitudes / norm

    def measure(self):
        """
        Collapse superposition to single state via measurement.
        Returns state with probability |c_n|²
        """
        # Compute probabilities from amplitude Born rule: P(n) = |c_n|²
        probabilities = torch.abs(self.amplitudes)**2
        probabilities = probabilities / torch.sum(probabilities)  # Renormalize

        # Sample state based on probabilities
        state_idx = torch.multinomial(probabilities.real, 1).item()

        # Collapse wavefunction to measured state
        collapsed_amplitudes = torch.zeros_like(self.amplitudes)
        collapsed_amplitudes[state_idx] = 1.0
        self.amplitudes = collapsed_amplitudes

        return self.states[state_idx]

    def get_probabilities(self):
        """Get current state probabilities without collapsing"""
        probabilities = torch.abs(self.amplitudes)**2
        return {state: prob.item() for state, prob in zip(self.states, probabilities)}

    def interfere(self, other_superposition, interference_strength=0.5):
        """
        Quantum interference between two superpositions.
        |ψ_total⟩ = α|ψ_1⟩ + β|ψ_2⟩
        """
        alpha = math.sqrt(interference_strength)
        beta = math.sqrt(1 - interference_strength)
        self.amplitudes = alpha * self.amplitudes + beta * other_superposition.amplitudes

        # Renormalize
        norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes)**2))
        if norm > 1e-10:
            self.amplitudes = self.amplitudes / norm

    def entangle(self, state1, state2, correlation=0.8):
        """
        Create entanglement between two states.
        |ψ⟩ = α|state1⟩ + β|state2⟩ with strong correlation
        """
        idx1 = self.states.index(state1)
        idx2 = self.states.index(state2)

        # Reset all amplitudes
        self.amplitudes = torch.zeros_like(self.amplitudes)

        # Create Bell-like state
        alpha = math.sqrt(correlation)
        beta = math.sqrt(1 - correlation)

        self.amplitudes[idx1] = alpha
        self.amplitudes[idx2] = beta * cmath.exp(1j * math.pi / 4)  # π/4 phase difference

        # Renormalize
        norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes)**2))
        self.amplitudes = self.amplitudes / norm

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
    """PyTorch Multi-Layer Perceptron for context-aware text classification"""
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
    """Cache preprocessed data for instant generation"""
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
        print(f"\n⚛️ Pre-computing quantum features for up to {max_segments:,} segments...")

        tokens = self.cache['tokens']
        word_freq = self.cache['word_freq']
        total_words = self.cache['total_words']
        segment_length = 10
        computed = 0

        # State machine pattern: process one segment per call
        def process_segment(i):
            nonlocal computed
            if i >= min(len(tokens) - segment_length, max_segments * segment_length):
                return None  # Done

            segment = tuple(tokens[i:i + segment_length])
            if segment not in self.cache['quantum_features_cache']:
                features = quantum_extractor.extract_quantum_features(
                    list(segment), word_freq, total_words
                )
                self.cache['quantum_features_cache'][segment] = features
                computed += 1

            return i + segment_length  # Next index

        # Execute state machine
        i = 0
        next_i = process_segment(i)
        while next_i is not None:
            next_i = process_segment(next_i)

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
        positions = torch.linspace(-13000, 13000, len(segment), device=self.device)

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

    # State machine pattern: process one window per call
    def process_window(i):
        if i >= len(tokens) - window:
            return None  # Done

        segment = tokens[i:i + window]
        if len(segment) < window:
            return i + window // 4  # Skip

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

        return i + window // 4  # Next window

    # Execute state machine
    i = 0
    next_i = process_window(i)
    while next_i is not None:
        next_i = process_window(next_i)

    features_array = np.array(features)
    labels_array = np.array(labels)

    return features_array, labels_array, contexts

# --- HAMILTONIAN TRAINING FUNCTION (REPLACES OLD TRAIN FUNCTION) ---
def train_pytorch_model(X_train, y_train, X_test, y_test, input_size, epochs=50):
    """
    Train PyTorch neural network with Hamiltonian evolution tracking
    """
    print("\n" + "="*60)
    print("TRAINING PYTORCH NEURAL NETWORK WITH HAMILTONIAN EVOLUTION")
    print("="*60)

    # Convert to PyTorch tensors DIRECTLY ON GPU
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    print(f"✓ All tensors loaded directly to GPU: {device}")
    print(f"  - Training data: {X_train_tensor.shape} on {X_train_tensor.device}")
    print(f"  - Test data: {X_test_tensor.shape} on {X_test_tensor.device}")

    # Create GPU-only DataLoader
    train_dataset = GPUTensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = GPUTensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=512, 
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0
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

    # Initialize Hamiltonian training system
    ham_trainer = HamiltonianTrainingSystem(
        accuracy_threshold=0.85,
        cache_file='hamiltonian_cache.pkl'
    )

    # Train with Hamiltonian evolution
    threshold_reached = ham_trainer.train_with_hamiltonian_evolution(
        model, train_loader, test_loader, criterion, optimizer, epochs=epochs
    )

    # Return model and trainer (which contains evolved Hamiltonians)
    return model, ham_trainer.training_history['test_loss'], ham_trainer.training_history['test_accuracy']

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

    # Train PyTorch model (with Hamiltonian evolution)
    model, loss_curve, accuracy_curve = train_pytorch_model(
        X_train, y_train, X_test, y_test, 
        input_size=X.shape[1], 
        epochs=50
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

        # State machine for dataset loading
        class LoadState:
            def __init__(self):
                self.i = 0
                self.all_text = []
                self.dataset_iter = iter(dataset)

        state = LoadState()

        def process_sample(state):
            if state.i >= max_samples:
                return None  # Done

            try:
                example = next(state.dataset_iter)
                question = example.get('question', '')
                response = example.get('model_response', '')
                reference = example.get('reference_answer', '')
                combined_text = f"{question} {response} {reference}"
                state.all_text.append(combined_text)

                if (state.i + 1) % 1000 == 0:
                    print(f"  Processed {state.i+1:,} samples...")

                state.i += 1
                return state.i
            except StopIteration:
                return None

        # Execute loading state machine
        next_i = process_sample(state)
        while next_i is not None:
            next_i = process_sample(state)

        corpus = " ".join(state.all_text)
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

    # State machine pattern for n-gram building
    def process_ngram(i):
        if i >= len(tokens) - n:
            return None  # Done

        key = tuple(tokens[i:i + n])
        next_word = tokens[i + n]
        model[key].append(next_word)

        if key not in context_map:
            context_map[key] = []
        context_map[key].append(i)

        return i + 1

    # Execute n-gram building state machine
    i = 0
    next_i = process_ngram(i)
    while next_i is not None:
        next_i = process_ngram(next_i)

    return model, context_map

# --- PURE GPU STATE MACHINE GENERATOR WITH QUANTUM MIXING ---
class TokenGeneratorFSM:
    """
    Finite State Machine for token generation with QUANTUM STATE MIXING.
    States exist in superposition and interfere quantum-mechanically.
    State transitions: INIT -> SELECT -> COMPUTE -> SAMPLE -> UPDATE -> EMIT -> (SELECT or DONE)
    """
    def __init__(self, model, model_keys, preprocessing_cache, quantum_extractor,
                 start_key, max_length=500, use_quantum_mixing=True, qa_context=None):
        self.model = model
        self.model_keys = model_keys
        self.preprocessing_cache = preprocessing_cache
        self.quantum_extractor = quantum_extractor
        self.start_key = start_key
        self.max_length = max_length
        self.use_quantum_mixing = use_quantum_mixing
        self.qa_context = qa_context  # Q&A context for boosting relevant tokens

        self.tokens = preprocessing_cache.cache['tokens']
        self.word_freq = preprocessing_cache.cache['word_freq']
        self.total_words = preprocessing_cache.cache['total_words']

        self.output = list(start_key)
        self.context_window = list(start_key)
        self.generated_count = 0
        self.key = start_key
        self.device = device

        # FSM state
        self.state = GeneratorState.INIT
        self.candidates = None
        self.selected_word = None

        # Quantum state superposition for FSM states
        if self.use_quantum_mixing:
            self.quantum_state = QuantumStateSuperposition(
                states=list(GeneratorState),
                hbar=1.0
            )
            # Initialize in superposition of INIT and SELECT states
            self.quantum_state.mix_states({
                GeneratorState.INIT: 10.7,
                GeneratorState.SELECT_CANDIDATES: 10.3
            })
            print("🌀 Quantum state mixing ENABLED for token generation")
        else:
            self.quantum_state = None

    def transition(self):
        """Execute one state transition with optional quantum mixing"""
        if self.use_quantum_mixing and self.quantum_state:
            # Evolve quantum state before transition
            # Hamiltonian weights based on state "energy"
            hamiltonian = [
                1.0,   # INIT
                2.0,   # SELECT_CANDIDATES
                3.0,   # COMPUTE_SCORES
                2.5,   # SAMPLE_TOKEN
                2.0,   # UPDATE_CONTEXT
                1.5,   # EMIT
                0.0    # DONE
            ]
            self.quantum_state.evolve(hamiltonian_weights=hamiltonian, dt=0.75)

        if self.state == GeneratorState.INIT:
            return self._state_init()
        elif self.state == GeneratorState.SELECT_CANDIDATES:
            return self._state_select_candidates()
        elif self.state == GeneratorState.COMPUTE_SCORES:
            return self._state_compute_scores()
        elif self.state == GeneratorState.SAMPLE_TOKEN:
            return self._state_sample_token()
        elif self.state == GeneratorState.UPDATE_CONTEXT:
            return self._state_update_context()
        elif self.state == GeneratorState.EMIT:
            return self._state_emit()
        elif self.state == GeneratorState.DONE:
            return None
        else:
            raise ValueError(f"Unknown state: {self.state}")

    def _state_init(self):
        """Initialize generation"""
        if self.generated_count >= self.max_length:
            self.state = GeneratorState.DONE
            if self.use_quantum_mixing:
                # Collapse to DONE state
                self.quantum_state.set_state_amplitude(GeneratorState.DONE, 1.0)
            return None

        self.state = GeneratorState.SELECT_CANDIDATES
        if self.use_quantum_mixing:
            # Create superposition between SELECT and COMPUTE states
            self.quantum_state.mix_states({
                GeneratorState.SELECT_CANDIDATES: 0.6,
                GeneratorState.COMPUTE_SCORES: 0.4
            })
        return self.transition()

    def _state_select_candidates(self):
        """Select candidate words"""
        self.candidates = self.model.get(self.key, [])

        if not self.candidates:
            # Fallback to similar keys using GPU operations
            key_tensor = torch.tensor([hash(w) % 100000 for w in self.key], device=self.device)
            similarities = []
            for k in self.model_keys:
                k_tensor = torch.tensor([hash(w) % 100000 for w in k], device=self.device)
                sim = torch.mean((key_tensor - k_tensor).float().abs())
                similarities.append(sim.item())

            best_idx = torch.argmin(torch.tensor(similarities)).item()
            self.key = self.model_keys[best_idx]
            self.candidates = self.model.get(self.key, [])

            if not self.candidates:
                # Ultimate fallback
                fallback_key = self.model_keys[torch.randint(0, len(self.model_keys), (1,)).item()]
                self.candidates = self.model.get(fallback_key, [])

                if not self.candidates:
                    self.selected_word = self.tokens[torch.randint(0, len(self.tokens), (1,), device=self.device).item()]
                    self.state = GeneratorState.UPDATE_CONTEXT
                    if self.use_quantum_mixing:
                        self.quantum_state.set_state_amplitude(GeneratorState.UPDATE_CONTEXT, 1.0)
                    return self.transition()

        self.state = GeneratorState.COMPUTE_SCORES
        if self.use_quantum_mixing:
            # Entangle COMPUTE and SAMPLE states (strong correlation)
            self.quantum_state.entangle(
                GeneratorState.COMPUTE_SCORES,
                GeneratorState.SAMPLE_TOKEN,
                correlation=0.85
            )

        return self.transition()

    @torch.no_grad()
    def _state_compute_scores(self):
        """Compute scores for candidates using GPU"""
        unique_candidates = list(set(self.candidates))

        # Compute context similarity scores
        context_tail = self.context_window[-10:]

        cand_lens = torch.tensor([len(c) for c in unique_candidates], dtype=torch.float32, device=self.device)
        ctx_lens = torch.tensor([len(c) for c in context_tail], dtype=torch.float32, device=self.device)
        ctx_mean = ctx_lens.mean()

        similarity_scores = 1.0 / (1.0 + (cand_lens - ctx_mean).abs())

        # Get quantum coherence scores
        coherence_scores = []
        for cand in unique_candidates:
            test_segment = self.context_window[-10:] + [cand]
            q = self.preprocessing_cache.get_quantum_features(
                test_segment, self.quantum_extractor, self.word_freq, self.total_words
            )
            coherence_scores.append(q['coherence'] * q['avg_probability'])

        coherence_tensor = torch.tensor(coherence_scores, dtype=torch.float32, device=self.device)

        # Apply Q&A context boosting if available
        qa_boost = torch.ones_like(similarity_scores)
        if self.qa_context and self.qa_context['answer_patterns']:
            answer_patterns = set(self.qa_context['answer_patterns'])
            answer_tokens = set(self.qa_context['answer_tokens'])

            for idx, cand in enumerate(unique_candidates):
                # Strong boost for answer patterns
                if cand in answer_patterns:
                    qa_boost[idx] = 2.5
                # Moderate boost for answer tokens
                elif cand in answer_tokens:
                    qa_boost[idx] = 1.5

        # Combine scores with quantum interference if enabled
        if self.use_quantum_mixing:
            # Get quantum state probabilities
            state_probs = self.quantum_state.get_probabilities()
            compute_prob = state_probs.get(GeneratorState.COMPUTE_SCORES, 1.0)

            # Modulate scores by quantum probability, Q&A boost
            combined_scores = similarity_scores * (1.0 + coherence_tensor) * compute_prob * qa_boost
        else:
            combined_scores = similarity_scores * (1.0 + coherence_tensor) * qa_boost

        self.probs = torch.softmax(combined_scores, dim=0)
        self.unique_candidates = unique_candidates

        self.state = GeneratorState.SAMPLE_TOKEN
        if self.use_quantum_mixing:
            # Measure quantum state (collapse to SAMPLE)
            measured_state = self.quantum_state.measure()
            if measured_state != GeneratorState.SAMPLE_TOKEN:
                # Quantum measurement forced different state - create superposition
                self.quantum_state.mix_states({
                    GeneratorState.SAMPLE_TOKEN: 0.7,
                    measured_state: 0.3
                })

        return self.transition()

    def _state_sample_token(self):
        """Sample token from probability distribution"""
        choice_idx = torch.multinomial(self.probs, 1).item()
        self.selected_word = self.unique_candidates[choice_idx]

        self.state = GeneratorState.UPDATE_CONTEXT
        if self.use_quantum_mixing:
            # Transition to UPDATE with phase coherence
            self.quantum_state.mix_states({
                GeneratorState.UPDATE_CONTEXT: 0.8,
                GeneratorState.EMIT: 0.2
            })
        return self.transition()

    def _state_update_context(self):
        """Update context window and key"""
        self.output.append(self.selected_word)
        self.context_window.append(self.selected_word)
        self.key = tuple(self.output[-N_GRAM_ORDER:])
        self.generated_count += 1

        self.state = GeneratorState.EMIT
        if self.use_quantum_mixing:
            # Strong transition to EMIT
            self.quantum_state.set_state_amplitude(GeneratorState.EMIT, 1.0)
        return self.transition()

    def _state_emit(self):
        """Emit the selected word"""
        word = self.selected_word

        # Check if we should continue
        if self.generated_count >= self.max_length:
            self.state = GeneratorState.DONE
            if self.use_quantum_mixing:
                self.quantum_state.set_state_amplitude(GeneratorState.DONE, 1.0)
        else:
            self.state = GeneratorState.SELECT_CANDIDATES
            if self.use_quantum_mixing:
                # Create superposition for next iteration
                self.quantum_state.mix_states({
                    GeneratorState.SELECT_CANDIDATES: 0.5,
                    GeneratorState.COMPUTE_SCORES: 0.3,
                    GeneratorState.SAMPLE_TOKEN: 0.2
                })

        return word

    def generate_all(self):
        """Generate all tokens using state machine"""
        results = []
        result = self.transition()
        while result is not None:
            results.append(result)
            result = self.transition()
        return results

# --- System-Level FSM WITH QUANTUM MIXING ---
class TextGenerationSystemFSM:
    """Main system state machine with quantum state mixing and Q&A comprehension"""
    def __init__(self, use_quantum_mixing=True, use_qa_system=True):
        self.state = SystemState.INIT
        self.quantum_extractor = None
        self.preprocessing_cache = None
        self.tokens = None
        self.model_keys = None
        self.use_quantum_mixing = use_quantum_mixing
        self.use_qa_system = use_qa_system

        # Q&A Comprehension System
        self.qa_system = None
        self.current_comprehension = None
        self.current_qa_context = None

        # Quantum state superposition for system states
        if use_quantum_mixing:
            self.quantum_state = QuantumStateSuperposition(
                states=list(SystemState),
                hbar=1.0
            )
            print("🌌 System-level quantum state mixing ENABLED")

    def transition(self):
        """Execute one system state transition with quantum mixing"""
        if self.use_quantum_mixing:
            # Evolve system quantum state
            system_hamiltonian = [
                1.0,   # INIT
                2.0,   # LOAD_CACHE
                3.0,   # LOAD_DATASET
                4.0,   # BUILD_NGRAM
                5.0,   # EXTRACT_FEATURES
                6.0,   # TRAIN_MODEL
                4.0,   # SAVE_CACHE
                2.0,   # READY
                1.5,   # AWAIT_INPUT
                3.0,   # GENERATE_TOKEN
                2.0,   # OUTPUT_TOKEN
                0.0,   # COMPLETE
                10.0   # ERROR (high energy - avoid)
            ]
            self.quantum_state.evolve(hamiltonian_weights=system_hamiltonian, dt=0.7)

        if self.state == SystemState.INIT:
            return self._state_init()
        elif self.state == SystemState.LOAD_CACHE:
            return self._state_load_cache()
        elif self.state == SystemState.LOAD_DATASET:
            return self._state_load_dataset()
        elif self.state == SystemState.BUILD_NGRAM:
            return self._state_build_ngram()
        elif self.state == SystemState.EXTRACT_FEATURES:
            return self._state_extract_features()
        elif self.state == SystemState.TRAIN_MODEL:
            return self._state_train_model()
        elif self.state == SystemState.SAVE_CACHE:
            return self._state_save_cache()
        elif self.state == SystemState.READY:
            return self._state_ready()
        elif self.state == SystemState.AWAIT_INPUT:
            return self._state_await_input()
        elif self.state == SystemState.GENERATE_TOKEN:
            return self._state_generate_token()
        elif self.state == SystemState.OUTPUT_TOKEN:
            return self._state_output_token()
        elif self.state == SystemState.COMPLETE:
            return 'done'
        else:
            raise ValueError(f"Unknown state: {self.state}")

    def _state_init(self):
        """Initialize system"""
        print("="*60)
        print("CONTEXT-AWARE TEXT GENERATION WITH QUANTUM FEATURES")
        print("="*60)

        self.quantum_extractor = SchrodingerQuantumFeatures(hbar=1.0)
        self.preprocessing_cache = PreprocessingCache(cache_file='preprocessing_cache.pkl')

        # Initialize Q&A comprehension system
        if self.use_qa_system:
            self.qa_system = AnswerBasedQASystem(device=device)

        self.state = SystemState.LOAD_CACHE
        if self.use_quantum_mixing:
            self.quantum_state.mix_states({
                SystemState.LOAD_CACHE: 0.6,
                SystemState.LOAD_DATASET: 0.4
            })
        return 'continue'

    def _state_load_cache(self):
        """Try to load cache"""
        cache_loaded = self.preprocessing_cache.load()

        # Try to load Q&A cache
        qa_loaded = False
        if self.use_qa_system and self.qa_system:
            qa_loaded = self.qa_system.load_cache()

        if cache_loaded:
            self.tokens = self.preprocessing_cache.cache['tokens']
            self.model_keys = self.preprocessing_cache.cache['model_keys']

            # Load Q&A dataset if cache not found
            if self.use_qa_system and not qa_loaded:
                print("\nLoading Q&A dataset...")
                self.qa_system.load_qa_dataset('squad', max_samples=50000)
                self.qa_system.build_qa_index()
                self.qa_system.save_cache()

            print("\n⚡ Using cached preprocessing data - ready for instant generation!")
            self.state = SystemState.READY
            if self.use_quantum_mixing:
                self.quantum_state.set_state_amplitude(SystemState.READY, 1.0)
        else:
            print("\n🔧 No cache found. Running initial setup...")
            self.state = SystemState.LOAD_DATASET
            if self.use_quantum_mixing:
                self.quantum_state.set_state_amplitude(SystemState.LOAD_DATASET, 1.0)

        return 'continue'

    def _state_load_dataset(self):
        """Load dataset"""
        use_hf_dataset = input("\nUse Facebook Natural Reasoning dataset? (y/n): ").strip().lower()

        if use_hf_dataset == 'y':
            self.tokens = load_facebook_reasoning_dataset(max_samples=max_samples)
            if self.tokens is None:
                filename = input("\nEnter corpus filename: ").strip()
                with open(filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                if KB_LEN > 0:
                    text = text[:KB_LEN]
                self.tokens = text.lower().split()
        else:
            filename = input("\nEnter corpus filename: ").strip()
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            if KB_LEN > 0:
                text = text[:KB_LEN]
            self.tokens = text.lower().split()

        if len(self.tokens) < 100:
            print("✗ Corpus too short. Provide at least a few paragraphs.")
            self.state = SystemState.ERROR
            if self.use_quantum_mixing:
                self.quantum_state.set_state_amplitude(SystemState.ERROR, 1.0)
            return 'error'

        print(f"\n✓ Loaded corpus with {len(self.tokens):,} tokens.")

        # Load Q&A dataset
        if self.use_qa_system and self.qa_system:
            self.qa_system.load_qa_dataset('squad', max_samples=50000)
            self.qa_system.build_qa_index()
            self.qa_system.save_cache()

        self.state = SystemState.BUILD_NGRAM
        if self.use_quantum_mixing:
            self.quantum_state.mix_states({
                SystemState.BUILD_NGRAM: 0.7,
                SystemState.EXTRACT_FEATURES: 0.3
            })
        return 'continue'

    def _state_build_ngram(self):
        """Build n-gram model"""
        print("\n🔨 Building n-gram model...")
        ngram_model, context_map = build_context_aware_ngram_model(self.tokens, n=N_GRAM_ORDER)
        self.model_keys = list(ngram_model.keys())
        print(f"✓ N-gram model built with {len(self.model_keys):,} {N_GRAM_ORDER}-word keys.")

        word_freq = Counter(self.tokens)
        total_words = len(self.tokens)

        self.preprocessing_cache.store_preprocessing(
            self.tokens, ngram_model, context_map, self.model_keys, word_freq, total_words
        )

        self.preprocessing_cache.precompute_quantum_features(self.quantum_extractor, max_segments=max_segments)

        self.state = SystemState.TRAIN_MODEL
        if self.use_quantum_mixing:
            self.quantum_state.set_state_amplitude(SystemState.TRAIN_MODEL, 1.0)
        return 'continue'

    def _state_extract_features(self):
        """Extract features (merged into train_model)"""
        self.state = SystemState.TRAIN_MODEL
        return 'continue'

    def _state_train_model(self):
        """Train neural network model"""
        model, X, y, scaler, contexts = train_model_with_real_data(self.tokens, self.quantum_extractor)

        self.state = SystemState.SAVE_CACHE
        if self.use_quantum_mixing:
            self.quantum_state.set_state_amplitude(SystemState.SAVE_CACHE, 1.0)
        return 'continue'

    def _state_save_cache(self):
        """Save preprocessing cache"""
        self.preprocessing_cache.save()
        print("\n✓ Setup complete! Next run will be instant.")
        self.state = SystemState.READY
        if self.use_quantum_mixing:
            self.quantum_state.set_state_amplitude(SystemState.READY, 1.0)
        return 'continue'

    def _state_ready(self):
        """System ready for generation"""
        print("\n" + "="*60)
        print("CONTEXT-AWARE TEXT GENERATION")
        print("="*60)
        self.state = SystemState.AWAIT_INPUT
        if self.use_quantum_mixing:
            self.quantum_state.set_state_amplitude(SystemState.AWAIT_INPUT, 1.0)
        return 'continue'

    def _state_await_input(self):
        """Wait for user input and generate FROM ANSWERS"""
        seed_input = input("\nUSER: ").strip()

        if seed_input.lower() == 'quit':
            self.state = SystemState.COMPLETE
            if self.use_quantum_mixing:
                self.quantum_state.set_state_amplitude(SystemState.COMPLETE, 1.0)
            return 'done'

        if self.use_qa_system and self.qa_system and self.qa_system.qa_pairs:
            print("\n🧠 Comprehending question (for retrieval only)...")
            comprehension = self.qa_system.comprehend_question(seed_input)

            if comprehension['is_question']:
                print(f"  Type: {comprehension['question_type']}")
                print(f"  Entities: {', '.join(comprehension['key_entities'][:3])}")

            print("\n📚 Retrieving Q&A pairs...")
            relevant_qa = self.qa_system.retrieve_relevant_qa_pairs(comprehension, top_k=5)

            if relevant_qa:
                top_qa, top_score = relevant_qa[0]
                print(f"  Best match:Q: {top_qa['question'][:60]}...A: {top_qa['answer'][:60]}... Score: {top_score:.1f}")

                answer_seed = self.qa_system.get_answer_seed(relevant_qa)
                print(f"\n  💡 Answer seed: {' '.join(answer_seed[:5])}")

                answer_context = self.qa_system.extract_answer_patterns(relevant_qa)
                self.current_qa_context = answer_context
                self.current_qa_context['answer_seed'] = answer_seed

                if answer_seed and len(answer_seed) >= N_GRAM_ORDER:
                    print(f"\n  ✨ Generating FROM answer: {' '.join(answer_seed[:N_GRAM_ORDER])}")
                    seed_tokens = answer_seed[:N_GRAM_ORDER]
                else:
                    seed_tokens = re.findall(r'\b\w+\b', seed_input.lower())[:N_GRAM_ORDER]
            else:
                seed_tokens = re.findall(r'\b\w+\b', seed_input.lower())[:N_GRAM_ORDER]
                self.current_qa_context = None
        else:
            seed_tokens = re.findall(r'\b\w+\b', seed_input.lower())[:N_GRAM_ORDER]
            self.current_qa_context = None

        while len(seed_tokens) < N_GRAM_ORDER:
            seed_tokens.append(self.tokens[len(seed_tokens) % len(self.tokens)])

        self.start_key = tuple(seed_tokens[:N_GRAM_ORDER])
        print(f"\n  🌱 Seed: {' '.join(self.start_key)}")

        self.state = SystemState.GENERATE_TOKEN
        if self.use_quantum_mixing:
            self.quantum_state.set_state_amplitude(SystemState.GENERATE_TOKEN, 1.0)
        return 'continue'
    def _state_generate_token(self):
        """Generate tokens using FSM with quantum mixing"""
        # Create generator FSM with quantum mixing
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                generator_fsm = TokenGeneratorFSM(
                    self.preprocessing_cache.cache['ngram_model'],
                    self.model_keys,
                    self.preprocessing_cache,
                    self.quantum_extractor,
                    self.start_key,
                    max_length=500,
                    use_quantum_mixing=self.use_quantum_mixing,
                    qa_context=self.current_qa_context
                )
        else:
            generator_fsm = TokenGeneratorFSM(
                self.preprocessing_cache.cache['ngram_model'],
                self.model_keys,
                self.preprocessing_cache,
                self.quantum_extractor,
                self.start_key,
                max_length=500,
                use_quantum_mixing=self.use_quantum_mixing,
                qa_context=self.current_qa_context
            )

        print("\n--- Context-Aware Generated Text (Quantum-Mixed States) ---\n")
        generated = list(self.current_qa_context['answer_seed'])
        # Generate all tokens using FSM
        generated.append(' '.join(list(generator_fsm.generate_all())))

        # Output
        for word in generated:
            print(word, end=' ', flush=True)
        print("\n")

        self.state = SystemState.AWAIT_INPUT
        if self.use_quantum_mixing:
            self.quantum_state.set_state_amplitude(SystemState.AWAIT_INPUT, 1.0)
        return 'continue'

    def _state_output_token(self):
        """Output token (merged into generate_token)"""
        self.state = SystemState.AWAIT_INPUT
        return 'continue'

# --- Main Execution (State Machine Driver) ---
def main():
    """Main entry point - drives the system FSM with quantum mixing and Q&A"""
    # Enable/disable quantum state mixing and Q&A system
    USE_QUANTUM_MIXING = True
    USE_QA_SYSTEM = True

    system_fsm = TextGenerationSystemFSM(
        use_quantum_mixing=USE_QUANTUM_MIXING,
        use_qa_system=USE_QA_SYSTEM
    )
    result = system_fsm.transition()

    while result == 'continue':
        result = system_fsm.transition()

    if result == 'done':
        print("\n✓ System shutdown complete.")
    elif result == 'error':
        print("\n✗ System encountered an error.")

if __name__ == "__main__":
    main()

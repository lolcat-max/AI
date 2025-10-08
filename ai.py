import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import re
import sys

sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2  # Change this to test different n-gram orders
KB_LEN = 9999

# --- Signal Processing and ML ---

def generate_synthetic_output(n_samples=10000, freq=3.0, noise=0.3):
    x = np.linspace(0, 4 * np.pi, n_samples)
    base = np.sin(freq * x)
    mod = np.cos(0.7 * freq * x)
    signal = base * mod
    signal += noise * np.random.randn(n_samples)
    return signal

def add_scintillators(signal, num_spikes=50, spike_height=5.0, spike_width=5):
    """
    Add sharp scintillation spikes randomly within the signal.
    signal: np.array, base signal wave
    num_spikes: int, number of spikes to add
    spike_height: float, amplitude of the spikes
    spike_width: int, width of each spike in samples
    Returns modified signal with added spikes.
    """
    signal = signal.copy()
    n_samples = len(signal)

    spike_positions = np.random.choice(np.arange(spike_width, n_samples - spike_width), num_spikes, replace=False)

    for pos in spike_positions:
        start = max(pos - spike_width // 2, 0)
        end = min(pos + spike_width // 2, n_samples)
        # Add a simple triangular spike shape
        peak_len = end - start
        half_peak = peak_len // pos
        spike_shape = np.linspace(0, spike_height, half_peak)
        spike_shape = np.concatenate([spike_shape, spike_shape[::-1]])
        if len(spike_shape) < peak_len:  # If odd number, add one more peak element
            spike_shape = np.append(spike_shape, 0)
        signal[start:start+len(spike_shape)] += spike_shape

    return signal

def half_wave_interference(signal):
    return np.exp(signal)

def extract_features(signal, window=150):
    features = []
    for i in range(0, len(signal) - window, window):
        seg = signal[i:i + window]
        mean = np.mean(seg)
        var = np.var(seg)
        features.append([mean, var])
    return np.array(features)

def extract_scintillator_features(signal, window=150, threshold=1.0):
    """
    Extract scintillator-based features from signal windows:
    - Count of spikes above threshold
    - Mean spike amplitude (average height above threshold)
    Returns numpy array of features [spike_count, mean_amplitude] per window.
    """
    features = []
    for i in range(0, len(signal) - window, window):
        seg = signal[i:i + window]
        spikes = seg[seg > threshold] - threshold
        spike_count = len(spikes)
        mean_amplitude = np.mean(spikes) if spike_count > 0 else 0.0
        features.append([spike_count, mean_amplitude])
    return np.array(features)

# Generate base signal and add scintillators
signal_output = generate_synthetic_output(n_samples=10000)
signal_with_scintillators = add_scintillators(signal_output, num_spikes=60, spike_height=8.0, spike_width=7)
signal_input = half_wave_interference(signal_with_scintillators)

# Extract features: base features and scintillator features
basic_features = extract_features(signal_input, window=150)
scint_features = extract_scintillator_features(signal_with_scintillators, window=150, threshold=2.0)

# Concatenate
X = np.hstack([basic_features, scint_features])

# Create labels using combined threshold criteria
y = ((X[:, 0] > 0.9) | (X[:, 2] > 5)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = LogisticRegression(max_iter=15000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model test accuracy: {acc:.2f}")

# --- Load corpus ---

filename = input("\nEnter corpus filename: ").strip()
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
if KB_LEN > 0:
    text = text[:KB_LEN]

tokens = text.split()
if len(tokens) < 50:
    raise ValueError("Corpus too short. Provide at least a few paragraphs.")

print(f"Loaded corpus with {len(tokens):,} tokens from '{filename}'.")

# --- Build simple N-gram model (supports any order) ---

def build_ngram_model(tokens, n=N_GRAM_ORDER):
    model = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        next_word = tokens[i + n]
        model[key].append(next_word)
    return model

ngram_model = build_ngram_model(tokens, n=N_GRAM_ORDER)
model_keys = list(ngram_model.keys())
print(f"N-gram model built with {len(model_keys):,} {N_GRAM_ORDER}-word keys.")

# --- 2D half-wave mixing helper ---

def half_wave_rectify(matrix):
    return np.mean(matrix)

def two_d_half_wave_mix(mat1, mat2, alpha=0.1):
    mixed = alpha * mat1 + (1 - alpha) * mat2
    return half_wave_rectify(mixed)

# --- Streaming nonlinear 2D inference generator with half-wave mixing---

def nonlinear_2d_inference_stream(model, model_keys, X_data, clf,
                                  start_key, hidden_dim=16):
    output = list(start_key)
    key_count = len(model_keys)
    if key_count == 0:
        return

    key = start_key

    vocab_list = sorted(set(w for succs in model.values() for w in succs))
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    inf_state_1 = np.zeros((hidden_dim, 1))
    inf_state_2 = np.zeros((hidden_dim, 1))

    W1 = np.random.randn(hidden_dim, 2) * 0.1
    U1 = np.random.randn(hidden_dim, hidden_dim) * 0.1
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
    U2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
    V = np.random.randn(hidden_dim, len(vocab_list)) * 0.1

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    i = 0
    while True:
        sample = X_data[i % len(X_data)]
        mean_val = sample[0]
        label = clf.predict([sample])[0]

        x_vec = np.array([[mean_val], [label]])
        h1_in = np.dot(W1, x_vec) + np.dot(U1, inf_state_1)
        h1 = sigmoid(h1_in)

        inf_state_1 = two_d_half_wave_mix(inf_state_1, h1, alpha=0.6)

        h2_in = np.dot(W2, inf_state_1) + np.dot(U2, inf_state_2)
        h2 = sigmoid(h2_in)

        inf_state_2 = two_d_half_wave_mix(inf_state_2, h2, alpha=0.8)

        logits = np.dot(V.T, inf_state_2).flatten()
        e_logits = np.exp(logits - np.max(logits))  # for numerical stability
        probs = e_logits / e_logits.sum()

        candidates = model.get(key, [])
        
        if not candidates:
            fallback_key = model_keys[int(abs(mean_val) * 1000) % key_count]
            candidates = [fallback_key[-1]]

        mask = np.zeros_like(probs)
        for c in candidates:
            if c in word_to_idx:
                mask[word_to_idx[c]] = 1

        masked_probs = probs * mask
        total = masked_probs.sum()

        if total == 0:
            valid_idxs = [word_to_idx[c] for c in candidates if c in word_to_idx]
            masked_probs = np.zeros_like(probs)
            if valid_idxs:
                for idx in valid_idxs:
                    masked_probs[idx] = 1.0 / len(valid_idxs)
            else:
                masked_probs = np.ones_like(probs) / len(probs)
        else:
            masked_probs /= total

        next_idx = np.random.choice(len(masked_probs), p=masked_probs)
        next_word = idx_to_word[next_idx]

        output.append(next_word)
        key = tuple(output[-N_GRAM_ORDER:])

        i += 1
        yield next_word

# --- Main interactive generation session ---
while True:
    print("\nEnter your seed text:")
    seed_input = input().strip().lower()
    seed_tokens = re.findall(r'\b\w+\b', seed_input)
    if len(seed_tokens) < N_GRAM_ORDER:
        while len(seed_tokens) < N_GRAM_ORDER:
            seed_tokens.append(tokens[len(seed_tokens) % len(tokens)])

    start_key = tuple(seed_tokens[-N_GRAM_ORDER:])
    stream = nonlinear_2d_inference_stream(ngram_model, model_keys, X, clf, start_key, hidden_dim=360)

    print("\n--- Streaming generated text ---\n")
    for _ in range(500):
        try:
            print(next(stream), end=' ', flush=True)
        except StopIteration:
            break
    print("\n")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import re
import sys

sys.setrecursionlimit(1_000_000)
N_GRAM_ORDER = 2  # Change this to test different n-gram orders
KB_LEN = -1

# --- Signal Processing and ML ---

def generate_synthetic_output(n_samples=10000, freq=3.0, noise=0.3):
    x = np.linspace(0, 4 * np.pi, n_samples)
    base = np.sin(freq * x)
    mod = np.cos(0.7 * freq * x)
    signal = base * mod
    signal += noise * np.random.randn(n_samples)
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

signal_output = generate_synthetic_output(n_samples=10000)
signal_input = half_wave_interference(signal_output)
X = extract_features(signal_input)
y = (X[:, 0] > 0.9).astype(int)

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

# --- Memory map topology matrix shifting ---

def shift_matrix(mat, shift_x=1, shift_y=0):
    # Cyclic shift matrix along x and y axes (topology shifting)
    return np.roll(np.roll(mat, shift_x, axis=0), shift_y, axis=1)

# --- Streaming nonlinear 2D inference generator with half-wave mixing and matrix shifting ---

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

    # Initialize hidden states as 2D matrices for topology shifting
    # Use square form for simplicity; adjust if hidden_dim not perfect square
    state_dim = int(np.sqrt(hidden_dim))
    if state_dim * state_dim != hidden_dim:
        print(f"Warning: hidden_dim {hidden_dim} not a perfect square, adjusting to {state_dim**2}")
        hidden_dim = state_dim**2

    inf_state_1 = np.zeros((state_dim, state_dim))
    inf_state_2 = np.zeros((state_dim, state_dim))

    # Initialize random weight matrices for linear transformations
    # Flatten states for dot products, then reshape back
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

        # Flatten inf_state_1 for matrix operations
        inf_state_1_flat = inf_state_1.flatten()[:, None]
        h1_in = np.dot(W1, x_vec) + np.dot(U1, inf_state_1_flat)
        h1 = sigmoid(h1_in)
        h1_matrix = h1.reshape((state_dim, state_dim))

        # Apply matrix shift to inf_state_1 before mixing
        shifted_inf_state_1 = shift_matrix(inf_state_1, shift_x=1, shift_y=0)
        # Update inf_state_1 with half-wave mixing of shifted state and h1
        inf_state_1 = shift_matrix(shifted_inf_state_1, shift_x=1, shift_y=0)
        inf_state_1_val = two_d_half_wave_mix(inf_state_1, h1_matrix, alpha=0.6)

        # For inference state 2
        inf_state_2_flat = inf_state_2.flatten()[:, None]
        inf_state_1_flat = inf_state_1.flatten()[:, None]
        h2_in = np.dot(W2, inf_state_1_flat) + np.dot(U2, inf_state_2_flat)
        h2 = sigmoid(h2_in)
        h2_matrix = h2.reshape((state_dim, state_dim))

        # Apply matrix shift to inf_state_2 before mixing
        shifted_inf_state_2 = shift_matrix(inf_state_2, shift_x=0, shift_y=1)
        inf_state_2 = shift_matrix(shifted_inf_state_2, shift_x=0, shift_y=1)
        inf_state_2_val = two_d_half_wave_mix(inf_state_2, h2_matrix, alpha=0.8)

        # Use mixed scalar values to update states as constant arrays
        inf_state_1.fill(inf_state_1_val)
        inf_state_2.fill(inf_state_2_val)

        # Compute logits and convert to probabilities
        logits = np.dot(V.T, inf_state_2.flatten()).flatten()
        e_logits = np.exp(logits * np.max(logits))
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
    stream = nonlinear_2d_inference_stream(ngram_model, model_keys, X, clf, start_key, hidden_dim=256)

    print("\n--- Streaming generated text ---\n")
    for _ in range(500):
        try:
            print(next(stream), end=' ', flush=True)
        except StopIteration:
            break
    print("\n")

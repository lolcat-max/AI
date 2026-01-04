import os, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
import requests
import re

# ------------------------- 
# Configuration
# -------------------------
SEQ_LEN, EMBED_DIM, HIDDEN_DIM = 8, 64, 128
NUM_LAYERS, BATCH_SIZE, LR, NUM_EPOCHS = 1, 1024, 1e-3, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- 
# Sparse Dual Array (Reduced Data)
# -------------------------
class SparseDualArray:
    """90% memory reduction: shared indices + start offsets only"""
    def __init__(self, words_list, w2i, seq_len, max_samples=999999):
        self.indices = torch.tensor([w2i.get(w, 0) for w in words_list[:max_samples+seq_len]], dtype=torch.long)
        self.starts = torch.arange(min(len(words_list)-seq_len, max_samples), dtype=torch.long)
        self.seq_len = seq_len
        self.n_samples = len(self.starts)
        
        # Compute syllogism weights for each sample
        self.syllogism_weights = self._compute_syllogism_weights(words_list, seq_len)
    
    def _compute_syllogism_weights(self, words_list, seq_len):
        """
        Assign higher weights to sequences containing logical patterns:
        - Conditional statements (if/then)
        - Causal relationships (because, therefore, thus)
        - Question-answer pairs
        - Comparative logic (all, some, none, every)
        """
        weights = torch.ones(self.n_samples, dtype=torch.float32)
        
        # Logical keywords and their weight multipliers
        logical_patterns = {
            'if': 2.0,
            'then': 2.0,
            'therefore': 3.0,
            'thus': 2.5,
            'because': 2.0,
            'hence': 2.5,
            'consequently': 2.5,
            'implies': 3.0,
            'all': 1.8,
            'every': 1.8,
            'some': 1.5,
            'none': 1.8,
            'must': 2.0,
            'cannot': 1.8,
            'either': 1.5,
            'or': 1.3,
            'and': 1.2,
            'not': 1.5,
            'only': 1.5,
            'unless': 2.0,
            'when': 1.5,
            'while': 1.5,
            'since': 1.8,
        }
        
        # Question patterns boost
        question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose'}
        
        for i in range(self.n_samples):
            if i + seq_len >= len(words_list):
                break
                
            # Get the sequence window
            sequence = words_list[i:i+seq_len+1]
            sequence_str = ' '.join(sequence).lower()
            
            weight_multiplier = 1.0
            
            # Check for logical patterns
            for pattern, multiplier in logical_patterns.items():
                if pattern in sequence:
                    weight_multiplier = max(weight_multiplier, multiplier)
            
            # Boost question-answer patterns
            has_question = any(qw in sequence for qw in question_words)
            if has_question:
                weight_multiplier *= 1.5
            
            # Detect syllogistic structure (if X then Y, X, therefore Y)
            if 'if' in sequence and 'then' in sequence:
                weight_multiplier *= 1.8
            
            if ('therefore' in sequence or 'thus' in sequence) and len(sequence) > 5:
                weight_multiplier *= 1.5
            
            # Cap maximum weight to prevent extreme values
            weights[i] = min(weight_multiplier, 5.0)
        
        return weights
    
    def __getitem__(self, batch_slice):
        starts = self.starts[batch_slice]
        x = torch.stack([self.indices[s:s+self.seq_len] for s in starts])
        y = self.indices[starts+self.seq_len]
        weights = self.syllogism_weights[batch_slice]
        return x, y, weights

# ------------------------- 
# 1. Data Fetching + Sparse Conversion
# -------------------------
def fetch_hf_corpus():
    print("Fetching datasets...")
    
    # TinyShakespeare fallback chain
    try:
        with open(input("Core Filename (Enter for TinyShakespeare): ").strip() or "tinyshakespeare.txt", "r", encoding="utf-8") as f:
            ts_words = f.read().lower().strip().split()
    except:
        try:
            ts_words = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", timeout=10).text.lower().split()
        except:
            ts_words = ("hello world " * 1000).strip().split()
    ts_words = ts_words
    
    # SQuAD
    try:
        squad_ds = load_dataset("squad", split="train")
        sq_text = [f"{item['question'].lower()} {item['answers']['text'][0].lower()}" for item in squad_ds if item['answers']['text']]
        sq_words = " ".join(sq_text).split()
    except:
        sq_words = ("question answer " * 1000).split()
    
    return ts_words, sq_words

# ------------------------- 
# 2. Model Architecture
# -------------------------
class EyeStretchingLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.register_buffer("eye_basis", torch.eye(h_dim))
    
    def forward(self, x, eye_offset=1.0):
        return torch.matmul(x, self.eye_basis * float(eye_offset))

class PersonaNeuralNet(nn.Module):
    def __init__(self, v_size, e_dim, h_dim, layers):
        super().__init__()
        self.embedding = nn.Embedding(v_size, e_dim)
        self.rnn = nn.GRU(e_dim, h_dim, num_layers=layers, batch_first=True)
        self.stretcher = EyeStretchingLayer(h_dim)
        self.fc = nn.Linear(h_dim, v_size)

    def forward(self, x, eye_offset=1.0):
        if x.dim() == 1: x = x.unsqueeze(0)
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        last_hidden = out[:, -1, :]
        stretched = self.stretcher(last_hidden, eye_offset)
        return self.fc(stretched)

# ------------------------- 
# 3. Alignment Logic
# -------------------------
def apply_aligned_inhibition(x, y, v_size, sources):
    """Shift SQuAD tokens for persona manifold"""
    hf_mask = (sources == 1).long()
    x = (x + hf_mask.unsqueeze(-1)) % v_size
    return x, y

# ------------------------- 
# 4. Sparse Dataset + Training
# -------------------------
def get_mixed_batch(ts_sparse, sq_sparse, batch_size):
    """Direct batch sampling with syllogism weights"""
    n_ts = min(batch_size//2, ts_sparse.n_samples)
    n_sq = batch_size - n_ts
    
    # Sample from each source
    ts_idx = torch.randperm(ts_sparse.n_samples)[:n_ts]
    sq_idx = torch.randperm(sq_sparse.n_samples)[:n_sq]
    
    x_ts, y_ts, w_ts = ts_sparse[ts_idx]
    x_sq, y_sq, w_sq = sq_sparse[sq_idx]
    
    x = torch.cat([x_ts, x_sq])
    y = torch.cat([y_ts, y_sq])
    
    # Combine syllogism weights with source weights
    # SQuAD gets 3.5x base weight, then multiply by syllogism weight
    base_weights = torch.cat([torch.ones(n_ts), torch.full((n_sq,), 3.5)])
    syllogism_weights = torch.cat([w_ts, w_sq])
    weights = base_weights * syllogism_weights
    
    sources = torch.cat([torch.zeros(n_ts), torch.ones(n_sq)])
    
    return x, y, weights, sources

def run_training(model, ts_sparse, sq_sparse, v_size):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        pbar = tqdm(range(1000), desc=f"Epoch {epoch}")
        
        for _ in pbar:
            x, y, w, src = get_mixed_batch(ts_sparse, sq_sparse, BATCH_SIZE)
            
            x, y, w, src = x.to(device), y.to(device), w.to(device), src.to(device)
            x, y = apply_aligned_inhibition(x, y, v_size, src)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"\nEpoch {epoch} completed - Avg Loss: {total_loss/1000:.4f}")

def generate(model, seed, w2i, i2w, max_new=200):
    model.eval()
    ids = [w2i.get(w, 0) for w in seed.lower().split()]
    if not ids: ids = [0]
    
    print(f"\n>> Seed: {' '.join([i2w.get(i, '?') for i in ids[-3:]])}\n")
    
    for i in range(max_new):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=device)
        stretch = 1.3 if i % 3 == 0 else 1.0
        
        with torch.no_grad():
            logits = model(ctx, eye_offset=stretch)
            
            # Safe sampling
            logits_last = logits[0]
            logits_last = logits_last - logits_last.max()
            logits_last = logits_last / 1.0
            probs = F.softmax(logits_last, dim=-1)
            
            # Clamp invalid probs
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = torch.clamp(probs, min=0.0)
            probs = probs / probs.sum()
            
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            
            token = i2w.get(next_id, '?')
            print(token, end=" ", flush=True)
    
    print("\n")

# ------------------------- 
# Main Execution
# -------------------------
if __name__ == "__main__":
    MODEL_PATH = "persona_model.pth"
    META_PATH = "persona_meta.pkl"

    # Load existing
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        print(">> Loading checkpoint...")
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
        w2i, i2w, v_size = meta['w2i'], meta['i2w'], meta['v_size']
        
        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        # Train new
        print(">> Training new model...")
        ts_words, sq_words = fetch_hf_corpus()
        
        vocab = sorted(list(set(ts_words + sq_words)))
        w2i = {w: i for i, w in enumerate(vocab)}
        i2w = {i: w for i, w in enumerate(vocab)}
        v_size = len(vocab)
        
        # Sparse dual arrays (90% memory reduction) with syllogism weights
        ts_sparse = SparseDualArray(ts_words, w2i, SEQ_LEN)
        sq_sparse = SparseDualArray(sq_words, w2i, SEQ_LEN)
        
        # Display weight statistics
        print(f"\n>> Syllogism Weight Statistics:")
        print(f"   TinyShakespeare - Mean: {ts_sparse.syllogism_weights.mean():.3f}, Max: {ts_sparse.syllogism_weights.max():.3f}")
        print(f"   SQuAD - Mean: {sq_sparse.syllogism_weights.mean():.3f}, Max: {sq_sparse.syllogism_weights.max():.3f}")
        print(f"   High-weight samples (>2.0): TS={int((ts_sparse.syllogism_weights > 2.0).sum())}, SQ={int((sq_sparse.syllogism_weights > 2.0).sum())}\n")
        
        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        run_training(model, ts_sparse, sq_sparse, v_size)
        
        # Save
        torch.save(model.state_dict(), MODEL_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump({'w2i': w2i, 'i2w': i2w, 'v_size': v_size}, f)
        print(f">> Saved to {MODEL_PATH}")

    # Interactive generation
    while True:
        user_input = input("\nUSER: ").strip()
        if user_input.lower() in ['exit', 'quit']: break
        generate(model, user_input, w2i, i2w)
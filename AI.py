import os
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
KB_len = -1
CKPT_PATH = "neural_trained.pth"
LOG_FILE = "generation_logs.txt"  # File where results will be stored
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
NUM_LAYERS = 2
BATCH_SIZE = 1024
LR = 5e-3
NUM_EPOCHS = 1

# -------------------------
# 1. Logging Helper
# -------------------------
def log_experiment(embed_dim, hidden_dim, seed, generated_text):
    """Appends model configuration and generated output to a log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"--- Experiment: EMBED={embed_dim}, HIDDEN={hidden_dim} ---\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Output: {generated_text}\n")
        f.write("-" * 50 + "\n\n")

# -------------------------
# 2. Dataset & CKY Inverter
# -------------------------
def load_data(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read().lower()
    except:
        text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower()
    return text.split()[:KB_len]

class CKYInverter:
    def __init__(self, words, w2i):
        self.vocab_size = len(w2i)
        self.w2i = w2i
        self.matrix = torch.zeros((self.vocab_size, self.vocab_size), device=device)
        self._induce_matrix(words)

    def _induce_matrix(self, words):
        print("Inducing CKY Transition Matrix...")
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 in self.w2i and w2 in self.w2i:
                self.matrix[self.w2i[w1], self.w2i[w2]] = 1.0

    def get_mask(self, last_word_id):
        mask = torch.full((self.vocab_size,), -float('inf'), device=device)
        valid_indices = torch.where(self.matrix[last_word_id] > 0)[0]
        if len(valid_indices) > 0:
            mask[valid_indices] = 0.0
        else:
            mask.fill_(0.0)
        return mask

# -------------------------
# 3. Model & Dataset
# -------------------------
class PlainNeuralNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        return self.fc(out[:, -1, :]), h

class TextDataset(Dataset):
    def __init__(self, words, w2i, seq_len):
        self.samples = []
        for i in range(len(words) - seq_len):
            x = [w2i[w] for w in words[i:i + seq_len]]
            y = w2i[words[i + seq_len]]
            self.samples.append((torch.tensor(x), torch.tensor(y)))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 4. Updated Generation
# -------------------------
def generate_text_inverted(model, inverter, seed, w2i, i2w, seq_len, max_len=1000, choose_words=None):
    model.eval()
    gen_ids = [w2i.get(w, 0) for w in seed.split()]
    output_words = []
    
    for _ in range(max_len):
        inp = torch.tensor([gen_ids[-seq_len:]], device=device)
        with torch.no_grad():
            logits, _ = model(inp)
            mask = inverter.get_mask(gen_ids[-1])
            constrained_logits = logits[0] + mask
            probs = F.softmax(constrained_logits, dim=-1)
            
            if choose_words is not None:
                indices = [w2i.get(w, 0) for w in choose_words]
                probs_segment = probs[indices]
                probs_segment = probs_segment / (probs_segment.sum() + 1e-9)
                next_id = np.random.choice(indices, p=probs_segment.cpu().numpy())
            else:
                next_id = torch.multinomial(probs, 1).item()
            
            gen_ids.append(next_id)
            output_words.append(i2w[next_id])
            
    return " ".join(output_words)

# -------------------------
# 5. Main Execution
# -------------------------
if __name__ == "__main__":
    fname = input("Filename: ")
    words = load_data(fname)
    vocab = sorted(list(set(words)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}

    inverter = CKYInverter(words, w2i)
    dataset = TextDataset(words, w2i, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    seed = "the king is" # Default seed for logging consistency

    for n in range(1, 99):
        for m in range(1, 99):
            EMBED_DIM, HIDDEN_DIM = n, m
            model = PlainNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            # Training
            for epoch in range(1, NUM_EPOCHS + 1):
                pbar = tqdm(loader, desc=f"Dim {n}/{m} Ep {epoch}")
                for i, (x, y) in enumerate(pbar):
                    x, y = x.to(device), y.to(device)
                    y = torch.clamp(y, 0, len(vocab) - 1)
                    
                    # Apply your perturbation logic
                    batch_size, seq_len = x.shape
                    idx_batch = i % batch_size
                    
                    if batch_size > 0 and seq_len > 0:
                        # Perturbation 1
                        idx_seq1 = (i % EMBED_DIM + 1) % seq_len
                        x[idx_batch, idx_seq1] = (y[idx_batch] + x[idx_batch, idx_seq1]) % len(vocab)
                        # Perturbation 2
                        idx_seq2 = (i % HIDDEN_DIM + 1) % seq_len
                        x[idx_batch, idx_seq2] = (y[idx_batch] - x[idx_batch, idx_seq2]) % len(vocab)
                    
                    optimizer.zero_grad()
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=f"{loss.item():.3f}")

            # Generate and Log
            generated = generate_text_inverted(model, inverter, seed, w2i, i2w, SEQ_LEN)
            print(f"\n[EMBED {n} | HIDDEN {m}] Generated: {generated[:1000]}...")
            log_experiment(n, m, seed, generated)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 1024
LR = 5e-3
NUM_EPOCHS = 1

# -------------------------
# 1. Dataset
# -------------------------
def load_data(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read().lower()
    except:
        text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower()
    return text.split()[:KB_len]

# -------------------------
# 2. Model & Dataset
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
# 3. Generation & Inference
# -------------------------
def generate_text(model, seed, w2i, i2w, seq_len, max_len=500, choose_words=None):
    model.eval()
    gen_ids = [w2i.get(w, 0) for w in seed.split()]
    
    print(f"\n>> Seed: {seed}")
    for _ in range(max_len):
        inp = torch.tensor([gen_ids[-seq_len:]], device=device)
        with torch.no_grad():
            logits, _ = model(inp)
            
            
            probs = F.softmax(logits, dim=-1)
            
            # If choose_words is specified, segment probs to only those words
            if choose_words is not None:
                # Get indices for the two words
                indices = [w2i.get(w, 0) for w in choose_words]
                probs_segment = probs[indices]
                probs_segment = probs_segment / probs_segment.sum()  # Normalize
                # Sample from only these two
                next_id = np.random.choice(indices, p=probs_segment.cpu().numpy())
            else:
                # Default: sample from all words
                next_id = torch.multinomial(probs, 1).item()
            
            gen_ids.append(next_id)
            print(i2w[next_id], end=' ', flush=True)
    print("\n")


# -------------------------
# 4. Main Execution
# -------------------------
if __name__ == "__main__":
    words = load_data(input("Filename: "))
    vocab = sorted(list(set(words)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    
    dataset = TextDataset(words, w2i, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PlainNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        i = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            # Clamp target indices to valid range
            y = torch.clamp(y, 0, len(vocab) - 1)
            
            # Ensure indices are within tensor dimensions
            batch_size, seq_len = x.shape
            idx_batch = i % batch_size
            idx_seq = (i % EMBED_DIM + 1) % seq_len
            
            # Modify x[idx_batch, idx_seq] using y[idx_batch]
            if batch_size > 0 and seq_len > 0:
                x[idx_batch, idx_seq] = (y[idx_batch] + x[idx_batch, idx_seq]) % len(vocab)
                
            idx_seq = (i % HIDDEN_DIM + 1) % seq_len
            
            # Modify x[idx_batch, idx_seq] using y[idx_batch]
            if batch_size > 0 and seq_len > 0:
                x[idx_batch, idx_seq] = (y[idx_batch] + x[idx_batch, idx_seq]) % len(vocab)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            i += 1
        torch.save(model.state_dict(), CKPT_PATH)

    while True:
        seed = input("\nSeed >> ").strip().lower()
        if not seed: break
        generate_text(model, seed, w2i, i2w, SEQ_LEN)

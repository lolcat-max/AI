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
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 512
LR = 5e-3
NUM_EPOCHS = 50 #Trained lingual styles

# -------------------------
# 1. Dataset & CKY Inverter
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
            x = [w2i.get(w, 0) for w in words[i:i + seq_len]]
            y = w2i.get(words[i + seq_len], 0)
            self.samples.append((torch.tensor(x), torch.tensor(y)))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 3. Generation with Flexible Sampling
# -------------------------
def generate_text_restructured(model, inverter, seed, w2i, i2w, seq_len, max_len=500, choose_words=None, sampling_strategy=None):
    model.eval()
    gen_ids = [w2i.get(w, 0) for w in seed.split()]
    print(f"\n>> Seed: {seed}")

    if sampling_strategy is None:
        def sampling_strategy(logits, choose_words):
            # Clamp logits to avoid overflow
            logits = torch.clamp(logits, min=-1e10, max=1e10)
            probs = F.softmax(logits, dim=-1)
            
            # Check for nan or inf
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.size(0)  # Uniform fallback

            if choose_words is not None:
                indices = [w2i.get(w, 0) for w in choose_words]
                probs_segment = probs[indices]
                if probs_segment.sum() == 0:
                    probs_segment = torch.ones_like(probs_segment) / len(probs_segment)
                probs_segment /= probs_segment.sum()
                return np.random.choice(indices, p=probs_segment.cpu().numpy())
            return torch.multinomial(probs, 1).item()


    for _ in range(max_len):
        last_ids = gen_ids[-seq_len:]
        inp = torch.tensor([last_ids], device=device)
        with torch.no_grad():
            logits, _ = model(inp)
            logits = logits[0]
            for last_id in last_ids:
                mask = inverter.get_mask(last_id)
                logits += mask
            next_id = sampling_strategy(logits, choose_words)
            gen_ids.append(next_id)
            print(i2w[next_id], end=' ', flush=True)
    print("\n")

# -------------------------
# 4. Non-Smooth Training with Semantic Warps
# -------------------------
def get_epoch_data(words, epoch, total_epochs):
    chunk_size = len(words) // total_epochs
    start = (epoch % total_epochs) * chunk_size
    end = start + chunk_size
    return words[start:end]

def get_loss_function(epoch):
    if epoch % 2 == 0:
        return nn.CrossEntropyLoss()
    else:
        # Example: Custom semantic loss (placeholder)
        return nn.CrossEntropyLoss()

def get_optimizer(model, epoch):
    if epoch % 2 == 0:
        return optim.Adam(model.parameters(), lr=5e-3)
    else:
        return optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# -------------------------
# 5. Main Execution
# -------------------------
if __name__ == "__main__":
    words = load_data(input("Filename: "))
    vocab = sorted(list(set(words)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}

    inverter = CKYInverter(words, w2i)
    model = PlainNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)

    # Training loop with warps
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_words = get_epoch_data(words, epoch, NUM_EPOCHS)
        dataset = TextDataset(epoch_words, w2i, SEQ_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        criterion = get_loss_function(epoch)
        optimizer = get_optimizer(model, epoch)

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        torch.save(model.state_dict(), CKPT_PATH)

    # Interactive generation
    while True:
        seed = input("\nSeed >> ").strip().lower()
        if not seed: break
        generate_text_restructured(model, inverter, seed, w2i, i2w, SEQ_LEN)

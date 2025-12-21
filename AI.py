import os
import requests
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
KB_len = 9999
CKPT_PATH = "plain_neural_trainer.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
SEQ_LEN = 3
EMBED_DIM = 640
HIDDEN_DIM = 1280
NUM_LAYERS = 2
BATCH_SIZE = 512
LR = 5e-3
NUM_EPOCHS = 15

# -------------------------
# 1. Dataset Preparation
# -------------------------
def load_data(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read().lower()
    except:
        try:
            text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower()
        except:
            text = "hello world " * 1000
    return text.split()[:KB_len]

def create_vocab(words):
    vocab_set = set(words)
    vocab = sorted(list(vocab_set))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w

def create_samples(words, w2i, seq_len):
    samples = []
    for i in range(len(words) - seq_len):
        x = [w2i[w] for w in words[i:i + seq_len]]
        y = w2i[words[i + seq_len]]
        samples.append((x, y))
    return samples

# -------------------------
# 2. Plain Neural Network
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
        logits = self.fc(out[:, -1, :])
        return logits, h

# -------------------------
# 3. Training Loop
# -------------------------
def train(model, optimizer, loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        
        print(f"Epoch {epoch} Done. Loss: {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), CKPT_PATH)

# -------------------------
# 4. Dataset and Generator
# -------------------------
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_list, y_val = self.data[idx]
        return torch.tensor(x_list, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)

def generate_text(model, seed, w2i, i2w, seq_len, max_len=500):
    model.eval()
    gen_ids = [w2i.get(w, 0) for w in seed.split()]
    
    for _ in range(max_len):
        inp = gen_ids[-seq_len:]
        if len(inp) < seq_len:
            inp = [0] * (seq_len - len(inp)) + inp
        xt = torch.tensor([inp], device=device)
        
        with torch.no_grad():
            logits, _ = model(xt)
            probs = F.softmax(logits[0], dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            gen_ids.append(next_id)
            print(i2w[next_id], end=' ', flush=True)
    print("\n")
def reconstruct_probs(model, words, w2i, i2w, seq_len, radius=5, boost=0.2):
    model.eval()
    probs_list = []
    
    for i in range(len(words) - seq_len):
        inp = [w2i[w] for w in words[i:i + seq_len]]
        if len(inp) < seq_len:
            inp = [0] * (seq_len - len(inp)) + inp
        xt = torch.tensor([inp], device=device)
        
        with torch.no_grad():
            logits, _ = model(xt)
            probs = F.softmax(logits[0], dim=-1)
            
            # 1. Identify the 'spatial' center (the prediction)
            pred_id = torch.argmax(probs)
            
            # 2. Define the spatial condition: indices within 'radius' of pred_id
            indices = torch.arange(len(probs), device=device)
            condition = torch.abs(indices - pred_id) <= radius
            
            # 3. Apply displacement using torch.where
            # If in neighborhood, boost probability; otherwise, keep as is
            displaced_probs = torch.where(condition, probs + boost, probs)
            
            # 4. Re-normalize to ensure they remain valid probabilities
            displaced_probs = displaced_probs / displaced_probs.sum()
            
            probs_list.append(displaced_probs.cpu().numpy())
    
    return probs_list



# -------------------------
# 5. Main
# -------------------------
if __name__ == "__main__":
    filename = input("Filename: ")
    words = load_data(filename)
    vocab, w2i, i2w = create_vocab(words)
    samples = create_samples(words, w2i, SEQ_LEN)
    
    dataset = TextDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PlainNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    if os.path.exists(CKPT_PATH):
        print("Loading checkpoint...")
        try:
            model.load_state_dict(torch.load(CKPT_PATH))
        except:
            print("Checkpoint mismatch or error, starting fresh.")
    
    print("Starting Plain Neural Trainer...")
    try:
        train(model, optimizer, loader, NUM_EPOCHS)
    except KeyboardInterrupt:
        print("Saving...")
        torch.save(model.state_dict(), CKPT_PATH)
    
    print("\nInteractive Generator:")
    while True:
        seed = input(">> ")
        if not seed:
            break
        generate_text(model, seed, w2i, i2w, SEQ_LEN)
    
    print("\nReconstructing Dataset Probabilities...")
    probs_list = reconstruct_probs(model, words, w2i, i2w, SEQ_LEN)
    print("Reconstructed probabilities:", probs_list)
import os
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter, defaultdict

# -------------------------
# Config
# -------------------------
KB_len = -1
GEN_LEN = 600
CKPT_PATH = "trained_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 1024
LR = 5e-3
NUM_EPOCHS = 1

# -------------------------
# 1. Biophysical Logic Gates
# -------------------------
class InhibitoryRenetworker(nn.Module):
    """Resolves ambiguity by inhibiting neurons in the 'Interference Zone'."""
    def __init__(self, gap=0.05):
        super().__init__()
        self.gap = gap

    def forward(self, activations):
        lead_energy = torch.max(activations, dim=-1, keepdim=True)[0]
        interference = lead_energy - activations
        # Identify neurons within the inhibitory gap
        suppression_mask = (interference > 0) & (interference < self.gap)
        renetworked = activations.clone()
        renetworked[suppression_mask] -= 150.0  # Synaptic inhibition
        return renetworked

class CKYInverter:
    """External Grounding: Ensures transitions exist in source text."""
    def __init__(self, words, w2i):
        self.vocab_size = len(w2i)
        self.w2i = w2i
        self.matrix = torch.zeros((self.vocab_size, self.vocab_size), device=device)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 in self.w2i and w2 in self.w2i:
                self.matrix[self.w2i[w1], self.w2i[w2]] = 1.0

    def get_grounding(self, last_word_id):
        mask = torch.full((self.vocab_size,), -float('inf'), device=device)
        valid = torch.where(self.matrix[last_word_id] > 0)[0]
        if len(valid) > 0: mask[valid] = 0.0
        else: mask.fill_(0.0)
        return mask

# -------------------------
# 2. Model with Certainty Head
# -------------------------
class DislocatedNeuralNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_vocab = nn.Linear(hidden_dim, vocab_size)
        self.fc_conf = nn.Linear(hidden_dim, 1) # Additional Certainty Neuron

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        last_hidden = out[:, -1, :]
        logits = self.fc_vocab(last_hidden)
        certainty = torch.sigmoid(self.fc_conf(last_hidden))
        return logits, certainty, h

# -------------------------
# 3. Training & Generation
# -------------------------
def load_data(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read().lower().split()
    except:
        text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower().split()
    return text[:KB_len]

class TextDataset(Dataset):
    def __init__(self, words, w2i, seq_len):
        self.samples = [(torch.tensor([w2i[w] for w in words[i:i+seq_len]]), w2i[words[i+seq_len]]) 
                        for i in range(len(words)-seq_len)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def generate_text(model, inverter, renetworker, seed, w2i, i2w, seq_len):
    model.eval()
    gen_ids = [w2i.get(w, 0) for w in seed.split()]
    print(f"\n>> Seed: {seed}")
    for _ in range(GEN_LEN):
        inp = torch.tensor([gen_ids[-seq_len:]], device=device)
        with torch.no_grad():
            logits, conf, _ = model(inp)
            # Stage I: Internal Disambiguation
            clean_activations = renetworker(logits[0])
            # Stage II: External Grounding
            grounding = inverter.get_grounding(gen_ids[-1])
            probs = F.softmax(clean_activations + grounding, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            gen_ids.append(next_id)
            print(f"{i2w[next_id]}", end=' ', flush=True)

if __name__ == "__main__":
    words = load_data(input("Filename: "))
    vocab = sorted(list(set(words)))

    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}          # correct

    inverter = CKYInverter(words, w2i)
    renetworker = InhibitoryRenetworker(gap=0.88)
    loader = DataLoader(TextDataset(words, w2i, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    
    model = DislocatedNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            # kernel dislocation: shift input indices
            if i % 10 == 0:
                with torch.no_grad():
                    model.embedding.weight.add_(torch.randn_like(model.embedding.weight) * 0.001)

            optimizer.zero_grad()
            logits, certainty, _ = model(x)
            v_loss = criterion(logits, y)
            with torch.no_grad():
                target_conf = torch.exp(-v_loss.detach()).unsqueeze(0).expand(certainty.size())
            c_loss = F.mse_loss(certainty, target_conf)
            (v_loss + c_loss).backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{v_loss.item():.3f}")

    torch.save(model.state_dict(), CKPT_PATH)
    while True:
        seed = input("\nSeed >> ").strip().lower()
        if not seed: break
        generate_text(model, inverter, renetworker, seed, w2i, i2w, SEQ_LEN)

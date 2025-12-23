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
KB_len = 9999
CKPT_PATH = "cky_neural_trainer.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
EMBED_DIM = 640
HIDDEN_DIM = 1280
NUM_LAYERS = 2
BATCH_SIZE = 512
LR = 5e-3
NUM_EPOCHS = 10

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
    """Inverts CKY logic: Induces a transition matrix to constrain neural output."""
    def __init__(self, words, w2i):
        self.vocab_size = len(w2i)
        self.w2i = w2i
        # The 'CKY Matrix' representation: rows=current_word, cols=next_word
        self.matrix = torch.zeros((self.vocab_size, self.vocab_size), device=device)
        self._induce_matrix(words)

    def _induce_matrix(self, words):
        print("Inducing CKY Transition Matrix...")
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 in self.w2i and w2 in self.w2i:
                self.matrix[self.w2i[w1], self.w2i[w2]] = 1.0

    def get_mask(self, last_word_id):
        """Returns a logit mask (-inf for illegal transitions)."""
        mask = torch.full((self.vocab_size,), -float('inf'), device=device)
        valid_indices = torch.where(self.matrix[last_word_id] > 0)[0]
        if len(valid_indices) > 0:
            mask[valid_indices] = 0.0
        else:
            mask.fill_(0.0) # Fallback if word was a leaf/terminal
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
            x = [w2i[w] for w in words[i:i + seq_len]]
            y = w2i[words[i + seq_len]]
            self.samples.append((torch.tensor(x), torch.tensor(y)))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 3. Generation & Inference
# -------------------------
def generate_text_inverted(model, inverter, seed, w2i, i2w, seq_len, max_len=500, choose_words=None):
    model.eval()
    gen_ids = [w2i.get(w, 0) for w in seed.split()]
    
    print(f"\n>> Seed: {seed}")
    for _ in range(max_len):
        inp = torch.tensor([gen_ids[-seq_len:]], device=device)
        with torch.no_grad():
            logits, _ = model(inp)
            
            # Apply CKY Matrix Mask
            mask = inverter.get_mask(gen_ids[-1])
            constrained_logits = logits[0] + mask
            
            probs = F.softmax(constrained_logits, dim=-1)
            
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

    # Initialize CKY Inverter (The Grammar Matrix)
    inverter = CKYInverter(words, w2i)
    
    dataset = TextDataset(words, w2i, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PlainNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, NUM_EPOCHS + 1):
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

    # Interactive Inverted CKY Generation
    while True:
        seed = input("\nSeed >> ").strip().lower()
        if not seed: break
        generate_text_inverted(model, inverter, seed, w2i, i2w, SEQ_LEN)

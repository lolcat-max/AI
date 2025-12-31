import os, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset

# -------------------------
# Configuration
# -------------------------
SEQ_LEN, EMBED_DIM, HIDDEN_DIM = 3, 64, 128
NUM_LAYERS, BATCH_SIZE, LR, NUM_EPOCHS = 2, 512, 1e-3, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. HF Data Fetching
# -------------------------
def fetch_hf_corpus():
    """Fetches narrative and query data from Hugging Face [web:209][web:225]."""
    print("Fetching HF Datasets...")
    # 1. TinyStories (Narrative) - 5000 stories [web:211]
    ts_ds = load_dataset('roneneldan/TinyStories', split='train[:5000]', trust_remote_code=True)
    ts_words = " ".join(ts_ds['text']).lower().split()
    
    # 2. SQuAD (Query/Persona) - 3000 items [web:213]
    squad_ds = load_dataset('squad', split='train[:3000]', trust_remote_code=True)
    sq_text = []
    for item in squad_ds:
        # Extract question and first answer for persona grounding [web:225]
        ans = item['answers']['text'][0].lower() if item['answers']['text'] else ""
        sq_text.append(f"{item['question'].lower()} {ans}")
    sq_words = " ".join(sq_text).split()
    
    return ts_words, sq_words

# -------------------------
# 2. Model Architecture
# -------------------------
class EyeStretchingLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.register_buffer('eye_basis', torch.eye(h_dim))
    def forward(self, x, eye_offset=1.0):
        return torch.matmul(x, self.eye_basis * eye_offset)

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
        # Always slice batch [:, -1, :] to prevent collapse [web:56][web:146]
        last_hidden = out[:, -1, :] 
        stretched = self.stretcher(last_hidden, eye_offset)
        return self.fc(stretched)

# -------------------------
# 3. Alignment Logic
# -------------------------
def apply_aligned_inhibition(x, y, v_size, sources):
    """Bidirectional manifold alignment [web:111]."""
    hf_mask = (sources == 1).long()
    # Shift indices for SQuAD source to bind persona-specific gradients
    x = (x + hf_mask.unsqueeze(-1)) % v_size
    return x, y

class AlignedDataset(Dataset):
    def __init__(self, ts_words, sq_words, w2i, seq_len):
        self.samples = []
        # TinyStories (Source 0)
        for i in range(len(ts_words) - seq_len):
            self.samples.append((
                torch.tensor([w2i[w] for w in ts_words[i:i+seq_len]]),
                torch.tensor(w2i[ts_words[i+seq_len]]),
                torch.tensor(1.0), # Weight
                torch.tensor(i)    # Source
            ))
        # SQuAD (Source 1) - Higher weight for query focus [web:25]
        for i in range(len(sq_words) - seq_len):
            self.samples.append((
                torch.tensor([w2i[w] for w in sq_words[i:i+seq_len]]),
                torch.tensor(w2i[sq_words[i+seq_len]]),
                torch.tensor(3.5), # Weight boost for persona scenarios
                torch.tensor(i+1)    # Source
            ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 4. Training & Generation
# -------------------------
def run_training(model, loader, v_size):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y, w, src in pbar:
            x, y, w, src = x.to(device), y.to(device), w.to(device), src.to(device)
            x, y = apply_aligned_inhibition(x, y, v_size, src)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

def generate(model, seed, w2i, i2w):
    model.eval()
    ids = [w2i.get(w, 0) for w in seed.lower().split()]
    print(f"\n>> Seed: {seed}")
    for i in range(600):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=device)
        stretch = 1.3 if i % ids[-SEQ_LEN:][min(i,SEQ_LEN-1)] >= 0.5 else i
        with torch.no_grad():
            logits = model(ctx, eye_offset=stretch)
            probs = F.softmax(logits, dim=-1).squeeze()
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            print(f"{i2w.get(next_id, '?')} ", end="", flush=True)
    print("\n")

# -------------------------
# Main
# -------------------------
# -------------------------
# Main Execution (Updated with Save/Load)
# -------------------------
if __name__ == "__main__":
    MODEL_PATH = "persona_model.pth"
    META_PATH = "persona_meta.pkl"

    # Attempt to load existing model and metadata [web:233][web:236]
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        print(">> Loading existing checkpoint...")
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
        w2i, i2w, v_size = meta['w2i'], meta['i2w'], meta['v_size']
        
        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        # If no checkpoint, perform training as usual
        print(">> No checkpoint found. Initializing training...")
        ts_words, sq_words = fetch_hf_corpus()
        vocab = sorted(list(set(ts_words + sq_words)))
        w2i, i2w, v_size = {w: i for i, w in enumerate(vocab)}, {i: w for i, w in enumerate(vocab)}, len(vocab)
        
        loader = DataLoader(AlignedDataset(ts_words, sq_words, w2i, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        
        # Standard training loop [web:99][web:171]
        run_training(model, loader, v_size)
        
        # Save weights and vocabulary for future use [web:228][web:241]
        torch.save(model.state_dict(), MODEL_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump({'w2i': w2i, 'i2w': i2w, 'v_size': v_size}, f)
        print(f">> Model saved to {MODEL_PATH}")

    # Final Generation
    while True:
        generate(model, input("USER: "), w2i, i2w)

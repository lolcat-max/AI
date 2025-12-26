import os, requests, re, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# -------------------------
# Config
# -------------------------
KB_len = -1
GEN_LEN = 500
CKPT_PATH = "persona_binding_model.pth"
VOCAB_PATH = "vocab_data.pkl"
INVERTER_PATH = "inverter_matrix.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 1024
LR = 5e-3
NUM_EPOCHS = 1
MODE = 'auto' # 'auto' loads if checkpoint exists

# -------------------------
# 1. Data Fetching (TinyStories + SQuAD)
# -------------------------
def fetch_hf_data():
    """Downloads TinyStories for narrative and SQuAD for Q&A logic."""
    try:
        from datasets import load_dataset
        print("  Downloading datasets from HuggingFace...")
        
        # 1.1 TinyStories (Narrative Stream)
        ts_ds = load_dataset('roneneldan/TinyStories', split='train[:5000]', trust_remote_code=True)
        ts_words = " ".join(ts_ds['text']).lower().split()
        
        # 1.2 SQuAD (Q&A Stream)
        squad_ds = load_dataset('squad', split='train[:3000]', trust_remote_code=True)
        questions, answers, contexts = [], [], []
        for item in squad_ds:
            questions.append(item['question'].lower())
            answers.append(item['answers']['text'][0].lower() if item['answers']['text'] else "")
            contexts.append(item['context'].lower())
            
        qa_words = " ".join(questions + answers + contexts).split()
        q_types = {q.split()[0] for q in questions if q.split()}
        
        print(f"  ✓ TinyStories: {len(ts_words)} words | SQuAD: {len(qa_words)} words")
        return list(q_types), ts_words, qa_words
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "datasets", "--break-system-packages", "-q"], check=True)
        return fetch_hf_data()

def extract_methods(words):
    action_suffixes = r".+(ing|ed|ion|al|ment|ance|ive|ize)$"
    candidates = [w for w in words if re.match(action_suffixes, w) and len(w) > 4]
    m_set = {w for w, c in Counter(candidates).items() if c >= 2}
    m_set.update({"act", "translate", "summarize", "generate", "write", "solve"})
    return m_set

# -------------------------
# 2. Logic Gates & Inverter
# -------------------------
class InhibitoryRenetworker(nn.Module):
    def __init__(self, gap=0.06):
        super().__init__(); self.gap = gap
    def forward(self, activations):
        lead = torch.max(activations, dim=-1, keepdim=True)[0]
        mask = (lead - activations > 0) & (lead - activations < self.gap)
        out = activations.clone()
        out[mask] -= 100.0  
        return out

class CKYInverter:
    def __init__(self, matrix=None):
        self.matrix = matrix
    @classmethod
    def from_words(cls, words, w2i, v_size):
        mat = torch.zeros((v_size, v_size), device=device)
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if w1 in w2i and w2 in w2i: mat[w2i[w1], w2i[w2]] = 1.0
        return cls(mat)
    def get_grounding(self, last_id, v_size):
        mask = torch.full((v_size,), -float('inf'), device=device)
        valid = torch.where(self.matrix[last_id] > 0)[0]
        if len(valid) > 0: mask[valid] = 0.0
        else: mask.fill_(0.0)
        return mask

# -------------------------
# 3. Model & Dataset
# -------------------------
class PersonaNeuralNet(nn.Module):
    def __init__(self, v_size, e_dim, h_dim, layers):
        super().__init__()
        self.embedding = nn.Embedding(v_size, e_dim)
        self.rnn = nn.GRU(e_dim, h_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(h_dim, v_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x); return self.fc(out[:, -1, :])

class PersonaBindingDataset(Dataset):
    def __init__(self, words, w2i, methods, scenarios, seq_len, sources):
        self.samples = []
        for i in range(len(words) - seq_len):
            ctx, target = words[i:i+seq_len], words[i+seq_len]
            is_bound = any(t in methods for t in ctx+[target]) and any(t in scenarios for t in ctx+[target])
            weight = 3.5 if is_bound else 1.0
            self.samples.append((torch.tensor([w2i[w] for w in ctx]), torch.tensor(w2i[target]), torch.tensor(weight), torch.tensor(sources[i+seq_len])))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 4. Aligned Axis Inhibition
# -------------------------
def apply_aligned_inhibition(x, y, v_size, sources):
    batch, seq = x.shape
    hf_mask, file_mask = (sources == 1), (sources == 0)
    # SQuAD (Source 1): Sequence Axis (Horizontal)
    for i in range(seq):
        if i < batch and hf_mask[i % batch]:
            x[file_mask[i] % batch, i % seq] = (hf_mask[i % batch] + x[i % batch, i % seq]) % v_size
    # TinyStories (Source 0): Batch Axis (Vertical)
    for i in range(batch):
        if file_mask[i]:
            y[i] = (y[i] - x[i, i % seq]) % v_size
    return x, y

# -------------------------
# 5. Execution Engine
# -------------------------
def save_all(model, w2i, i2w, inverter, scenarios, methods):
    torch.save(model.state_dict(), CKPT_PATH)
    torch.save(inverter.matrix, INVERTER_PATH)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'w2i':w2i,'i2w':i2w,'scenarios':scenarios,'methods':methods,'v_size':len(i2w)}, f)

def generate(model, inverter, renetworker, seed, w2i, i2w):
    model.eval(); v_size = len(i2w)
    ids = [w2i.get(w, 0) for w in seed.lower().split()]
    print(f"\n>> {seed}")
    for _ in range(GEN_LEN):
        inp = torch.tensor([ids[-SEQ_LEN:]], device=device)
        with torch.no_grad():
            logits = model(inp)
            clean = renetworker(logits[0])
            grounding = inverter.get_grounding(ids[-1], v_size)
            probs = F.softmax(clean + grounding, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id); print(i2w.get(next_id, "?"), end=' ', flush=True)

if __name__ == "__main__":
    exists = all(os.path.exists(p) for p in [CKPT_PATH, VOCAB_PATH, INVERTER_PATH])
    if MODE == 'load' or (MODE == 'auto' and exists):
        with open(VOCAB_PATH, 'rb') as f: d = pickle.load(f)
        w2i, i2w, scenarios, methods, v_size = d['w2i'], d['i2w'], d['scenarios'], d['methods'], d['v_size']
        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        inverter = CKYInverter(torch.load(INVERTER_PATH, map_location=device))
        renetworker = InhibitoryRenetworker()
    else:
        q_types, ts_words, qa_words = fetch_hf_data()
        sources = ([0] * len(ts_words)) + ([1] * len(qa_words))
        all_words = (ts_words + qa_words)[:KB_len]
        vocab = sorted(list(set(all_words)))
        w2i, i2w = {w: i for i, w in enumerate(vocab)}, {i: w for i, w in enumerate(vocab)}
        
        methods, scenarios = extract_methods(all_words), set(q_types)
        inverter = CKYInverter.from_words(all_words, w2i, len(vocab))
        renetworker = InhibitoryRenetworker()
        
        loader = DataLoader(PersonaBindingDataset(all_words, w2i, methods, scenarios, SEQ_LEN, sources), batch_size=BATCH_SIZE, shuffle=True)
        model = PersonaNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(1, NUM_EPOCHS + 1):
            pbar = tqdm(loader, desc=f"Epoch {epoch}")
            for x, y, w, src in pbar:
                x, y, w, src = x.to(device), y.to(device), w.to(device), src.to(device)
                x, y = apply_aligned_inhibition(x, y, len(vocab), src)
                optimizer.zero_grad()
                loss = (F.cross_entropy(model(x), y, reduction='none') * w).mean()
                loss.backward(); optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        save_all(model, w2i, i2w, inverter, scenarios, methods)

    while True:
        seed = input("\nSeed >> ").strip()
        if not seed: break
        generate(model, inverter, renetworker, seed, w2i, i2w)

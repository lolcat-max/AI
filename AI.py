# persona_false_memory_interpretation.py
# Your full script, rewritten so the "false memory birth" dynamics are explicit and inspectable,
# WITHOUT top-k / candidate sampling tricks.
#
# What changes vs your original:
# 1) FIXES the source label bug (src becomes 0/1 so apply_aligned_inhibition actually does something).
# 2) Adds instrumentation to SHOW where "false memories" are born during generation:
#    - measures model confidence (max prob, entropy)
#    - measures "self-surprise" after committing a sampled token (next-step NLL under the model)
#    - flags "false-memory events" = confident commitment followed by high surprise
# 3) Makes the stretch logic stable (your old stretch expression can explode and dominate behavior).
#
# Importantly: generation is still plain multinomial sampling from softmax.
# No top-k, no beam search, no lookahead heuristics. Just visibility.

import os, pickle, math
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

# Generation settings (no top-k tricks)
GEN_TOKENS = 600
TEMP = 0.7  # keep 1.0 for baseline

# False-memory instrumentation thresholds (tune to taste)
CONFIDENT_P = 0.15          # "model acted confident"
HIGH_SURPRISE_NLL = 113.0     # "then the model was surprised next step"
LOW_ENTROPY = 114.0           # "distribution was sharp-ish" (depends on vocab size)


# -------------------------
# 1. HF Data Fetching
# -------------------------
def fetch_hf_corpus():
    print("Fetching HF Datasets...")
    # 1) Narrative
    try:
        with open(input("Core Filename: "), "r", encoding="utf-8") as f:
            ts_words = f.read().lower().split()
    except Exception:
        try:
            import requests
            ts_words = requests.get(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                timeout=20
            ).text.lower().split()
        except Exception:
            ts_words = ("hello world " * 1000).split()

    # 2) External / Q-A stream
    squad_ds = load_dataset('squad', split='train', trust_remote_code=True)
    sq_text = []
    for item in squad_ds:
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
        last_hidden = out[:, -1, :]
        stretched = self.stretcher(last_hidden, eye_offset)
        return self.fc(stretched)


# -------------------------
# 3. Alignment Logic (FIXED)
# -------------------------
def apply_aligned_inhibition(x, y, v_size, sources):
    """
    Optional structured perturbation between corpora.
    With src properly {0,1}, this actually applies to SQuAD samples.
    """
    hf_mask = (sources == 1).long()          # [B]
    x = (x + hf_mask.unsqueeze(-1)) % v_size # [B,T]
    return x, y

class AlignedDataset(Dataset):
    def __init__(self, ts_words, sq_words, w2i, seq_len):
        self.samples = []

        # Source 0: narrative
        for i in range(len(ts_words) - seq_len):
            self.samples.append((
                torch.tensor([w2i[w] for w in ts_words[i:i+seq_len]], dtype=torch.long),
                torch.tensor(w2i[ts_words[i+seq_len]], dtype=torch.long),
                torch.tensor(1.0),
                torch.tensor(0, dtype=torch.long)   # FIX: label, not index
            ))

        # Source 1: query / persona
        for i in range(len(sq_words) - seq_len):
            self.samples.append((
                torch.tensor([w2i[w] for w in sq_words[i:i+seq_len]], dtype=torch.long),
                torch.tensor(w2i[sq_words[i+seq_len]], dtype=torch.long),
                torch.tensor(3.5),
                torch.tensor(1, dtype=torch.long)   # FIX: label, not index
            ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# -------------------------
# 4. Training
# -------------------------
def run_training(model, loader, v_size):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y, w, src in pbar:
            x, y, w, src = x.to(device), y.to(device), w.to(device), src.to(device)
            x, y = apply_aligned_inhibition(x, y, v_size, src)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")


# -------------------------
# 5. False-memory instrumentation
# -------------------------
def entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())

@torch.no_grad()
def next_step_surprise(model, ids, eye_offset=1.0):
    """
    After committing a token, measure how surprised the model is about its own next step.
    We approximate this by NLL of the next-step argmax under the model distribution.

    This isn't "truth"; it's "self-consistency pressure".
    A 'false memory event' can look like:
      - step t was made with high confidence (sharp distribution)
      - after committing it, the model's next-step distribution becomes messy or low-confidence
    """
    ctx = torch.tensor([ids[-SEQ_LEN:]], device=device)
    logits = model(ctx, eye_offset=eye_offset).squeeze(0) / max(TEMP, 1e-6)
    probs = F.softmax(logits, dim=-1)
    y_hat = int(torch.argmax(probs).item())
    nll = float(-torch.log(probs[y_hat].clamp_min(1e-12)).item())
    return nll, float(probs.max().item()), entropy(probs), y_hat


# -------------------------
# 6. Generation (plain multinomial, no topk)
# -------------------------
@torch.no_grad()
def generate(model, seed, w2i, i2w):
    model.eval()
    ids = [w2i.get(w, 0) for w in seed.lower().split()]
    if len(ids) < SEQ_LEN:
        ids = [0] * (SEQ_LEN - len(ids)) + ids

    print(f"\n>> Seed: {seed}\n")

    false_events = 0

    for t in range(GEN_TOKENS):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=device)

        # Stabilized stretch schedule (your old one can explode)
        eye_offset = 1.0 + 0.15 * (1 if (t % 20) < 10 else -1)

        logits = model(ctx, eye_offset=eye_offset).squeeze(0) / max(TEMP, 1e-6)
        probs = F.softmax(logits, dim=-1)

        p_max = float(probs.max().item())
        H = entropy(probs)

        # Commit a token (this is where generation "makes history")
        next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)

        # Measure self-surprise after committing it (proxy for "this step forced a weird arc")
        nll2, p2, H2, y2 = next_step_surprise(model, ids, eye_offset=eye_offset)

        # "False memory birth" flag:
        # model acted confident at step t, but immediately after committing, it becomes surprised / unstable
        is_false_event = (p_max >= CONFIDENT_P and H <= LOW_ENTROPY and nll2 >= HIGH_SURPRISE_NLL)
        if is_false_event:
            false_events += 1

        w = i2w.get(next_id, "?")
        print(w, end=" ", flush=True)

       
    print(f"\n\n>> False-memory-like events flagged: {false_events} / {GEN_TOKENS}\n")


# -------------------------
# Main Execution (Save/Load)
# -------------------------
if __name__ == "__main__":
    MODEL_PATH = "persona_model.pth"
    META_PATH = "persona_meta.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        print(">> Loading existing checkpoint...")
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
        w2i, i2w, v_size = meta['w2i'], meta['i2w'], meta['v_size']

        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print(">> No checkpoint found. Initializing training...")
        ts_words, sq_words = fetch_hf_corpus()
        vocab = sorted(list(set(ts_words + sq_words)))
        w2i = {w: i for i, w in enumerate(vocab)}
        i2w = {i: w for w, i in w2i.items()}
        v_size = len(vocab)

        loader = DataLoader(
            AlignedDataset(ts_words, sq_words, w2i, SEQ_LEN),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        model = PersonaNeuralNet(v_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)

        run_training(model, loader, v_size)

        torch.save(model.state_dict(), MODEL_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump({'w2i': w2i, 'i2w': i2w, 'v_size': v_size}, f)
        print(f">> Model saved to {MODEL_PATH}")

    while True:
        try:
            prompt = input("USER: ")
            if not prompt:
                continue
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            generate(model, prompt, w2i, i2w)
        except KeyboardInterrupt:
            break

import os
import re
import requests
from collections import defaultdict, Counter

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
KB_len = -1
CKPT_PATH = "rnn_alt_fgsm_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/data hyperparams (saved into checkpoint too)
SEQ_LEN = 8
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Training hyperparams
BATCH_SIZE = 512
LR = 1e-2
NUM_EPOCHS = 15

EPS_START = 0.10
EPS_MAX = 0.30
EPS_GROW_EVERY = 4
EPS_GROW_MULT = 1.15

ADV_EVERY = 4          # 2 => clean, adv, clean, adv...
EMB_CLAMP = 2.0
GRAD_CLIP_NORM = 1.0

# Markov corridor config
MARKOV_ORDER = 2       # n-gram order
CORRIDOR_LEN = 8       # how many visible recent words
MARKOV_ALPHA = 0.5     # blend between GRU and Markov
MARKOV_TEMP = 1.0      # temperature for Markov probs

# -------------------------
# Dataset
# -------------------------
class SeqDataset(Dataset):
    """
    Takes a word stream and returns (x_seq, y_next).
    x_seq: (T,) token ids
    y_next: ()  token id for next word
    """
    def __init__(self, words, word_to_ix, seq_len=8):
        self.seq_len = seq_len
        self.word_to_ix = word_to_ix

        ids = [self.word_to_ix.get(w, 0) for w in words]
        self.samples = []
        for i in range(len(ids) - seq_len):
            x = ids[i:i + seq_len]
            y = ids[i + seq_len]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -------------------------
# Model
# -------------------------
class RNNNextWord(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """
        x: (B,T) long
        h: (num_layers,B,H) or None
        returns: logits (B,V), h_next
        """
        emb = self.embedding(x)                 # (B,T,E)
        out, h_next = self.rnn(emb, h)          # out: (B,T,H)
        logits = self.fc_out(out[:, -1, :])     # (B,V)
        return logits, h_next

    def forward_step(self, token_id, h=None):
        """
        token_id: (B,1) long
        """
        emb = self.embedding(token_id)          # (B,1,E)
        out, h_next = self.rnn(emb, h)          # (B,1,H)
        logits = self.fc_out(out[:, -1, :])     # (B,V)
        return logits, h_next

# -------------------------
# Save / load
# -------------------------
def save_checkpoint(model, optimizer, epoch, loss, word_to_ix, ix_to_word, path=CKPT_PATH,
                    seq_len=SEQ_LEN, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
    ckpt = {
        "epoch": epoch,
        "loss": float(loss),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
        "seq_len": seq_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    torch.save(ckpt, path)
    print(f"💾 Saved checkpoint to {path}")

def load_checkpoint(path=CKPT_PATH):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"📂 Loaded checkpoint epoch={ckpt['epoch']} loss={ckpt['loss']:.3f}")
    return ckpt

# -------------------------
# Markov (corridor compaction)
# -------------------------
def build_markov_model(words, order=2):
    """
    Simple n-gram Markov model over word strings.
    model[context_tuple] -> Counter(next_word_string)
    """
    model = defaultdict(Counter)
    if len(words) <= order:
        return model
    for i in range(len(words) - order):
        ctx = tuple(words[i:i+order])
        nx = words[i+order]
        model[ctx][nx] += 1
    return model

def markov_next_dist(model, context_words, word_to_ix, temp=1.0):
    """
    context_words: list[str], recent words
    Returns a 1D tensor over vocab with probabilities or None if no context found.
    Backoff uses decreasing context length.
    """
    if len(context_words) == 0:
        return None

    max_order = min(len(context_words), MARKOV_ORDER)
    for order in range(max_order, 0, -1):
        ctx = tuple(context_words[-order:])
        if ctx in model:
            cnt = model[ctx]
            total = sum(cnt.values())
            probs = torch.zeros(len(word_to_ix), dtype=torch.float32)
            for w, c in cnt.items():
                idx = word_to_ix.get(w, 0)
                probs[idx] = c / total
            if temp != 1.0:
                # soft temperature on Markov distribution
                probs = probs.pow(1.0 / temp)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                return None
            probs = probs / probs_sum
            return probs
    return None

# -------------------------
# FGSM on embeddings
# -------------------------
def fgsm_embeddings(model, x, y, criterion, epsilon, clamp_val=2.0):
    """
    Create adversarial embeddings for x via FGSM, without backpropagating into model params
    during perturbation construction.
    """
    emb = model.embedding(x).detach().requires_grad_(True)    # (B,T,E)
    out, _ = model.rnn(emb)                                   # (B,T,H)
    logits = model.fc_out(out[:, -1, :])                      # (B,V)
    loss = criterion(logits, y)

    grad = torch.autograd.grad(loss, emb, retain_graph=False, create_graph=False)[0]
    emb_adv = emb + epsilon * grad.sign()
    emb_adv = torch.clamp(emb_adv, -clamp_val, clamp_val).detach()
    return emb_adv

# -------------------------
# Train (alternating)
# -------------------------
def train_epoch_alternating(model, optimizer, criterion, loader,
                            epsilon=0.15, adv_every=2, clamp_val=2.0):
    """
    Alternates computation: clean update, then adversarial update, repeating.
    adv_every=2 => clean (step 0), adv (step 1), clean (step 2), adv (step 3), ...
    """
    model.train()
    total_loss, batches = 0.0, 0

    progress_bar = tqdm(loader, desc=f"Epoch alt ε={epsilon:.2f}", leave=False)
    for step, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        do_adv = (step % adv_every) == (adv_every - 1)

        if not do_adv:
            logits, _ = model(x)
            loss = criterion(logits, y)
        else:
            emb_adv = fgsm_embeddings(model, x, y, criterion, epsilon, clamp_val=clamp_val)
            out_adv, _ = model.rnn(emb_adv)
            logits_adv = model.fc_out(out_adv[:, -1, :])
            loss = criterion(logits_adv, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / max(1, batches)

# -------------------------
# Generation (plain GRU)
# -------------------------
@torch.no_grad()
def generate(model, word_to_ix, ix_to_word, seed_text, length=50, temp=0.8):
    model.eval()

    seed_words = seed_text.split()
    if len(seed_words) == 0:
        seed_words = ["the"]

    # Prime hidden state with all seed tokens
    h = None
    ids = [word_to_ix.get(w, 0) for w in seed_words]
    x0 = torch.tensor([ids], device=device, dtype=torch.long)  # (1,Tseed)
    logits, h = model(x0, h=h)

    generated = list(seed_words)
    cur_id = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)  # (1,1)

    for _ in range(length):
        logits, h = model.forward_step(cur_id, h=h)
        probs = F.softmax(logits[0] / max(1e-6, temp), dim=-1)
        next_ix = torch.multinomial(probs, 1).item()
        generated.append(ix_to_word[next_ix])
        cur_id = torch.tensor([[next_ix]], device=device, dtype=torch.long)

    return " ".join(generated)

# -------------------------
# Generation (Markov corridor)
# -------------------------
@torch.no_grad()
def generate_corridor(model, word_to_ix, ix_to_word,
                      seed_text, markov_model,
                      corridor_len=CORRIDOR_LEN,
                      length=50, temp=0.8,
                      alpha=MARKOV_ALPHA,
                      markov_temp=MARKOV_TEMP):
    """
    Hybrid RNN + Markov corridor:
      - GRU sees only recent tokens (like usual).
      - Markov model sees recent words, acts as compact summary of longer history.
      - Next-token probs = (1-alpha)*GRU + alpha*Markov (when Markov info available).
    """
    model.eval()

    seed_words = seed_text.split()
    if len(seed_words) == 0:
        seed_words = ["the"]

    # corridor over seed
    ctx_words = seed_words[-corridor_len:]

    # prime GRU on corridor ids only
    h = None
    ids = [word_to_ix.get(w, 0) for w in ctx_words]
    x0 = torch.tensor([ids], device=device, dtype=torch.long)
    logits, h = model(x0, h=h)

    generated = list(seed_words)
    cur_id = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)

    for _ in range(length):
        logits, h = model.forward_step(cur_id, h=h)
        nn_probs = F.softmax(logits[0] / max(1e-6, temp), dim=-1)

        # Markov corridor: use words in current visible corridor
        ctx_words = generated[-corridor_len:]
        mk_probs = markov_next_dist(markov_model, ctx_words, word_to_ix, temp=markov_temp)
        if mk_probs is not None:
            mk_probs = mk_probs.to(nn_probs.device)
            probs = (1.0 - alpha) * nn_probs + alpha * mk_probs
            probs = probs / probs.sum().clamp_min(1e-8)
        else:
            probs = nn_probs

        next_ix = torch.multinomial(probs, 1).item()
        next_word = ix_to_word[next_ix]
        generated.append(next_word)
        cur_id = torch.tensor([[next_ix]], device=device, dtype=torch.long)

    return " ".join(generated)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    ckpt = load_checkpoint(CKPT_PATH)

    markov_model = None
    words = None

    if ckpt is not None:
        word_to_ix = ckpt["word_to_ix"]
        ix_to_word = ckpt["ix_to_word"]
        vocab_size = len(word_to_ix)

        seq_len = int(ckpt.get("seq_len", SEQ_LEN))
        embed_dim = int(ckpt.get("embed_dim", EMBED_DIM))
        hidden_dim = int(ckpt.get("hidden_dim", HIDDEN_DIM))
        num_layers = int(ckpt.get("num_layers", NUM_LAYERS))

        model = RNNNextWord(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        model.load_state_dict(ckpt["model_state"])

        optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer.load_state_dict(ckpt["optim_state"])
        criterion = nn.CrossEntropyLoss()

        start_epoch = int(ckpt["epoch"])
        print("✅ Model fully loaded from checkpoint.")
        print(f"ℹ️  Vocab={vocab_size} | seq_len={seq_len} | embed={embed_dim} | hidden={hidden_dim} | layers={num_layers}")

        # If you still have your original corpus file around, you can rebuild words & Markov here.
        # Otherwise Markov corridor will be unavailable until you retrain once with this script.
        # For safety, keep markov_model as None here unless you reload text.

    else:
        # Load corpus
        try:
            filename = input("Filename (blank = fallback Shakespeare): ").strip()
            if filename:
                with open(filename, "r", encoding="utf-8") as f:
                    text = f.read().lower()
            else:
                raise FileNotFoundError
            print(f"✅ Loaded '{filename}'")
        except FileNotFoundError:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url, timeout=30).text.lower()
            print("✅ Loaded Shakespeare (fallback)")

        words = text.split()[:KB_len]

        # Stable vocab
        vocab = sorted(set(words))
        word_to_ix = {w: i for i, w in enumerate(vocab)}
        ix_to_word = {i: w for w, i in word_to_ix.items()}
        vocab_size = len(word_to_ix)

        print(f"Vocab: {vocab_size}, Words: {len(words)}")

        dataset = SeqDataset(words, word_to_ix, seq_len=SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Samples: {len(dataset)} (seq_len={SEQ_LEN})")

        model = RNNNextWord(vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # Build Markov model (for corridor compaction)
        print("🧮 Building Markov corridor model...")
        markov_model = build_markov_model(words, order=MARKOV_ORDER)
        print(f"Markov model states: {len(markov_model)}")

        print("🚀 RNN (GRU) + Alternating Clean/FGSM Training")
        epsilon = EPS_START

        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training"):
            avg_loss = train_epoch_alternating(
                model, optimizer, criterion, train_loader,
                epsilon=epsilon, adv_every=ADV_EVERY, clamp_val=EMB_CLAMP
            )
            print(f"Epoch {epoch:2d}: avg_loss={avg_loss:.3f}")


        save_checkpoint(
            model, optimizer, NUM_EPOCHS, avg_loss,
            word_to_ix, ix_to_word, path=CKPT_PATH,
            seq_len=SEQ_LEN, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
        )

    # If we didn't build Markov because we only loaded a ckpt, corridor gen will fall back to pure GRU
    if markov_model is None:
        print("ℹ️  No Markov model built in this run; corridor generation will fall back to pure GRU.")

    # Interactive generation
    print("\n🎯 Interactive mode:")
    print("Type text and press Enter. Ctrl+C to exit.")
    while True:
        try:
            cmd = input("SEED TEXT: ").strip()
            if not cmd:
                continue
            seed = cmd[3:].strip() or "the"
            if markov_model is not None:
                out = generate_corridor(
                    model, word_to_ix, ix_to_word,
                    seed_text=seed,
                    markov_model=markov_model,
                    corridor_len=CORRIDOR_LEN,
                    length=800,
                    temp=0.8,
                    alpha=MARKOV_ALPHA,
                    markov_temp=MARKOV_TEMP
                )
        
            print(f"CONTINUATION: {out}\n")
        except KeyboardInterrupt:
            print("\nExiting!")
            break

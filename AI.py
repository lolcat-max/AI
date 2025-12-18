import os
import requests
import math
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
CKPT_PATH = "slope_tensor_fusion.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
SEQ_LEN = 8
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Training
BATCH_SIZE = 512
LR = 5e-3
NUM_EPOCHS = 1

EPS_START = 0.10
EPS_MAX = 0.30
EPS_GROW_EVERY = 4
EPS_GROW_MULT = 1.15

ADV_EVERY = 4
EMB_CLAMP = 2.0
GRAD_CLIP_NORM = 1.0

# -------------------------
# 1. Non-Commutative Physics Operators
# -------------------------
class NonCommutativeMatMul(torch.autograd.Function):
    """
    Explicit autograd for A @ B emphasizing non-commutative operator mechanics.
    """
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B

    @staticmethod
    def backward(ctx, grad_out):
        A, B = ctx.saved_tensors
        # dL/dA = dL/dC @ B^T
        grad_A = grad_out @ B.transpose(-1, -2)
        # dL/dB = A^T @ dL/dC
        grad_B = A.transpose(-1, -2) @ grad_out
        return grad_A, grad_B

nc_matmul = NonCommutativeMatMul.apply

class NCLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (math.sqrt(fan_in)) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = nc_matmul(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            y = y + self.bias
        return y

# -------------------------
# 2. Slope Dynamics Module
# -------------------------
class SlopeTensorLayer(nn.Module):
    """
    Calculates the first derivative (slope) of the embedding sequence to model momentum.
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.slope_projector = NCLinear(embed_dim, hidden_dim)
        
    def forward(self, emb):
        # d[t] = emb[t] - emb[t-1]
        slope = emb[:, 1:, :] - emb[:, :-1, :] 
        
        # Pad t=0 with zero velocity
        b, _, e = slope.shape
        zero_pad = torch.zeros(b, 1, e, device=emb.device, dtype=emb.dtype)
        slope = torch.cat([zero_pad, slope], dim=1) 
        
        return F.tanh(self.slope_projector(slope))

# -------------------------
# 3. Component Models
# -------------------------
class BroadbandQuarterWaveNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = NCLinear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)

class RNNBranch(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = NCLinear(hidden_dim, vocab_size)

    def forward(self, emb, h=None):
        out, h_next = self.rnn(emb, h)
        return out, h_next 

# -------------------------
# 4. Fusion Paradigm Model
# -------------------------
class SemanticFusionNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Branch 1: Particle Dynamics
        self.rnn_branch = RNNBranch(embed_dim, hidden_dim, vocab_size, num_layers)
        
        # Branch 2: Slope Dynamics
        self.slope_layer = SlopeTensorLayer(embed_dim, hidden_dim)
        self.slope_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Branch 3: Wave Impedance
        self.wave_branch = BroadbandQuarterWaveNet(embed_dim, vocab_size)
        
        # Final Readout
        self.fc_final = NCLinear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        emb = self.embedding(x) # (B, T, E)

        # Compute Slope (Velocity)
        slope_h = self.slope_layer(emb)

        # Particle Path (RNN)
        rnn_out, h_next = self.rnn_branch.forward(emb, h)

        # Slope Fusion
        combined = torch.cat([rnn_out, slope_h], dim=-1)
        gate = torch.sigmoid(self.slope_gate(combined))
        fused_state = rnn_out + (gate * slope_h)
        
        logits_dynamics = self.fc_final(fused_state[:, -1, :])

        # Wave Path (CNN)
        emb_permuted = emb.transpose(1, 2)
        logits_wave = self.wave_branch(emb_permuted)

        return logits_dynamics + logits_wave, h_next

    def forward_from_embeddings(self, emb, h=None):
        slope_h = self.slope_layer(emb)
        rnn_out, h_next = self.rnn_branch.forward(emb, h)
        
        combined = torch.cat([rnn_out, slope_h], dim=-1)
        gate = torch.sigmoid(self.slope_gate(combined))
        fused_state = rnn_out + (gate * slope_h)
        logits_dynamics = self.fc_final(fused_state[:, -1, :])
        
        emb_permuted = emb.transpose(1, 2)
        logits_wave = self.wave_branch(emb_permuted)
        
        return logits_dynamics + logits_wave, h_next

# -------------------------
# 5. Data & Training Utils
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, words, word_to_ix, seq_len=8):
        self.seq_len = seq_len
        self.word_to_ix = word_to_ix
        unk = self.word_to_ix.get("<unk>", 0)
        ids = [self.word_to_ix.get(w, unk) for w in words]
        self.samples = []
        for i in range(len(ids) - seq_len):
            x = ids[i : i + seq_len]
            y = ids[i + seq_len]
            self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def fgsm_accelerate(model, x, y, criterion, epsilon, clamp_val=2.0):
    emb = model.embedding(x).detach().requires_grad_(True)
    logits, _ = model.forward_from_embeddings(emb, h=None)
    loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, emb, retain_graph=False, create_graph=False)[0]
    emb_adv = emb + epsilon * grad.sign()
    emb_adv = torch.clamp(emb_adv, -clamp_val, clamp_val).detach()
    return emb_adv

def train_epoch_fusion(model, optimizer, criterion, loader, epsilon, adv_every=2, clamp_val=2.0):
    model.train()
    total_loss, batches = 0.0, 0
    pbar = tqdm(loader, desc=f"Slope Fusion eps={epsilon:.2f}", leave=False)
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        if (step % adv_every) != (adv_every - 1):
            logits, _ = model(x)
            loss = criterion(logits, y)
        else:
            emb_adv = fgsm_accelerate(model, x, y, criterion, epsilon, clamp_val)
            logits_adv, _ = model.forward_from_embeddings(emb_adv)
            loss = criterion(logits_adv, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})
    return total_loss / max(1, batches)

@torch.no_grad()
def generate(model, word_to_ix, ix_to_word, seed_text, length=50, temp=0.8):
    model.eval()
    seed_words = seed_text.lower().split()
    if not seed_words: seed_words = ["the"]
    unk = word_to_ix.get("<unk>", 0)
    current_ids = [word_to_ix.get(w, unk) for w in seed_words]
    generated = list(seed_words)
    
    for _ in range(length):
        if len(current_ids) < SEQ_LEN:
            inp = [0]*(SEQ_LEN - len(current_ids)) + current_ids
        else:
            inp = current_ids[-SEQ_LEN:]
            
        x_tens = torch.tensor([inp], device=device, dtype=torch.long)
        logits, _ = model(x_tens) 
        probs = F.softmax(logits[0] / temp, dim=-1)
        next_ix = torch.multinomial(probs, 1).item()
        word = ix_to_word[next_ix]
        generated.append(word)
        current_ids.append(next_ix)

    return " ".join(generated)

# -------------------------
# 6. Save / Load Logic
# -------------------------
def save_checkpoint(model, optimizer, epoch, loss, word_to_ix, ix_to_word, path):
    print(f"\nSaving checkpoint to {path}...")
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
    }, path)
    print("Done.")

def load_checkpoint(path, model, optimizer):
    if not os.path.exists(path):
        return None, 1, EPS_START
    
    print(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device)
    
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    
    # Restore vocab maps from checkpoint to ensure consistency
    word_to_ix = ckpt["word_to_ix"]
    ix_to_word = ckpt["ix_to_word"]
    
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {ckpt['epoch']} (Loss: {ckpt['loss']:.4f})")
    
    return (word_to_ix, ix_to_word), start_epoch, EPS_START

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    try:
        filename = "xaa"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f: text = f.read().lower()
            print(f"Loaded local '{filename}'")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Downloading Shakespeare fallback...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url, timeout=30).text.lower()
    
    # Init Vocab (will be overwritten if checkpoint exists)
    words = text.split()
    if KB_len > 0: words = words[:KB_len]
    uniq = sorted(list(set(words)))
    vocab = ["<unk>"] + uniq
    word_to_ix = {w:i for i,w in enumerate(vocab)}
    ix_to_word = {i:w for w,i in word_to_ix.items()}
    vocab_size = len(vocab)
    print(f"Initial Vocab: {vocab_size} | Tokens: {len(words)}")

    dataset = SeqDataset(words, word_to_ix, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Model Init
    model = SemanticFusionNet(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Load Checkpoint
    vocab_data, start_epoch, epsilon = load_checkpoint(CKPT_PATH, model, optimizer)
    if vocab_data:
        word_to_ix, ix_to_word = vocab_data
        # Note: If vocab size changed, model loading would have failed above.
        # Strict consistency is assumed.

    print("\nStarting Slope Tensor Fusion Training...")
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            loss = train_epoch_fusion(model, optimizer, criterion, loader, epsilon, ADV_EVERY, EMB_CLAMP)
            
            if epoch % EPS_GROW_EVERY == 0: 
                epsilon = min(EPS_MAX, epsilon * EPS_GROW_MULT)
            
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Eps: {epsilon:.3f}")
            
            # Save every epoch
            save_checkpoint(model, optimizer, epoch, loss, word_to_ix, ix_to_word, CKPT_PATH)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        save_checkpoint(model, optimizer, epoch, loss, word_to_ix, ix_to_word, CKPT_PATH)

    print("\nInteractive Generator:")
    while True:
        try:
            seed = input(">> ")
            if not seed: continue
            print(generate(model, word_to_ix, ix_to_word, seed, length=100))
        except KeyboardInterrupt: break

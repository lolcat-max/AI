import os, pickle, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from tqdm import tqdm
import requests  # needed for TinyShakespeare fallback

SEQ_LEN, E, H, B, LR, EPOCHS = 3, 180, 1160, 1640, 1e-2, 1
D = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ---------------- Data ----------------
class MicroData:
    def __init__(self, ids_tensor):
        self.ids = ids_tensor.to(D)
        # ensure we don't index past end; cap at len-SEQ_LEN-1
        self.n_safe = min(5500, len(self.ids) - SEQ_LEN - 1)

    def sample(self, size):
        idx = torch.randint(0, self.n_safe, (size,), device=D)
        positions = torch.stack([idx + i for i in range(SEQ_LEN)], dim=1)  # [size, SEQ_LEN]
        x = torch.gather(self.ids, 0, positions.flatten()).reshape(size, SEQ_LEN)
        y = self.ids[idx + SEQ_LEN]
        return x, y, torch.ones(size, device=D)



# ---------------- Model ----------------
class NanoNet(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.e = nn.Embedding(v, E)
        self.g = nn.GRU(E, H, 1, batch_first=True)
        self.l = nn.Linear(H, v, bias=False)

    def forward(self, x):
        x = self.e(x)
        o, _ = self.g(x)
        return self.l(o[:, -1] * 1.1)

def batch():
    size = B // 2
    x1, y1, w1 = ts.sample(size)
    x2, y2, w2 = sq.sample(size)
    return torch.cat([x1, x2]), torch.cat([y1, y2]), torch.cat([w1, w2 * 2.5])

def train(m):
    o = optim.Adam(m.parameters(), LR)
    m.train()
    for e in range(EPOCHS):
        p = tqdm(range(300))
        L = 0
        for _ in p:
            x, y, w = batch()
            o.zero_grad()
            logits = m(x)
            loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            o.step()
            L += loss.item()
            p.set_postfix(loss=f"{loss.item():.3f}")
        print(f"E{e+1}: {L/300:.3f}")

# ---------------- Generation ----------------
@torch.no_grad()
def generate(m, seed, w2i, i2w, length=80, temp=0.8, top_p=0.9):
    ids = [w2i.get(w, 0) for w in seed.lower().split()][-SEQ_LEN:]
    if len(ids) < SEQ_LEN:
        ids = [0] * (SEQ_LEN - len(ids)) + ids
    print(f"\n📝 '{seed}' →", end=" ")

    for _ in range(length):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=D)
        logits = m(ctx)[0] / temp

        # NaN/Inf proof
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = torch.clamp(logits, -20, 20)

        sorted_l, sorted_i = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_l, dim=0)
        probs = torch.nan_to_num(probs, nan=1.0 / len(w2i), posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum()

        next_id = torch.multinomial(probs, 1).item()
        token_id = sorted_i[next_id].item()
        ids.append(token_id)
        print(i2w.get(token_id, '?'), end=" ", flush=True)
    print()

# ---------------- Portable checkpoint (model + vocab) ----------------
CKPT_PATH = "nano_portable.pth"

if os.path.exists(CKPT_PATH):
    print("🔥 LOADING PORTABLE CHECKPOINT...")
    ckpt = torch.load(CKPT_PATH, map_location=D)
    V_loaded = ckpt["v_size"]
    W2I = ckpt["w2i"]
    I2W = ckpt["i2w"]

    m = NanoNet(V_loaded).to(D)
    m.load_state_dict(ckpt["model_state_dict"])
else:
    
    
    print("⚡ TRAINING NEW MODEL...")
        
    # ---------------- Corpus + Vocab (build once) ----------------
    try:
        with open(input("Core Filename (Enter for TinyShakespeare): ").strip() or "tinyshakespeare.txt",
                  "r", encoding="utf-8") as f:
            CORPUS = f.read().lower().strip().split()
    except:
        try:
            CORPUS = requests.get(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                timeout=10
            ).text.lower().split()
        except:
            CORPUS = ("hello world " * 1000).strip().split()
    VOCAB = sorted(set(CORPUS))
    V = len(VOCAB)
    W2I = {w: i for i, w in enumerate(VOCAB)}
    I2W = {i: w for i, w in enumerate(VOCAB)}
    ids_tensor = torch.tensor([W2I[w] for w in CORPUS], dtype=torch.long)
    ts = MicroData(ids_tensor)
    sq = MicroData(ids_tensor)
    m = NanoNet(V).to(D)
    train(m)

    ckpt = {
        "v_size": V,
        "w2i": W2I,
        "i2w": I2W,
        "model_state_dict": m.state_dict(),
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"💾 SAVED PORTABLE CHECKPOINT TO {CKPT_PATH}")

print("🎯 Portable TinyShakespeare GEN! t=temp p=top_p l=length")

while True:
    try:
        inp = input(">> ").strip()
        if inp.lower() in ['q', 'quit', 'exit']:
            break
        parts = inp.split()
        seed = parts[0]
        kw = {'length': 800, 'temp': 0.8, 'top_p': 0.9}
        for p in parts[1:]:
            if p.startswith('t='): kw['temp'] = float(p[2:])
            if p.startswith('p='): kw['top_p'] = float(p[2:])
            if p.startswith('l='): kw['length'] = int(p[2:])
        generate(m, seed, W2I, I2W, **kw)
    except KeyboardInterrupt:
        break

import os, pickle, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from tqdm import tqdm
import requests
import gradio as gr

SEQ_LEN, E, H, B, LR, EPOCHS = 3, 180, 1160, 1640, 1e-2, 1
D = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Data ----------------
class MicroData:
    def __init__(self, ids_tensor):
        self.ids = ids_tensor.to(D)
        self.n_safe = min(5500, len(self.ids) - SEQ_LEN - 1)

    def sample(self, size):
        idx = torch.randint(0, self.n_safe, (size,), device=D)
        positions = torch.stack([idx + i for i in range(SEQ_LEN)], dim=1)
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

# ---------------- Compendium Encoder (No Unknown Tokens) ----------------
class CompendiumEncoder:
    def __init__(self, vocab, w2i):
        self.vocab = vocab
        self.w2i = w2i
        self.char_to_id = {}
        for char_ord in range(256):
            char = chr(char_ord)
            min_dist, best_id = float('inf'), 0
            for i, word in enumerate(vocab):
                if len(word) == 1:
                    dist = abs(ord(char) - ord(word))
                    if dist < min_dist:
                        min_dist, best_id = dist, i
                elif char in word:
                    dist = 0
                    if dist < min_dist:
                        min_dist, best_id = dist, i
            self.char_to_id[char_ord] = best_id
    
    def encode_unknown(self, word):
        ids = []
        for c in word.lower():
            char_ord = ord(c)
            tid = self.char_to_id.get(char_ord, 0)
            ids.append(tid)
        return ids


# ---------------- Enhanced Generation ----------------
@torch.no_grad()
def generate_text(m, seed, w2i, i2w, compendium, length=80, temp=0.8, top_p=0.9, grep_mode=False):
    full_seed = seed
    
    # Compendium tokenization: known + char-fallback
    ids = []
    for w in full_seed.lower().split():
        if w in w2i:
            ids.append(w2i[w])
        else:
            ids.extend(compendium.encode_unknown(w))
    ids = ids[-SEQ_LEN:]
    if len(ids) < SEQ_LEN:
        ids = [0] * (SEQ_LEN - len(ids)) + ids

    out_words = []
    for _ in range(length):
        ctx = torch.tensor([ids[-SEQ_LEN:]], device=D)
        logits = m(ctx)[0] / temp
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
        out_words.append(i2w.get(token_id, '?'))
    return " ".join(out_words)

# ---------------- Portable checkpoint ----------------
CKPT_PATH = "nano_portable.pth"

# ---------------- Portable checkpoint ----------------
CKPT_PATH = "nano_portable.pth"

if os.path.exists(CKPT_PATH):
    print("🔥 LOADING PORTABLE CHECKPOINT...")
    ckpt = torch.load(CKPT_PATH, map_location=D)
    V = ckpt["v_size"]
    
    # Backward compatible vocab loading
    if "vocab" in ckpt:
        VOCAB = ckpt["vocab"]
    else:
        VOCAB = sorted([w for w in ckpt["w2i"]])
        print("📚 Reconstructed VOCAB from legacy checkpoint")
    
    W2I = ckpt["w2i"]
    I2W = ckpt["i2w"]
    compendium_char_to_id = ckpt.get("compendium_char_to_id", {})
    
    m = NanoNet(V).to(D)
    m.load_state_dict(ckpt["model_state_dict"])
    
    compendium = CompendiumEncoder(VOCAB, W2I)
    compendium.char_to_id = compendium_char_to_id
    print("✅ Loaded model + compendium (handles unknown words)")
else:
    print("⚡ TRAINING NEW MODEL...")
    try:
        filename = input("Core Filename (Enter for TinyShakespeare): ").strip() or "tinyshakespeare.txt"
        with open(filename, "r", encoding="utf-8") as f:
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
    
    compendium = CompendiumEncoder(VOCAB, W2I)
    
    ckpt = {
        "v_size": V,
        "vocab": VOCAB,
        "w2i": W2I,
        "i2w": I2W,
        "compendium_char_to_id": compendium.char_to_id,
        "model_state_dict": m.state_dict(),
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"💾 SAVED PORTABLE CHECKPOINT TO {CKPT_PATH}")

print("🎯 NanoNet + Compendium + Grep UI")

# ---------------- Gradio UI ----------------
def ui_generate(seed, temp, top_p, length, grep_mode):
    if not seed.strip():
        seed = "the"
    text = generate_text(m, seed, W2I, I2W, compendium, 
                        length=int(length), temp=float(temp), top_p=float(top_p), 
                        grep_mode=grep_mode)
    prefix = "grep → " if grep_mode else ""
    return f"{prefix}'{seed}' → {text}"

with gr.Blocks(title="NanoNet Compendium") as demo:
    gr.Markdown("# 🚀 NanoNet: Shakespeare + Unknown Words")
    gr.Markdown("**Compendium**: Encodes ANY word via char→vocab-index mapping. No OOV!")
    
    with gr.Row():
        seed = gr.Textbox(label="Seed (try 'quantum🤖')", value="the", lines=1)
    with gr.Row():
        temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        length = gr.Slider(10, 800, value=80, step=10, label="Length")
        grep_mode = gr.Checkbox(label="🖥️ Grep Mode", value=False)
    
    out = gr.Textbox(label="Generation", lines=10, max_lines=15)
    
    # Live regeneration on ALL changes
    seed.change(ui_generate, inputs=[seed, temp, top_p, length, grep_mode], outputs=out)
    temp.change(ui_generate, inputs=[seed, temp, top_p, length, grep_mode], outputs=out)
    top_p.change(ui_generate, inputs=[seed, temp, top_p, length, grep_mode], outputs=out)
    length.change(ui_generate, inputs=[seed, temp, top_p, length, grep_mode], outputs=out)
    grep_mode.change(ui_generate, inputs=[seed, temp, top_p, length, grep_mode], outputs=out)
    
    btn = gr.Button("🔥 Generate", variant="primary", size="lg")
    btn.click(ui_generate, inputs=[seed, temp, top_p, length, grep_mode], outputs=out)
    
    gr.Markdown("**Tips**: `quantum foo🤖` (compendium)")

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)

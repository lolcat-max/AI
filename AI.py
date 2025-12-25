import os
import requests
import re
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 1024
LR = 5e-3
NUM_EPOCHS = 1

# -------------------------
# 1. Generic Q&A Data Fetching
# -------------------------
def fetch_hf_data():
    """Extracts questions and answers from a generic Q&A dataset."""
    # Built-in generic Q&A - always reliable
    generic_qa = """
    what is python? python is a high-level programming language known for simplicity and readability.
    who created linux? linus torvalds created the linux operating system kernel in 1991.
    where is silicon valley? silicon valley is located in the san francisco bay area of california.
    when was the internet invented? the internet was invented in the late 1960s with arpanet.
    why do we use databases? we use databases to efficiently store organize and retrieve structured data.
    how does machine learning work? machine learning uses algorithms and statistical models to learn patterns from data.
    what is artificial intelligence? artificial intelligence is technology that enables machines to simulate human intelligence.
    who invented the telephone? alexander graham bell invented the first practical telephone in 1876.
    where is the eiffel tower? the eiffel tower is located in paris france on the champ de mars.
    when was world war two? world war two lasted from 1939 to 1945 involving most nations.
    why is water important? water is essential for all known forms of life and biological processes.
    how do computers process information? computers process information using binary code transistors and logical circuits.
    what is the capital of france? the capital of france is paris a major european city.
    who wrote hamlet? william shakespeare wrote the tragedy of hamlet prince of denmark.
    where do penguins live? penguins live primarily in antarctica and cold southern hemisphere regions.
    when was the first airplane flight? the wright brothers achieved the first powered flight in 1903.
    why is the sky blue? the sky appears blue due to rayleigh scattering of sunlight by atmosphere.
    how does photosynthesis work? photosynthesis converts light energy into chemical energy using chlorophyll in plants.
    what is democracy? democracy is a system of government where power is vested in the people.
    who discovered penicillin? alexander fleming discovered penicillin antibiotics in 1928 revolutionizing medicine.
    what is gravity? gravity is the force that attracts objects with mass toward each other.
    who painted the mona lisa? leonardo da vinci painted the mona lisa during the renaissance period.
    where is mount everest? mount everest is located in the himalayas on the nepal-tibet border.
    when was the declaration of independence signed? the declaration of independence was signed on july 4 1776.
    why do seasons change? seasons change due to earth's axial tilt as it orbits the sun.
    how do vaccines work? vaccines work by training the immune system to recognize and fight specific pathogens.
    what is quantum physics? quantum physics studies the behavior of matter and energy at atomic scales.
    who founded microsoft? bill gates and paul allen founded microsoft corporation in 1975.
    where is the great wall of china? the great wall of china stretches across northern china spanning thousands of miles.
    when did the moon landing occur? the first moon landing occurred on july 20 1969 with apollo 11.
    """
    
    # Parse the Q&A pairs
    lines = [line.strip() for line in generic_qa.strip().split('\n') if line.strip()]
    questions = []
    answers = []
    
    for line in lines:
        if '?' in line:
            parts = line.split('?', 1)
            if len(parts) == 2:
                questions.append(parts[0].strip().lower())
                answers.append(parts[1].strip().lower())
    
    # Extract question types
    question_types = set()
    for q in questions:
        words = q.split()
        if words:
            first_word = words[0]
            if first_word in ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']:
                question_types.add(first_word)
    
    # Combine all text
    all_text = " ".join(questions + answers)
    all_words = all_text.split()
    
    print(f"  Loaded {len(questions)} Q&A pairs, {len(all_words)} words")
    return list(question_types), all_words

def extract_methods(words):
    """Detects action-based Methods using morphological heuristics."""
    candidates = []
    action_suffixes = r".+(ing|ed|ion|al|ment|ance|ive|ize)$"
    for w in words:
        if re.match(action_suffixes, w) and len(w) > 4:
            candidates.append(w)
    m_set = {w for w, c in Counter(candidates).items() if c >= 2}
    m_set.update({"act", "translate", "summarize", "generate", "write", "solve"})
    return m_set

# -------------------------
# 2. Logic Gates
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
    def __init__(self, words, w2i, v_size):
        self.matrix = torch.zeros((v_size, v_size), device=device)
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if w1 in w2i and w2 in w2i: self.matrix[w2i[w1], w2i[w2]] = 1.0
    def get_grounding(self, last_id, v_size):
        mask = torch.full((v_size,), -float('inf'), device=device)
        valid = torch.where(self.matrix[last_id] > 0)[0]
        if len(valid) > 0: mask[valid] = 0.0
        else: mask.fill_(0.0)
        return mask

# -------------------------
# 3. Model & Weighted Dataset with Source Tracking
# -------------------------
class PersonaNeuralNet(nn.Module):
    def __init__(self, v_size, e_dim, h_dim, layers):
        super().__init__()
        self.embedding = nn.Embedding(v_size, e_dim)
        self.rnn = nn.GRU(e_dim, h_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(h_dim, v_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class PersonaBindingDataset(Dataset):
    def __init__(self, words, w2i, methods, scenarios, seq_len, word_sources):
        """
        word_sources: list where 0=file_data, 1=hf_qa_data
        """
        self.samples = []
        for i in range(len(words) - seq_len):
            ctx, target = words[i:i+seq_len], words[i+seq_len]
            # 3.5x Weight boost for Method + Question Type (Scenario) co-occurrence
            is_bound = any(t in methods for t in ctx+[target]) and any(t in scenarios for t in ctx+[target])
            weight = 3.5 if is_bound else 1.0
            
            # Track source of this sequence
            source = word_sources[i+seq_len]  # 0=file, 1=hf_qa
            
            self.samples.append((
                torch.tensor([w2i[w] for w in ctx]), 
                torch.tensor(w2i[target]), 
                torch.tensor(weight),
                torch.tensor(source)
            ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 4. Aligned Axis Inhibition
# -------------------------
def apply_aligned_inhibition(x, y, vocab_size, sources):
    """
    Inhibit different axes based on data source:
    - Generic Q&A data (source=1): Inhibit sequence axis (dim 1)
    - File data (source=0): Inhibit batch axis (dim 0)
    Ensures alignment between the two data streams.
    """
    batch_size, seq_len = x.shape
    
    # Create masks for each source type
    hf_mask = (sources == 1)  # Generic Q&A data
    file_mask = (sources == 0)  # File data
    
    # For Generic Q&A: inhibit sequence dimension (horizontal axis)
    # Modify positions along sequence for Q&A samples
    for i in range(seq_len):
        if i < batch_size and hf_mask[i % batch_size]:
            idx_batch = i % batch_size
            idx_seq = i % seq_len
            if idx_seq < seq_len:
                # Inhibit by modulating with target
                x[idx_batch, idx_seq] = (y[idx_batch] + x[idx_batch, idx_seq]) % vocab_size
    
    # For File data: inhibit batch dimension (vertical axis)
    # Modify positions along batch for file samples
    for i in range(batch_size):
        if file_mask[i]:
            idx_seq = file_mask[i] % seq_len
            # Inhibit by inverse modulation with target
            y[i] = (y[i] - x[i, idx_seq]) % vocab_size
    
    return x, y

# -------------------------
# 5. Main Engine
# -------------------------
def generate_text(model, inverter, renetworker, seed, w2i, i2w, seq_len):
    model.eval(); v_size = len(i2w)
    gen_ids = [w2i.get(w, 0) for w in seed.lower().split()]
    print(f"\n>> Q&A-Bound Seed: {seed}")
    for _ in range(GEN_LEN):
        inp = torch.tensor([gen_ids[-seq_len:]], device=device)
        with torch.no_grad():
            logits = model(inp)
            clean = renetworker(logits[0])
            grounding = inverter.get_grounding(gen_ids[-1], v_size)
            probs = F.softmax(clean + grounding, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            word = i2w.get(next_id, i2w.get(0, "unknown"))
            gen_ids.append(next_id); print(word, end=' ', flush=True)

if __name__ == "__main__":
    try: 
        local_raw = open(input("Filename: "), "r", encoding="utf-8").read().lower().split()
    except: 
        local_raw = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower().split()
    
    print("Fetching generic Q&A data...")
    hf_labels, hf_qa_words = fetch_hf_data()
    
    if not hf_qa_words:
        print("WARNING: No Q&A data loaded! Using empty list.")
    
    print(f"  Question types found: {hf_labels}")
    print(f"  Total Q&A words: {len(hf_qa_words)}")
    
    # Track source of each word: 0=file, 1=hf_qa
    file_sources = [0] * len(local_raw)
    hf_sources = [1] * len(hf_qa_words)
    
    all_words = (local_raw + hf_qa_words)[:KB_len]
    word_sources = (file_sources + hf_sources)[:KB_len]
    
    vocab = sorted(list(set(all_words)))
    w2i, i2w = {w: i for i, w in enumerate(vocab)}, {i: w for i, w in enumerate(vocab)}
    
    methods = extract_methods(all_words)
    # Question types from Q&A are treated as Scenarios for tangential binding
    scenarios = set(hf_labels)
    
    inverter = CKYInverter(all_words, w2i, len(vocab))
    renetworker = InhibitoryRenetworker()
    loader = DataLoader(
        PersonaBindingDataset(all_words, w2i, methods, scenarios, SEQ_LEN, word_sources), 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    model = PersonaNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining with aligned inhibition:")
    print(f"  - File data ({len(local_raw)} words): Batch axis inhibition")
    print(f"  - Generic Q&A data ({len(hf_qa_words)} words): Sequence axis inhibition")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y, w, sources in pbar:
            x, y, w, sources = x.to(device), y.to(device), w.to(device), sources.to(device)
            
            # Apply aligned axis inhibition
            x, y = apply_aligned_inhibition(x, y, len(vocab), sources)
            
            optimizer.zero_grad()
            loss = (F.cross_entropy(model(x), y, reduction='none') * w).mean()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nModel saved to {CKPT_PATH}")
    
    while True:
        seed = input("\nSeed >> ").strip()
        if not seed: break
        generate_text(model, inverter, renetworker, seed, w2i, i2w, SEQ_LEN)

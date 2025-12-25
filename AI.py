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
import pickle

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

# Mode: 'train', 'load', or 'auto'
MODE = 'auto'  # 'auto' will load if checkpoint exists, else train

# -------------------------
# 1. Generic Q&A Data Fetching
# -------------------------
def fetch_hf_data():
    """Downloads and extracts questions and answers from HuggingFace Q&A dataset."""
    try:
        print("  Attempting to download from HuggingFace...")
        from datasets import load_dataset
        
        # Load SQuAD dataset from HuggingFace
        dataset = load_dataset('squad', split='train[:5000]', trust_remote_code=True)
        
        questions = []
        answers = []
        contexts = []
        
        for item in dataset:
            question = item['question'].lower()
            context = item['context'].lower()
            answer = item['answers']['text'][0].lower() if item['answers']['text'] else ""
            
            questions.append(question)
            answers.append(answer)
            contexts.append(context)
        
        # Extract question types
        question_types = set()
        for q in questions:
            words = q.split()
            if words:
                first_word = words[0]
                if first_word in ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']:
                    question_types.add(first_word)
        
        # Combine all text
        all_text = " ".join(questions + answers + contexts)
        all_words = all_text.split()
        
        print(f"  ✓ Downloaded SQuAD: {len(questions)} Q&A pairs, {len(all_words)} words")
        return list(question_types), all_words
        
    except ImportError:
        print("  datasets library not found, installing...")
        import subprocess
        subprocess.run(["pip", "install", "datasets", "--break-system-packages", "-q"], check=True)
        return fetch_hf_data()
        
    except Exception as e:
        print(f"  Failed to download from HuggingFace: {e}")
        print("  Falling back to built-in Q&A data...")
        
        # Fallback to embedded generic Q&A
        generic_qa = """
        what is python? python is a high-level interpreted programming language known for simplicity readability and extensive libraries used in web development data science and automation.
        who created linux? linus torvalds created the linux operating system kernel in 1991 as a free open-source alternative to proprietary unix systems.
        where is silicon valley? silicon valley is located in the san francisco bay area of california and is the global center for technology innovation and venture capital.
        when was the internet invented? the internet was invented in the late 1960s with arpanet which connected research institutions and evolved into the modern internet by the 1980s.
        why do we use databases? we use databases to efficiently store organize retrieve and manage large amounts of structured data with support for transactions and concurrent access.
        how does machine learning work? machine learning uses algorithms and statistical models to analyze patterns in data enabling computers to improve performance on tasks without explicit programming.
        what is artificial intelligence? artificial intelligence is technology that enables machines to simulate human intelligence including learning reasoning problem solving perception and language understanding.
        who invented the telephone? alexander graham bell invented the first practical telephone in 1876 revolutionizing long-distance communication and connecting people across vast distances.
        where is the eiffel tower? the eiffel tower is located in paris france on the champ de mars near the seine river and was built in 1889 for the world's fair.
        when was world war two? world war two lasted from 1939 to 1945 involving most nations of the world forming two opposing military alliances the allies and the axis.
        why is water important? water is essential for all known forms of life serving as a solvent for biochemical reactions regulating temperature and enabling cellular processes in organisms.
        how do computers process information? computers process information using binary code transistors and logical circuits executing instructions through the fetch-decode-execute cycle in the cpu.
        what is the capital of france? the capital of france is paris a major european city known for art fashion gastronomy and culture with landmarks like the louvre and notre-dame.
        who wrote hamlet? william shakespeare wrote the tragedy of hamlet prince of denmark around 1600 exploring themes of revenge madness and moral corruption.
        where do penguins live? penguins live primarily in antarctica and cold southern hemisphere regions including south america south africa australia and new zealand with adaptations for aquatic life.
        """
        
        lines = [line.strip() for line in generic_qa.strip().split('\n') if line.strip()]
        questions = []
        answers = []
        
        for line in lines:
            if '?' in line:
                parts = line.split('?', 1)
                if len(parts) == 2:
                    questions.append(parts[0].strip().lower())
                    answers.append(parts[1].strip().lower())
        
        question_types = {'what', 'who', 'where', 'when', 'why', 'how'}
        all_text = " ".join(questions + answers)
        all_words = all_text.split()
        
        print(f"  ✓ Using fallback: {len(questions)} Q&A pairs, {len(all_words)} words")
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
    - HuggingFace Q&A data (source=1): Inhibit sequence axis (dim 1)
    - File data (source=0): Inhibit batch axis (dim 0)
    Ensures alignment between the two data streams.
    """
    batch_size, seq_len = x.shape
    
    # Create masks for each source type
    hf_mask = (sources == 1)  # HuggingFace Q&A data
    file_mask = (sources == 0)  # File data
    
    # For HuggingFace Q&A: inhibit sequence dimension (horizontal axis)
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
            idx_seq = i % seq_len
            # Inhibit by inverse modulation with target
            y[i] = (y[i] - x[i, idx_seq]) % vocab_size
    
    return x, y

# -------------------------
# 5. Main Engine
# -------------------------
def save_model(model, w2i, i2w, inverter, scenarios, methods, path_prefix=""):
    """Save model, vocabulary, and auxiliary data."""
    import pickle
    
    # Save model weights
    model_path = path_prefix + CKPT_PATH
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save vocabulary and metadata
    vocab_data = {
        'w2i': w2i,
        'i2w': i2w,
        'scenarios': scenarios,
        'methods': methods,
        'vocab_size': len(i2w),
        'seq_len': SEQ_LEN,
        'embed_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS
    }
    vocab_path = path_prefix + VOCAB_PATH
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"✓ Vocabulary saved to {vocab_path}")
    
    # Save inverter matrix
    inverter_path = path_prefix + INVERTER_PATH
    torch.save(inverter.matrix, inverter_path)
    print(f"✓ Inverter matrix saved to {inverter_path}")

def load_model(path_prefix=""):
    """Load model, vocabulary, and auxiliary data."""
    import pickle
    
    # Load vocabulary and metadata
    vocab_path = path_prefix + VOCAB_PATH
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    w2i = vocab_data['w2i']
    i2w = vocab_data['i2w']
    scenarios = vocab_data['scenarios']
    methods = vocab_data['methods']
    vocab_size = vocab_data['vocab_size']
    
    print(f"✓ Loaded vocabulary: {vocab_size} words")
    
    # Load model
    model = PersonaNeuralNet(
        vocab_size,
        vocab_data['embed_dim'],
        vocab_data['hidden_dim'],
        vocab_data['num_layers']
    ).to(device)
    
    model_path = path_prefix + CKPT_PATH
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from {model_path}")
    
    # Load inverter matrix
    inverter_path = path_prefix + INVERTER_PATH
    inverter_matrix = torch.load(inverter_path, map_location=device)
    
    # Reconstruct inverter
    class LoadedInverter:
        def __init__(self, matrix):
            self.matrix = matrix
        def get_grounding(self, last_id, v_size):
            mask = torch.full((v_size,), -float('inf'), device=device)
            valid = torch.where(self.matrix[last_id] > 0)[0]
            if len(valid) > 0: mask[valid] = 0.0
            else: mask.fill_(0.0)
            return mask
    
    inverter = LoadedInverter(inverter_matrix)
    print(f"✓ Inverter loaded from {inverter_path}")
    
    return model, w2i, i2w, inverter, scenarios, methods

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
    import os
    
    # Check if we should load existing model
    model_exists = os.path.exists(CKPT_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(INVERTER_PATH)
    
    if MODE == 'load' or (MODE == 'auto' and model_exists):
        print("=" * 60)
        print("LOADING EXISTING MODEL")
        print("=" * 60)
        
        model, w2i, i2w, inverter, scenarios, methods = load_model()
        renetworker = InhibitoryRenetworker()
        
    else:
        print("=" * 60)
        print("TRAINING NEW MODEL")
        print("=" * 60)
        
        try: 
            local_raw = open(input("Filename: "), "r", encoding="utf-8").read().lower().split()
        except: 
            local_raw = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower().split()
        
        print("Downloading Q&A data from HuggingFace...")
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
        print(f"  - HuggingFace Q&A data ({len(hf_qa_words)} words): Sequence axis inhibition")
        
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

        # Save the trained model
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        save_model(model, w2i, i2w, inverter, scenarios, methods)
    
    # Interactive generation loop
    print("\n" + "=" * 60)
    print("INTERACTIVE GENERATION")
    print("=" * 60)
    print("Enter a seed phrase to generate text (empty to quit)")
    
    while True:
        seed = input("\nSeed >> ").strip()
        if not seed: break
        generate_text(model, inverter, renetworker, seed, w2i, i2w, SEQ_LEN)

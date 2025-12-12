import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import re
import os
from tqdm import tqdm  # 🆕 Progress bars!

KB_len = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
# 1. Dataset + Model definitions (unchanged)
################################

class BigramDataset(Dataset):
    def __init__(self, words, word_to_ix):
        self.word_to_ix = word_to_ix
        self.bigrams = []
        for i in range(len(words) - 1):
            self.bigrams.append(
                (self.word_to_ix.get(words[i], 0),
                 self.word_to_ix.get(words[i+1], 0))
            )
    
    def __len__(self):
        return len(self.bigrams)
    
    def __getitem__(self, idx):
        context, target = self.bigrams[idx]
        return torch.tensor([context]), torch.tensor(target)

class BigramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        embed = self.embedding(x)
        logits = self.fc_out(embed.squeeze(1))
        return logits

#########################
# 2. Save / load helpers (fixed typo in save_checkpoint)
#########################

def save_checkpoint(model, optimizer, epoch, loss,
                    word_to_ix, ix_to_word, path="bigram_model.pth"):
    ckpt = {
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
    }
    torch.save(ckpt, path)
    print(f"💾 Saved checkpoint to {path}")

def load_checkpoint(path="bigram_model.pth"):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"📂 Loaded checkpoint epoch={ckpt['epoch']} loss={ckpt['loss']:.3f}")
    return ckpt

#########################
# 3. FGSM + training loop (unchanged)
#########################

def fgsm_destruction(embeddings, epsilon, data_grad):
    return torch.clamp(embeddings + epsilon * data_grad.sign(), -2, 2)

def train_epoch(model, optimizer, criterion, loader, epsilon=0.15):
    model.train()
    total_loss, batches = 0.0, 0
    
    # 🆕 TQDM Progress bar for batches!
    progress_bar = tqdm(loader, desc=f"Epoch ε={epsilon:.2f}", leave=False)
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss_clean = criterion(output, target)
        
        embeddings = model.embedding(data).detach().requires_grad_(True)
        temp_logits = model.fc_out(embeddings.squeeze(1))
        loss_temp = criterion(temp_logits, target)
        loss_temp.backward()
        
        destroyed = fgsm_destruction(embeddings, epsilon, embeddings.grad)
        destroyed_logits = model.fc_out(destroyed.squeeze(1))
        loss_destroy = criterion(destroyed_logits, target)
        
        total_loss_batch = 0.7 * loss_clean + 0.3 * loss_destroy
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        batches += 1
        
        # 🆕 Update progress bar with loss
        progress_bar.set_postfix({"loss": f"{total_loss_batch.item():.3f}"})
    
    return total_loss / batches

def generate(model, word_to_ix, ix_to_word, seed_word, length=50, temp=0.8):
    model.eval()
    generated = [seed_word]
    context = torch.tensor([[word_to_ix.get(seed_word, 0)]], device=device)
    
    with torch.no_grad():
        for _ in range(length):
            logits = model(context)
            probs = F.softmax(logits[0] / temp, dim=-1)
            next_ix = torch.multinomial(probs, 1).item()
            generated.append(ix_to_word[next_ix])
            context = torch.tensor([[next_ix]], device=device)
    
    return " ".join(generated)

#########################
# 4. Main (with epoch progress bar)
#########################

if __name__ == "__main__":
    ckpt = load_checkpoint("bigram_model.pth")
    
    if ckpt is not None:
        word_to_ix = ckpt["word_to_ix"]
        ix_to_word = ckpt["ix_to_word"]
        vocab_size = len(word_to_ix)
        
        model = BigramModel(vocab_size).to(device)
        model.load_state_dict(ckpt["model_state"])
        
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        optimizer.load_state_dict(ckpt["optim_state"])
        criterion = nn.CrossEntropyLoss()
        
        start_epoch = ckpt["epoch"]
        print("✅ Model fully loaded from checkpoint; skipping re-init.")
    
    else:
        try:
            filename = input("Filename: ")
            with open(filename, "r", encoding="utf-8") as f:
                words = re.findall(r"\b\w+\b", f.read().lower())[:KB_len]
            print(f"✅ Loaded '{filename}'")
        except FileNotFoundError:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url).text
            words = re.findall(r"\b\w+\b", text.lower())[:KB_len]
            print("✅ Loaded Shakespeare (fallback)")
        
        word_to_ix = {w: i for i, w in enumerate(set(words))}
        ix_to_word = {i: w for w, i in word_to_ix.items()}
        vocab_size = len(word_to_ix)
        print(f"Vocab: {vocab_size}, Words: {len(words)}")
        
        dataset = BigramDataset(words, word_to_ix)
        train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
        print(f"Bigram pairs: {len(dataset)}")
        
        model = BigramModel(vocab_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        
        print("🚀 BIGRAM ANTI-RETROFITTING TRAINING")
        num_epochs = 15
        epsilon = 0.1
        
        # 🆕 TQDM for epochs!
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
            loss = train_epoch(model, optimizer, criterion, train_loader, epsilon)
            print(f"Epoch {epoch:2d}: avg_loss={loss:.3f}")
            
            if epoch % 4 == 0:
                epsilon = min(0.3, epsilon * 1.15)
                sample = generate(model, word_to_ix, ix_to_word, "the", 20)
                print(f"  ↑ ε={epsilon:.3f} | Sample: {sample}")
        
        save_checkpoint(model, optimizer, num_epochs, loss,
                        word_to_ix, ix_to_word, "bigram_model.pth")
    
    # Interactive generation
    print("\n🎯 Interactive mode:")
    while True:
        try:
            cmd = input("WORD: ").strip()
            if not cmd: continue
            out = generate(model, word_to_ix, ix_to_word, cmd, length=100, temp=0.8)
            print(f"CONTINUATION: {out}\n")
        except KeyboardInterrupt:
            print("\nExiting!")
            break

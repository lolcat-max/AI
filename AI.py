import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import re
KB_len = 99999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. TEXT FILE INPUT - reads from local 'input.txt'
try:
    with open(input("Filename: "), 'r', encoding='utf-8') as f:
        words = re.findall(r'\b\w+\b', f.read().lower())[:KB_len]
    print("✅ Loaded local file")
except FileNotFoundError:
    # 1. BIGRAM Data
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    text = requests.get(url).text
    words = re.findall(r'\b\w+\b', text.lower())[:KB_len]


word_to_ix = {word: i for i, word in enumerate(set(words))}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(word_to_ix)

print(f"Vocab: {vocab_size}, Words: {len(words)}")

# 2. FIXED BIGRAM Dataset - 1D targets
class BigramDataset(Dataset):
    def __init__(self, words):
        self.bigrams = []
        for i in range(len(words)-1):
            self.bigrams.append((word_to_ix.get(words[i], 0), word_to_ix.get(words[i+1], 0)))
    
    def __len__(self): return len(self.bigrams)
    def __getitem__(self, idx):
        context, target = self.bigrams[idx]
        return torch.tensor([context]), torch.tensor(target)  # ✅ 1D target (not [target])

dataset = BigramDataset(words)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
print(f"Bigram pairs: {len(dataset)}")

# 3. BIGRAM Model
class BigramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        embed = self.embedding(x)  # [B, 1, D]
        logits = self.fc_out(embed.squeeze(1))  # [B, Vocab]
        return logits

model = BigramModel(vocab_size).to(device)

# 4. FGSM DESTRUCTION
def fgsm_destruction(embeddings, epsilon, data_grad):
    return torch.clamp(embeddings + epsilon * data_grad.sign(), -2, 2)

# 5. FIXED Training - proper shapes
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, epsilon=0.15):
    model.train()
    total_loss, batches = 0.0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)  # target is now 1D [B]
        optimizer.zero_grad()
        
        # Clean prediction [B, Vocab] vs target [B]
        output = model(data)
        loss_clean = criterion(output, target)
        
        # Get gradients for destruction
        embeddings = model.embedding(data).detach().requires_grad_(True)
        temp_logits = model.fc_out(embeddings.squeeze(1))
        loss_temp = criterion(temp_logits, target)
        loss_temp.backward()
        
        # Destroy embeddings
        destroyed_emb = fgsm_destruction(embeddings, epsilon, embeddings.grad)
        destroyed_logits = model.fc_out(destroyed_emb.squeeze(1))
        loss_destroy = criterion(destroyed_logits, target)
        
        # Combined anti-retrofitting loss
        total_loss_batch = 0.7 * loss_clean + 0.3 * loss_destroy
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        batches += 1
    
    return total_loss / batches

# 6. Generation
def generate(model, seed_word, length=50, temp=0.8):
    model.eval()
    generated = [seed_word]
    context = torch.tensor([[word_to_ix.get(seed_word, 0)]]).to(device)
    
    with torch.no_grad():
        for _ in range(length):
            logits = model(context)
            probs = F.softmax(logits[0] / temp, dim=-1)
            next_ix = torch.multinomial(probs, 1).item()
            generated.append(ix_to_word[next_ix])
            context = torch.tensor([[next_ix]]).to(device)
    
    return ' '.join(generated)

# 7. TRAINING LOOP
print("🚀 BIGRAM ANTI-RETROFITTING TRAINING")
num_epochs = 15
epsilon = 0.1

for epoch in range(1, num_epochs + 1):
    loss = train_epoch(model, train_loader, epsilon)
    print(f"Epoch {epoch:2d}: loss={loss:.3f}")
    
    if epoch % 4 == 0:
        epsilon = min(0.3, epsilon * 1.15)
        sample = generate(model, "the", length=20)
        print(f"  ↑ ε={epsilon:.3f} | Sample: {sample}")

print("\n🎯 BIGRAM TRAINING COMPLETE - Interactive!")
print("Type single words to continue (Ctrl+C to exit)")

while True:
    try:
        user_input = input("WORD: ").strip()
        if not user_input: continue
        sample = generate(model, user_input, length=100, temp=0.8)
        print(f"CONTINUATION: {sample}\n")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break

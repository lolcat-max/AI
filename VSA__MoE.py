import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, deque
import random
from typing import List, Dict

# =====================================================================
# DATASET WITH PROPER CARDINAL SHUFFLE
# =====================================================================

class CardinalShuffledTextDataset(Dataset):
    def __init__(self, corpus: List[str], vocab: Dict[str, int], 
                 seq_len: int = 32, shuffle_seed: int = 42):
        self.vocab = vocab
        self.seq_len = seq_len
        self.idx_to_word = {v: k for k, v in vocab.items()}
        
        # Tokenize sentences
        self.sentences = []
        for sent in corpus:
            tokens = [w.lower() for w in sent.split() if w.strip()]
            if len(tokens) >= 3:
                self.sentences.append(tokens)
        
        # Cardinal shuffle
        self.cardinal_indices = list(range(len(self.sentences)))
        random.seed(shuffle_seed)
        random.shuffle(self.cardinal_indices)
        self.shuffled_sentences = [self.sentences[i] for i in self.cardinal_indices]
        
        print(f"Dataset: {len(self.sentences)} sentences (shuffled)")
    
    def __len__(self):
        return len(self.shuffled_sentences)
    
    def __getitem__(self, idx):
        sentence = self.shuffled_sentences[idx]
        cardinal_idx = self.cardinal_indices[idx]
        
        token_ids = [self.vocab[w] for w in sentence if w in self.vocab]
        
        if len(token_ids) < 2:
            token_ids = [self.vocab['<pad>']] * (self.seq_len + 1)
        
        if len(token_ids) < self.seq_len + 1:
            token_ids += [self.vocab['<pad>']] * (self.seq_len + 1 - len(token_ids))
        else:
            token_ids = token_ids[:self.seq_len + 1]
        
        input_seq = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_seq = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return input_seq, target_seq, cardinal_idx

# =====================================================================
# IMPROVED MODEL WITH LAYER NORM
# =====================================================================

class CardinalRNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, 
                 dropout: float = 0.3, max_cardinal: int = 10000):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cardinal_embed = nn.Embedding(max_cardinal, 32)
        
        self.rnn = nn.LSTM(
            embed_dim + 32,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Better initialization
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, input_seq, cardinal_indices, hidden=None):
        batch_size, seq_len = input_seq.size()
        
        token_emb = self.embedding(input_seq)
        card_emb = self.cardinal_embed(cardinal_indices).unsqueeze(1).expand(-1, seq_len, -1)
        
        combined_emb = torch.cat([token_emb, card_emb], dim=-1)
        combined_emb = self.dropout(combined_emb)
        
        rnn_out, hidden = self.rnn(combined_emb, hidden)
        rnn_out = self.layer_norm(rnn_out)
        rnn_out = self.dropout(rnn_out)
        
        logits = self.fc(rnn_out)
        
        return logits, hidden

# =====================================================================
# VOCABULARY BUILDER
# =====================================================================

def build_vocabulary(corpus: List[str], min_freq: int = 2, max_vocab: int = 10000):
    word_counts = Counter()
    
    for sent in corpus:
        tokens = [w.lower() for w in sent.split() if w.strip()]
        word_counts.update(tokens)
    
    # Take most common words
    vocab_words = [w for w, c in word_counts.most_common(max_vocab) if c >= min_freq]
    
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
    for w in vocab_words:
        vocab[w] = len(vocab)
    
    return vocab, word_counts

# =====================================================================
# ADVANCED GENERATION WITH REPETITION PENALTY
# =====================================================================

def generate_text_advanced(model, vocab, idx_to_word, seed: List[str], 
                          max_len: int = 50, temperature: float = 0.9, 
                          top_p: float = 0.9, repetition_penalty: float = 1.2,
                          device='cpu'):
    """Generate with nucleus sampling and repetition penalty."""
    model.eval()
    
    # Convert seed
    token_ids = [vocab.get(w.lower(), vocab['<unk>']) for w in seed]
    generated = seed.copy()
    
    # Track recent tokens for repetition penalty
    recent_tokens = deque(maxlen=20)
    recent_tokens.extend(token_ids)
    
    cardinal_idx = torch.tensor([0], dtype=torch.long).to(device)
    hidden = None
    
    with torch.no_grad():
        for step in range(max_len):
            # Prepare input
            input_seq = torch.tensor([token_ids[-32:]], dtype=torch.long).to(device)
            
            if input_seq.size(1) < 32:
                pad_len = 32 - input_seq.size(1)
                padding = torch.full((1, pad_len), vocab['<pad>'], dtype=torch.long).to(device)
                input_seq = torch.cat([padding, input_seq], dim=1)
            
            # Forward
            logits, hidden = model(input_seq, cardinal_idx, hidden)
            next_logits = logits[0, -1, :].clone()
            
            # Apply repetition penalty
            for token_id in set(recent_tokens):
                if token_id < len(next_logits):
                    next_logits[token_id] /= repetition_penalty
            
            # Temperature
            next_logits = next_logits / temperature
            
            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens outside nucleus
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Stop conditions
            if next_token_id in [vocab['<pad>'], vocab['<eos>']]:
                break
            
            # Avoid immediate repetition
            if len(token_ids) > 0 and next_token_id == token_ids[-1] and step > 0:
                # Resample once
                next_logits[next_token_id] = -float('Inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
            
            token_ids.append(next_token_id)
            recent_tokens.append(next_token_id)
            
            word = idx_to_word.get(next_token_id, '<unk>')
            if word not in ['<pad>', '<unk>', '<eos>']:
                generated.append(word)
    
    return ' '.join(generated)

# =====================================================================
# TRAINING WITH GRADIENT CLIPPING
# =====================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, clip_norm=0.5):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for input_seq, target_seq, cardinal_idx in dataloader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        cardinal_idx = cardinal_idx.to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(input_seq, cardinal_idx)
        
        logits = logits.view(-1, model.vocab_size)
        target_seq = target_seq.view(-1)
        
        loss = criterion(logits, target_seq)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        
        mask = target_seq != 0
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    
    return total_loss / max(total_tokens, 1)

# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    print("="*80)
    print("PYTORCH RNN - FIXED REPETITION ISSUE")
    print("="*80)
    
    # Load corpus
    filename = input("Corpus filename: ")
    with open(filename, 'r', encoding='utf-8') as f:
        corpus_text = f.read()[:99999]
    
    # Clean
    corpus = [s.strip() for s in corpus_text.replace('\n', ' ').split(".") 
              if s.strip() and len(s.split()) >= 3]
    
    print(f"\n✓ Sentences: {len(corpus)}")
    
    # Build vocab
    vocab, word_counts = build_vocabulary(corpus, min_freq=1, max_vocab=5000)
    idx_to_word = {i: w for w, i in vocab.items()}
    
    print(f"✓ Vocabulary: {len(vocab)} tokens")
    print(f"✓ Top words: {[w for w, _ in word_counts.most_common(10)]}")
    
    # Dataset
    dataset = CardinalShuffledTextDataset(corpus, vocab, seq_len=32, shuffle_seed=42)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    model = CardinalRNNLanguageModel(
        vocab_size=len(vocab),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        max_cardinal=len(dataset)
    ).to(device)
    
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    print("="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_norm=0.5)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for input_seq, target_seq, cardinal_idx in val_loader:
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                cardinal_idx = cardinal_idx.to(device)
                
                logits, _ = model(input_seq, cardinal_idx)
                logits = logits.view(-1, model.vocab_size)
                target_flat = target_seq.view(-1)
                
                loss = criterion(logits, target_flat)
                mask = target_flat != 0
                val_loss += loss.item() * mask.sum().item()
                val_tokens += mask.sum().item()
        
        val_loss = val_loss / max(val_tokens, 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f"Epoch {epoch+1:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        scheduler.step()
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Generate
    print("\n" + "="*80)
    print("GENERATION (with repetition penalty & nucleus sampling)")
    print("="*80)
    
    while True:
        user_input = input("\nSeed (or 'quit'): ")
        if user_input.lower() == 'quit':
            break
        
        seed = user_input.strip().split() if user_input.strip() else ['the']
        
        generated = generate_text_advanced(
            model, vocab, idx_to_word, seed,
            max_len=60, 
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.3,
            device=device
        )
        
        print(f"\n✨ {generated}\n")

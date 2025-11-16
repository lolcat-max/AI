import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FoldingIsomorphismEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, fold_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.fold_proj = nn.Linear(embed_dim, fold_dim)

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        embed = self.embed_norm(embed)
        fold_embed = self.fold_proj(embed)
        similarity = torch.matmul(fold_embed, fold_embed.transpose(-1, -2))
        similarity = torch.clamp(similarity, min=-10.0, max=10.0)
        return embed, fold_embed, similarity


class IsomorphicFoldLLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        fold_dim=64,
        d_model=256,
        nhead=8,
        num_layers=2,
        max_seq_len=8192,
    ):
        super().__init__()
        self.fold_embedder = FoldingIsomorphismEmbedding(
            vocab_size, embed_dim, fold_dim
        )
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        embed, _, _ = self.fold_embedder(input_ids)
        x = self.input_proj(embed)
        x = self.pos_enc(x)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


def build_bigrams(words):
    return [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]


def bigrams_from_text(text):
    words = text.lower().split()
    if len(words) < 2:
        return []
    return [words[i] + " " + words[i + 1] for i in range(len(words) - 1)]


def interactive_chat(model, bigram2idx, idx2bigram, device='cpu', max_len=700):
    model.eval()
    print("You can now chat with the model. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == 'exit':
            break

        user_bigrams = bigrams_from_text(user_input)
        if not user_bigrams:
            print("Please enter at least two words.")
            continue

        token_ids = []
        for bg in user_bigrams:
            if bg in bigram2idx:
                token_ids.append(bigram2idx[bg])
            else:
                print(f"Bigram '{bg}' is unknown; skipping.")
        
        if len(token_ids) == 0:
            print("No known bigrams to process.")
            continue
        
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        generated = token_ids[:]

        for _ in range(max_len):
            logits = model(input_ids)
            logits = logits[:, -1, :]
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            probs = F.softmax(logits, dim=-1)
            
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token_id)
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

        pred_bigrams = [idx2bigram.get(idx, '<UNK>') for idx in generated]
        print("Model:", " ".join(pred_bigrams))


def train_demo():
    filename = input("Enter training corpus filename: ")
    with open(filename, "r", encoding="utf-8") as f:
        corpus = f.read().lower().split()[:8190]

    bigrams = build_bigrams(corpus)
    if len(bigrams) < 10:
        print("Warning: very small corpus, training may be unstable.")
    print(f"Num tokens: {len(corpus)}; Num bigrams: {len(bigrams)}")

    vocab = sorted(set(bigrams))
    bigram2idx = {bg: i for i, bg in enumerate(vocab)}
    idx2bigram = {i: bg for bg, i in bigram2idx.items()}
    print(f"Vocab size: {len(vocab)}")

    token_ids = torch.tensor(
        [bigram2idx[bg] for bg in bigrams], dtype=torch.long
    ).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsomorphicFoldLLM(vocab_size=len(vocab)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        input_ids = token_ids[:, :-1].to(device)
        targets = token_ids[:, 1:].to(device)

        logits = model(input_ids)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        loss = criterion(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item():.6f}")

        if torch.isnan(loss) or torch.isinf(loss):
            print("Invalid loss detected, aborting training.")
            break

    interactive_chat(model, bigram2idx, idx2bigram, device=device)


if __name__ == "__main__":
    train_demo()

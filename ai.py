import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Vocab:
    def __init__(self, specials=None):
        self.token2idx = {}
        self.idx2token = {}
        self.specials = specials or []
        for t in self.specials:
            self.add_token(t)

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build(self, token_lists):
        for i, tokens in enumerate(token_lists):
            for t in tokens:
                if t in token_lists[i-2]:
                    self.add_token(t)

    def encode(self, tokens):
        unk = self.token2idx.get("<unk>")
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices):
        return [self.idx2token.get(i, "<unk>") for i in indices]

    def __len__(self):
        return len(self.token2idx)


def preprocess(text):
    text = text.lower()
    #text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def build_ngrams(tokens, n=2):
    # Build list of n-grams from tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def pad_sequences(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


def read_corpus(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f_in:
        inputs = [line.strip() for line in f_in if line.strip()]
    with open(output_path, "r", encoding="utf-8") as f_out:
        outputs = [line.strip() for line in f_out if line.strip()]
    return inputs, outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = torch.tanh(self.fc(hidden_cat)).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input)  # (batch, 1, emb_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # (batch, 1, hid_dim)
        prediction = self.fc_out(outputs.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.9):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, -1]  # Start token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


def train_model(
    model,
    src_tensor,
    trg_tensor,
    optimizer,
    criterion,
    epochs=3,
    batch_size=32,
    device="cpu",
):
    model.train()
    n_samples = src_tensor.size(0)

    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idxs = permutation[i : i + batch_size]
            batch_src = src_tensor[idxs].to(device)
            batch_trg = trg_tensor[idxs].to(device)

            optimizer.zero_grad()
            output = model(batch_src, batch_trg)
            output_dim = output.shape[-1]

            loss = criterion(
                output[:, 1:].reshape(-1, output_dim), batch_trg[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_src.size(0)

        print(f"Epoch {epoch+1} Loss: {epoch_loss / n_samples:.4f}")


def generate_sequence(
    model, src_tensor, trg_vocab, max_len=30, device="cpu", temperature=1.0
):
    model.eval()
    with torch.no_grad():
        batch_size = src_tensor.size(0)
        encoder_outputs, hidden, cell = model.encoder(src_tensor.to(device))

        inputs = torch.tensor(
            [trg_vocab.token2idx["<sos>"]] * batch_size, device=device
        )

        generated_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            output, hidden, cell = model.decoder(inputs, hidden, cell)
            logits = output / temperature
            probs = torch.softmax(logits, dim=-1)
            inputs = torch.multinomial(probs, 1).squeeze(1)

            for i, token_idx in enumerate(inputs.tolist()):
                generated_tokens[i].append(token_idx)

        decoded = [
            trg_vocab.decode(tokens) for tokens in generated_tokens
        ]

        # Cut off tokens after <eos>
        for i in range(len(decoded)):
            if "<eos>" in decoded[i]:
                idx = decoded[i].index("<eos>")
                decoded[i] = decoded[i][:idx]

        return decoded

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_file = "xaa"
    output_file = "xaa"

    raw_inputs, raw_outputs = read_corpus(input_file, output_file)

    input_tokens = [build_ngrams(preprocess(s), n=2) for s in raw_inputs]
    output_tokens = [build_ngrams(preprocess(s), n=2) for s in raw_outputs]

    input_specials = ["<pad>", "<unk>"]
    output_specials = ["<pad>", "<unk>", "<sos>", "<eos>"]

    input_vocab = Vocab(specials=input_specials)
    output_vocab = Vocab(specials=output_specials)

    input_vocab.build(input_tokens)
    output_vocab.build(output_tokens)

    input_indices = [input_vocab.encode(toks)[:16] for toks in input_tokens]   # truncate max_len=16
    output_indices = [output_vocab.encode(["<sos>"] + toks + ["<eos>"])[:18] for toks in output_tokens]  # truncate

    src_tensor = pad_sequences(input_indices, input_vocab.token2idx["<pad>"]).to(device)
    trg_tensor = pad_sequences(output_indices, output_vocab.token2idx["<pad>"]).to(device)

    # Drastically reduce embedding and hidden dims
    encoder = Encoder(len(input_vocab), emb_dim=8, hid_dim=16).to(device)
    decoder = Decoder(len(output_vocab), emb_dim=8, hid_dim=16).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Use float16 mixed precision if GPU available (optional advanced)
    if device.type == 'cuda':
        model = model.half()
        src_tensor = src_tensor.half()
        trg_tensor = trg_tensor.half()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab.token2idx["<pad>"])

    # Use very small batch size for memory
    train_model(model, src_tensor, trg_tensor, optimizer, criterion, epochs=3, batch_size=2, device=device)

    # Interactive demo
    while True:
        sentence = input("\nEnter input sentence (or type exit): ").strip()
        if sentence.lower() == "exit":
            break

        tokenized = build_ngrams(preprocess(sentence), n=2)[:16]  # truncate input length
        encoded = input_vocab.encode(tokenized)
        if len(encoded) == 0:
            print("Input contains unknown tokens.")
            continue

        src = torch.tensor([encoded], device=device)
        if device.type == 'cuda':
            src = src.half()
        outputs = generate_sequence(model, src, output_vocab, max_len=800, device=device, temperature=0.8)
        print("Generated symbolic equation:", " ".join(outputs[0]))


if __name__ == "__main__":
    main()

from collections import defaultdict
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_mora_timing(trigram):
    vowels = set('aeiouAEIOU')
    num_vowels = sum(1 for c in trigram if c in vowels)
    return num_vowels * 0.1

class HybridTrisemicNgramLLMWithMorae:
    def __init__(self, text):
        self.words = [w.strip('.,!?;:"\'') for w in re.split(r'\s+', text) if w.strip() and len(w) >= 3]
        self.char_model = defaultdict(lambda: defaultdict(int))
        self.word_model = defaultdict(lambda: defaultdict(int))
        self.timings = defaultdict(lambda: defaultdict(float))
        self.start_to_words = defaultdict(list)
        self.end_to_words = defaultdict(list)
        self._build_models()
        self._build_word_maps()

    def _build_models(self):
        full_text = ' '.join(self.words).lower()
        for i in range(len(full_text) - 3):
            prev_tri = full_text[i:i+3]
            next_tri = full_text[i+1:i+4] if i+4 <= len(full_text) else None
            if next_tri:
                self.char_model[prev_tri][next_tri] += 1
                timing = compute_mora_timing(prev_tri) + compute_mora_timing(next_tri[-1])
                self.timings[prev_tri][next_tri] += timing

        for i in range(len(self.words) - 1):
            prev_word = self.words[i].lower()
            next_word = self.words[i+1].lower()
            prev_tri = prev_word[-3:]
            next_tri = next_word[:3]
            self.word_model[prev_tri][next_tri] += 1
            timing = compute_mora_timing(prev_tri) + compute_mora_timing(next_tri)
            self.timings[prev_tri][next_tri] += timing

        for prev in self.timings:
            for next_t in list(self.timings[prev]):
                total_count = self.char_model[prev][next_t] + self.word_model[prev][next_t]
                if total_count > 0:
                    self.timings[prev][next_t] /= total_count

    def _build_word_maps(self):
        for word in self.words:
            word_lower = word.lower()
            if len(word_lower) >= 2:
                start_tri = word_lower[:3]
                end_tri = word_lower[-3:]
                self.start_to_words[start_tri].append(word_lower)
                self.end_to_words[end_tri].append(word_lower)

    def predict_next_trigram(self, current_trigram, level='hybrid'):
        current_lower = current_trigram.lower()
        if level == 'char':
            model = self.char_model
        elif level == 'word':
            model = self.word_model
        else:
            char_cands = self.char_model[current_lower]
            word_cands = self.word_model[current_lower]
            combined = defaultdict(int)
            for tri in set(list(char_cands) + list(word_cands)):
                combined[tri] += char_cands[tri] + word_cands[tri]
            if not combined:
                return {}
            weighted = {}
            for tri, count in combined.items():
                timing_weight = self.timings[current_lower].get(tri, 1.0)
                weighted[tri] = count * (1 + timing_weight)
            return dict(weighted)
        if current_lower not in model:
            return {}
        candidates = model[current_lower]
        if not candidates:
            return {}
        weighted = {}
        for tri, count in candidates.items():
            timing_weight = self.timings[current_lower].get(tri, 1.0)
            weighted[tri] = count * (1 + timing_weight)
        return dict(weighted)

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
        for tokens in token_lists:
            for t in tokens:
                self.add_token(t)  # Simplified to add all unique tokens

    def encode(self, tokens):
        unk = self.token2idx.get("<unk>", 0)
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices):
        return [self.idx2token.get(i, "<unk>") for i in indices]

    def __len__(self):
        return len(self.token2idx)


def preprocess(text):
    text = text.lower()
    return text.split()


def build_ngrams(tokens, n=3):  # Trisemic: n=3
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def pad_sequences(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences) if sequences else 1
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


def read_corpus(input_path, output_path):
    # Fallback to simulated if files missing
    try:
        with open(input_path, "r", encoding="utf-8") as f_in:
            inputs = [line.strip() for line in f_in if line.strip()]
        with open(output_path, "r", encoding="utf-8") as f_out:
            outputs = [line.strip() for line in f_out if line.strip()]
    except FileNotFoundError:
        # Simulated corpus for n-gram context
        sample_text = """
        The quick brown fox jumps over the lazy dog. Artificial intelligence machine learning natural language processing.
        Trisemic ngram model with mora timings for word generation. Predict next trigram using hybrid seq2seq approach.
        """ * 50
        ngram_model = HybridTrisemicNgramLLMWithMorae(sample_text)
        inputs = ["the quick brown fox"] * 20  # Sample inputs
        outputs = ["fox jumps lazy dog"] * 20  # Sample outputs
    return inputs, outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=8, hid_dim=16):  # Reduced for efficiency
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
    def __init__(self, vocab_size, emb_dim=8, hid_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(outputs.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, ngram_model=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.ngram_model = ngram_model  # Integrate n-gram for mora weighting

    def forward(self, src, trg, teacher_forcing_ratio=0.9):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]  # Start with <sos> at index 0

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

    def mora_weight_logits(self, logits, predicted_tokens, trg_vocab):
        if self.ngram_model is None:
            return logits
        # Approximate mora weighting: for predicted trigram tokens, scale logits by mora timing
        weighted_logits = logits.clone()
        batch_size = logits.size(0)
        for b in range(batch_size):
            pred_token = trg_vocab.decode([predicted_tokens[b]])[0]
            if ' ' in pred_token:  # It's a trigram like "the qui"
                trigram = pred_token.replace(' ', '').lower()[:3]  # Extract trigram
                mora = compute_mora_timing(trigram)
                weighted_logits[b] *= (1 + mora)  # Boost by mora duration
        return weighted_logits


def train_model(
    model,
    src_tensor,
    trg_tensor,
    optimizer,
    criterion,
    epochs=3,
    batch_size=2,  # Small for efficiency
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
    model, src_tensor, trg_vocab, max_len=800, device="cpu", temperature=0.8
):
    model.eval()
    with torch.no_grad():
        batch_size = src_tensor.size(0)
        encoder_outputs, hidden, cell = model.encoder(src_tensor.to(device))

        inputs = torch.tensor([trg_vocab.token2idx["<sos>"]] * batch_size, device=device, dtype=torch.long)

        generated_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            output, hidden, cell = model.decoder(inputs, hidden, cell)
            # Apply mora weighting to logits
            predicted = output.argmax(dim=-1)
            weighted_output = model.mora_weight_logits(output, predicted, trg_vocab)
            logits = weighted_output / temperature
            probs = F.softmax(logits, dim=-1)
            inputs = torch.multinomial(probs, 1).squeeze(1)

            for i, token_idx in enumerate(inputs.tolist()):
                generated_tokens[i].append(token_idx)
                if token_idx == trg_vocab.token2idx["<eos>"]:
                    break  # Early stop per sequence

        decoded = [trg_vocab.decode(tokens) for tokens in generated_tokens]

        for i in range(len(decoded)):
            if "<eos>" in decoded[i]:
                idx = decoded[i].index("<eos>")
                decoded[i] = decoded[i][:idx]

        return decoded


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_file = "xaa"  # Replace with actual path
    output_file = "xaa"

    raw_inputs, raw_outputs = read_corpus(input_file, output_file)

    # Preprocess with trisemic n-grams
    sample_text = ' '.join(raw_inputs + raw_outputs)
    ngram_model = HybridTrisemicNgramLLMWithMorae(sample_text)

    input_tokens = [build_ngrams(preprocess(s), n=3) for s in raw_inputs]
    output_tokens = [build_ngrams(preprocess(s), n=3) for s in raw_outputs]

    input_specials = ["<pad>", "<unk>"]
    output_specials = ["<pad>", "<unk>", "<sos>", "<eos>"]

    input_vocab = Vocab(specials=input_specials)
    output_vocab = Vocab(specials=output_specials)

    input_vocab.build(input_tokens)
    output_vocab.build(output_tokens)

    input_indices = [input_vocab.encode(toks)[:2] for toks in input_tokens if toks]
    output_indices = [output_vocab.encode(["<sos>"] + toks + ["<eos>"])[:2] for toks in output_tokens if toks]

    if not input_indices or not output_indices:
        print("No valid data after preprocessing. Use larger corpus.")
        return

    src_tensor = pad_sequences(input_indices, input_vocab.token2idx["<pad>"]).to(device)
    trg_tensor = pad_sequences(output_indices, output_vocab.token2idx["<pad>"]).to(device)

    encoder = Encoder(len(input_vocab), emb_dim=8, hid_dim=16).to(device)
    decoder = Decoder(len(output_vocab), emb_dim=8, hid_dim=16).to(device)
    model = Seq2Seq(encoder, decoder, device, ngram_model=ngram_model).to(device)

    if device.type == 'cuda':
        model = model.half()
        src_tensor = src_tensor.half()
        trg_tensor = trg_tensor.half()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=output_vocab.token2idx["<pad>"])

    train_model(model, src_tensor, trg_tensor, optimizer, criterion, epochs=3, batch_size=2, device=device)

    # Interactive demo
    while True:
        sentence = input("\nEnter input sentence (or type exit): ").strip()
        if sentence.lower() == "exit":
            break

        tokenized = build_ngrams(preprocess(sentence), n=3)[:2]
        if not tokenized:
            print("Invalid input.")
            continue
        encoded = input_vocab.encode(tokenized)
        src = torch.tensor([encoded], device=device, dtype=torch.long)
        if device.type == 'cuda':
            src = src.half()
        outputs = generate_sequence(model, src, output_vocab, max_len=800, device=device, temperature=10000.01)
        print("Generated sequence:", " ".join(outputs[0]))


if __name__ == "__main__":
    main()

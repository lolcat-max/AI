import torch
import torch.nn as nn
import torch.optim as optim
KB_LEN = 999
# Example CBOW model definition (same as before)
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        combined = embeds.mean(dim=0).view(1, -1)
        out = self.linear1(combined)
        out = self.activation(out)
        out = self.linear2(out)
        return out

# Setup vocabulary and mapping (initially empty)
vocab = set()
word_to_ix = {}
embedding_dim = 50
model = None
optimizer = None
loss_function = nn.CrossEntropyLoss()

def update_vocab_and_model(new_words):
    global vocab, word_to_ix, model, optimizer
    new_vocab = set(new_words) - vocab
    if new_vocab:
        vocab.update(new_vocab)
        word_to_ix = {w:i for i,w in enumerate(sorted(vocab))}
        vocab_size = len(vocab)
        old_state = None
        if model is not None:
            old_state = model.state_dict()
        model = CBOW(vocab_size, embedding_dim)
        if old_state:
            # Attempt to transfer old weights - simple approach (sizes may differ)
            for name, param in model.named_parameters():
                if name in old_state and old_state[name].size() == param.size():
                    param.data.copy_(old_state[name])
        optimizer = optim.Adam(model.parameters(), lr=0.005)

def generate_training_data(tokens, context_window=2):
    data = []
    for i in range(context_window, len(tokens) - context_window):
        context = tokens[i - context_window:i] + tokens[i+1:i+1+context_window]
        target = tokens[i]
        if all(word in word_to_ix for word in context) and target in word_to_ix:
            data.append((context, target))
    return data

def train_on_stream(text_stream, epochs=3):
    for chunk in text_stream:
        tokens = chunk.lower().split()
        update_vocab_and_model(tokens)
        data = generate_training_data(tokens)
        if not data:
            continue
        for epoch in range(epochs):
            total_loss = 0
            for context, target in data:
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
                target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)
                optimizer.zero_grad()
                output = model(context_idxs)
                loss = loss_function(output, target_idx)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Chunk training epoch loss: {total_loss:.4f}")

# Simulated text stream chunks
filename = input("Filename: ")
with open(filename, 'r', encoding='utf-8') as f:
    text_chunks = f.read().split(".")[:KB_LEN]

# Train model on the streaming data chunks
train_on_stream(text_chunks)

# Example function to get similar words after training on streams
def get_similar_words(word, top_n=6):
    if word not in word_to_ix:
        return []
    embeddings = model.embeddings.weight.data
    word_vec = embeddings[word_to_ix[word]].unsqueeze(0)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarities = cos(word_vec, embeddings)
    values, indices = torch.topk(similarities, top_n + 1)
    neighbors = [list(word_to_ix.keys())[idx] for idx in indices[1:]]
    return neighbors

while True:
    word = input("USER: ")
    print(f"Words related to {word}:", get_similar_words(word))

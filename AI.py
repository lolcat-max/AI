import numpy as np
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox

# --- 1. Data preparation (abbreviated for brevity) ---
SPECIAL_TOKENS = ['<PAD>', '<UNK>']
MIN_FREQUENCY = 2
filename = input("Enter filename: ")
with open(filename, 'r', encoding='utf-8') as f:
    sentences = f.read().split(".")[:20]
all_words = " ".join(sentences).split()
word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count >= MIN_FREQUENCY]
unique_words = sorted(list(set(filtered_words)))
vocab = SPECIAL_TOKENS + unique_words
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
PAD_IDX = word_to_idx['<PAD>']
UNK_IDX = word_to_idx['<UNK>']

# Model params
s1_size = 50
c1_size = s1_size // 2
s2_size = 25
c2_size = max(1, s2_size // 2)
output_seq_len = 100  # length of predicted output sequence

# Initialize weights and biases
W_s1 = np.random.randn(s1_size, vocab_size) * 0.01
b_s1 = np.zeros((s1_size, 1))
W_s2 = np.random.randn(s2_size, c1_size) * 0.01
b_s2 = np.zeros((s2_size, 1))
W_out = np.random.randn(vocab_size * output_seq_len, c2_size) * 0.01
b_out = np.zeros((vocab_size * output_seq_len, 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def save_model(filename):
    np.savez(filename,
             W_s1=W_s1, b_s1=b_s1,
             W_s2=W_s2, b_s2=b_s2,
             W_out=W_out, b_out=b_out)
    print(f"Model saved to {filename}")

def load_model(filename):
    global W_s1, b_s1, W_s2, b_s2, W_out, b_out
    data = np.load(filename)
    W_s1, b_s1 = data['W_s1'], data['b_s1']
    W_s2, b_s2 = data['W_s2'], data['b_s2']
    W_out, b_out = data['W_out'], data['b_out']
    print(f"Model loaded from {filename}")

def predict_with_painted_s1(input_word, painted_s1):
    input_vector = np.zeros((vocab_size,1))
    input_vector[word_to_idx.get(input_word, UNK_IDX)] = 1
    s1_out = painted_s1.reshape(-1,1)
    c1_out = np.array([np.max(s1_out[j*2:(j+1)*2]) for j in range(c1_size)]).reshape(-1,1)
    s2_out = np.tanh(W_s2 @ c1_out + b_s2)
    c2_out = np.array([np.max(s2_out[j*2:(j+1)*2]) for j in range(c2_size)]).reshape(-1,1)
    output = W_out @ c2_out + b_out
    y_pred = softmax(output)

    predicted_words = []
    for idx in range(0, len(y_pred), vocab_size):
        probs = y_pred[idx:idx + vocab_size].flatten()
        word_idx = np.argmax(probs)
        predicted_words.append(idx_to_word[word_idx])
    return predicted_words

class PaintAndType(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neuron Paint + Seed Word Input + Save/Load + Word Wrap")
        self.geometry("450x750")
        self.grid_w = 10
        self.grid_h = 5
        self.cell_size = 30

        self.canvas_w = self.grid_w * self.cell_size
        self.canvas_h = self.grid_h * self.cell_size
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(pady=10)

        self.activations = np.ones((self.grid_h, self.grid_w)) * -1
        self.rects = np.empty((self.grid_h, self.grid_w), dtype=object)
        self.draw_grid()

        self.canvas.bind("<B1-Motion>", self.paint_on)
        self.canvas.bind("<B3-Motion>", self.paint_off)
        self.canvas.bind("<Button-1>", self.paint_on)
        self.canvas.bind("<Button-3>", self.paint_off)

        self.entry_label = tk.Label(self, text="Enter Seed Word:")
        self.entry_label.pack()
        self.seed_entry = tk.Entry(self, font=("Arial", 14))
        self.seed_entry.pack(ipadx=5, ipady=5)

        self.predict_button = tk.Button(self, text="Predict Succeeding Words", command=self.predict)
        self.predict_button.pack(pady=10)

        self.save_button = tk.Button(self, text="Save Model", command=self.save_model_dialog)
        self.save_button.pack()

        self.load_button = tk.Button(self, text="Load Model", command=self.load_model_dialog)
        self.load_button.pack(pady=5)

        # Changed to Message widget for word wrap
        self.prediction_message = tk.Message(self, text="", width=400, font=("Arial", 12), fg="green", bg="white")
        self.prediction_message.pack(pady=10, fill='x')

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = self.activation_to_color(self.activations[i, j])
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
                self.rects[i, j] = rect

    def activation_to_color(self, val):
        norm = (np.tanh(val) + 1) / 2
        r = int(norm * 255)
        b = 255 - r
        return f"#{r:02x}00{b:02x}"

    def paint_on(self, event):
        self.paint(event, +1)

    def paint_off(self, event):
        self.paint(event, -1)

    def paint(self, event, val):
        j = event.x // self.cell_size
        i = event.y // self.cell_size
        if 0 <= i < self.grid_h and 0 <= j < self.grid_w:
            self.activations[i, j] = val
            self.canvas.itemconfig(self.rects[i, j], fill=self.activation_to_color(val))

    def predict(self):
        seed = self.seed_entry.get().strip()
        if not seed:
            self.prediction_message.config(text="Please enter a seed word to predict.")
            return
        painted_s1_flat = self.activations.flatten()
        if painted_s1_flat.size < s1_size:
            padded = np.ones(s1_size) * -1
            padded[:painted_s1_flat.size] = painted_s1_flat
            painted_s1_flat = padded
        else:
            painted_s1_flat = painted_s1_flat[:s1_size]
        preds = predict_with_painted_s1(seed, painted_s1_flat)
        preds = [w for w in preds if w not in SPECIAL_TOKENS]
        self.prediction_message.config(text="Prediction: " + ' '.join([seed] + preds))

    def save_model_dialog(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy Zip", "*.npz")])
        if filepath:
            save_model(filepath)
            messagebox.showinfo("Save Model", f"Model saved to:\n{filepath}")

    def load_model_dialog(self):
        filepath = filedialog.askopenfilename(filetypes=[("NumPy Zip", "*.npz")])
        if filepath:
            load_model(filepath)
            messagebox.showinfo("Load Model", f"Model loaded from:\n{filepath}")

if __name__ == "__main__":
    app = PaintAndType()
    app.mainloop()

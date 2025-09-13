import numpy as np
from collections import Counter
import tkinter as tk
from tkinter import filedialog, scrolledtext

# --- Setup vocab and model params ---
SPECIAL_TOKENS = ['<PAD>', '<UNK>']
MIN_FREQUENCY = 2

filename = input("Enter filename: ")
with open(filename, 'r', encoding='utf-8') as f:
    sentences = f.read().split(".")

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

# Model sizes
s1_size = 50
c1_size = s1_size // 2
s2_size = 25
c2_size = max(1, s2_size // 2)

# Toy model weights (random, fixed seed for reproducibility)
np.random.seed(1)
W_s2 = np.random.randn(s2_size, c1_size) * 0.01
b_s2 = np.zeros((s2_size, 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- Prediction function with painted S1 and adjustable output length ---
def predict_with_painted_s1(input_word, painted_s1, output_len=25):
    input_vector = np.zeros((vocab_size, 1))
    input_vector[word_to_idx.get(input_word, UNK_IDX)] = 1

    s1_out = painted_s1.reshape(-1, 1)
    c1_out = np.array([np.max(s1_out[j*2:(j+1)*2]) for j in range(c1_size)]).reshape(-1, 1)
    s2_out = np.tanh(W_s2 @ c1_out + b_s2)
    c2_out = np.array([np.max(s2_out[j*2:(j+1)*2]) for j in range(c2_size)]).reshape(-1, 1)

    # Build dynamic output layer for requested length
    W_out_dyn = np.random.randn(vocab_size * output_len, c2_size) * 0.01
    b_out_dyn = np.zeros((vocab_size * output_len, 1))

    output = W_out_dyn @ c2_out + b_out_dyn
    y_pred = softmax(output)

    predicted_words = []
    for idx in range(0, len(y_pred), vocab_size):
        probs = y_pred[idx:idx + vocab_size].flatten()
        probs = probs / probs.sum()
        word_idx = np.random.choice(len(probs), p=probs)
        predicted_words.append(idx_to_word[word_idx])
    return predicted_words

# --- Tkinter UI ---
class PaintAndType(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neuron Paint + Seed Word Input")
        self.geometry("900x700")

        # Grid settings
        self.grid_w = 30
        self.grid_h = 30
        self.cell_size = 15

        # Canvas for painting neurons
        self.canvas_w = self.grid_w * self.cell_size
        self.canvas_h = self.grid_h * self.cell_size
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack()

        # Initialize activations
        self.activations = np.ones((self.grid_h, self.grid_w)) * -1
        self.rects = np.empty((self.grid_h, self.grid_w), dtype=object)
        self.draw_grid()

        # Paint bindings
        self.canvas.bind("<B1-Motion>", self.paint_on)
        self.canvas.bind("<B3-Motion>", self.paint_off)
        self.canvas.bind("<Button-1>", self.paint_on)
        self.canvas.bind("<Button-3>", self.paint_off)

        # Controls
        controls_frame = tk.Frame(self)
        controls_frame.pack(pady=10)

        tk.Label(controls_frame, text="Seed Word:").grid(row=0, column=0, padx=5)
        self.seed_entry = tk.Entry(controls_frame, font=("Arial", 14))
        self.seed_entry.grid(row=0, column=1, padx=5)

        tk.Label(controls_frame, text="Output Length:").grid(row=0, column=2, padx=5)
        self.output_length_entry = tk.Entry(controls_frame, font=("Arial", 14), width=5)
        self.output_length_entry.insert(0, "25")
        self.output_length_entry.grid(row=0, column=3, padx=5)

        self.predict_button = tk.Button(controls_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=0, column=4, padx=10)

        self.save_button = tk.Button(controls_frame, text="Save Activations", command=self.save_model)
        self.save_button.grid(row=0, column=5, padx=5)

        self.load_button = tk.Button(controls_frame, text="Load Activations", command=self.load_model)
        self.load_button.grid(row=0, column=6, padx=5)

        # Scrollable text box for predictions
        self.prediction_label = scrolledtext.ScrolledText(self, wrap="word",
                                                          font=("Arial", 12),
                                                          fg="green", bg="white",
                                                          height=15, width=100)
        self.prediction_label.pack(pady=10, fill="both", expand=True)

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
            self.prediction_label.delete("1.0", "end")
            self.prediction_label.insert("end", "Please enter a seed word to predict.")
            return

        try:
            out_len = int(self.output_length_entry.get().strip())
            if out_len <= 0:
                out_len = 25
        except:
            out_len = 25

        painted_s1_flat = self.activations.flatten()
        if painted_s1_flat.size < s1_size:
            padded = np.ones(s1_size) * -1
            padded[:painted_s1_flat.size] = painted_s1_flat
            painted_s1_flat = padded
        else:
            painted_s1_flat = painted_s1_flat[:s1_size]

        preds = predict_with_painted_s1(seed, painted_s1_flat, output_len=out_len)
        preds = [w for w in preds if w not in SPECIAL_TOKENS]

        self.prediction_label.delete("1.0", "end")
        self.prediction_label.insert("end", "Prediction:\n" + ' '.join([seed] + preds))

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
        if file_path:
            np.save(file_path, self.activations)

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.activations = np.load(file_path)
            self.draw_grid()

if __name__ == "__main__":
    app = PaintAndType()
    app.mainloop()

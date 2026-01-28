#!/usr/bin/env python3
# NEUROSYNTHETIC V3.7 — TRIGRAM + CAUCHY SEQUENCES

import os, re, time, math
import torch
import torch.nn.functional as F
import gradio as gr
from dataclasses import dataclass, field
from collections import deque
from typing import List

# =========================================================
# 🔥 RYZEN KAKUTANI RNG
# =========================================================

@dataclass
class RyzenKakutani:
    head: int = 0
    fixed_point: float = 0.5
    history: deque = field(default_factory=lambda: deque(maxlen=64))

    def tick(self):
        r = int.from_bytes(os.urandom(4), "little")
        phase = (self.head * 0.13 + r / 2**36) % (2 * math.pi)
        self.head = (self.head + 1) & 127

        x = abs(math.sin(phase))
        lo, hi = x * 0.7, 0.3 + x * 0.9

        self.fixed_point += ((lo + hi) / 2 - self.fixed_point) * 0.12
        fixed = abs(x - self.fixed_point) < 0.045

        self.history.append(int(fixed))
        return sum(self.history) / len(self.history), (r >> 8) & 0xFF, x


rk = RyzenKakutani()

# =========================================================
# 📐 CAUCHY SEQUENCE TRACKER
# =========================================================

@dataclass
class CauchyTracker:
    eps: float = 0.002
    window: int = 8
    seq: deque = field(default_factory=lambda: deque(maxlen=32))

    def update(self, value: float) -> float:
        self.seq.append(value)

        if len(self.seq) < self.window:
            return 0.0

        diffs = [
            abs(self.seq[i] - self.seq[i-1])
            for i in range(-1, -self.window, -1)
        ]

        # convergence score ∈ [0,1]
        score = sum(d < self.eps for d in diffs) / len(diffs)
        return score


cauchy = CauchyTracker()

# =========================================================
# 🧠 TEXT UTILS
# =========================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+|[.,!?]", text.lower())

# =========================================================
# 📊 TRIGRAM LANGUAGE MODEL
# =========================================================

class TrigramLM:
    def __init__(self):
        self.reset()

    def reset(self):
        self.uni = {}
        self.tri = {}
        self.total = 0

    def ingest(self, toks: List[str]):
        for t in toks:
            self.uni[t] = self.uni.get(t, 0) + 1
            self.total += 1
        for i in range(len(toks) - 2):
            k = (toks[i], toks[i+1], toks[i+2])
            self.tri[k] = self.tri.get(k, 0) + 1

    def dist(self, w1, w2):
        cand = [c for (a,b,c) in self.tri if (a,b) == (w1,w2)]
        if not cand:
            cand = sorted(self.uni, key=self.uni.get, reverse=True)[:128]

        probs = torch.tensor(
            [self.uni.get(w, 1) / self.total for w in cand],
            dtype=torch.float32
        )
        probs /= probs.sum()
        return cand, probs

# =========================================================
# 🐍 NEUROSYNTHETIC GENERATOR
# =========================================================

class NeuroRyzen:
    def __init__(self):
        self.lm = TrigramLM()
        self.ready = False
        self.cauchy_x = 0.5

    def ingest_text(self, text: str):
        self.lm.reset()
        toks = tokenize(normalize(text))
        if len(toks) < 10:
            toks += ["the", "system", "is", "booting"]
        self.lm.ingest(toks)
        self.ready = True
        return f"📄 Dataset loaded ({len(toks)} tokens)"

    def step(self, context: str, steer: float):
        if not self.ready:
            self.ingest_text(context)

        toks = tokenize(context)
        if len(toks) < 2:
            toks += ["the", "is"]

        w1, w2 = toks[-2:]

        fixed, mapv, x = rk.tick()

        # 🔁 Update Cauchy sequence
        self.cauchy_x = 0.9 * self.cauchy_x + 0.1 * x
        c_score = cauchy.update(self.cauchy_x)

        cand, base = self.lm.dist(w1, w2)

        # 🎯 Bias modulation
        bias = (
            fixed * steer
            + (mapv / 255) * 0.4
            + c_score * 1.5
        )

        probs = F.softmax(base + bias, dim=0)

        # 🐍 SnakeFold strengthened by convergence
        if c_score > 0.6:
            mid = len(probs) // 2
            probs = torch.cat([probs[:mid], torch.flip(probs[mid:], [0])])

        nxt = cand[torch.multinomial(probs, 1).item()]
        status = (
            f"K:{fixed:.0%} "
            f"C:{c_score:.2f} "
            f"MAP:0x{mapv:02X}"
        )

        return nxt, status

gen = NeuroRyzen()

# =========================================================
# 📄 DATASET UPLOAD
# =========================================================

def load_dataset(file):
    if file is None:
        return "⚠️ No file uploaded"
    with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
        return gen.ingest_text(f.read())

# =========================================================
# 🔁 REALTIME STREAM
# =========================================================

def stream(context, steer, steps, delay):
    text = context
    out = []
    
    while True:
        nxt, status = gen.step(text, steer)
        text += " " + nxt
        out.append(f"{nxt}")
        yield "\n".join(out[-20:]), status, text
        time.sleep(delay)

# =========================================================
# 🖥️ UI
# =========================================================

with gr.Blocks() as demo:
    gr.Markdown("# 🔥 NeuroSynthetic V3.7 — Trigram + Cauchy Sequences")

    status = gr.Textbox(label="System State")

    with gr.Row():
        dataset = gr.File(label="📄 Upload TXT Dataset", file_types=[".txt"])
        load_btn = gr.Button("Load Dataset")

    prompt = gr.Textbox(
        "the universe is a fixed point attractor",
        label="Context"
    )

    steer = gr.Slider(0.5, 3.0, 1.3, label="Steer")
    output = gr.Textbox(lines=12, label="Realtime Stream")

    steps = gr.Slider(1, 300, 100, step=1)
    delay = gr.Slider(0.0, 0.3, 0.03, step=0.01)

    run = gr.Button("▶ Run Realtime")

    load_btn.click(load_dataset, inputs=dataset, outputs=status)
    run.click(
        stream,
        inputs=[prompt, steer, steps, delay],
        outputs=[output, status, prompt]
    )

if __name__ == "__main__":
    demo.queue().launch()

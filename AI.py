import os
import requests
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------
KB_len = -1
CKPT_PATH = "zen_neural_trainer_no_img.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
SEQ_LEN = 8
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Training
BATCH_SIZE = 512
LR = 5e-3
NUM_EPOCHS = 1

EPS_START = 0.10
EPS_MAX = 0.30
EPS_GROW_EVERY = 4
EPS_GROW_MULT = 1.15

ADV_EVERY = 4
EMB_CLAMP = 2.0
GRAD_CLIP_NORM = 1.0

# -------------------------
# 1. Zen Simulation Engine (2D Physics)
# -------------------------
# -------------------------
# 1. Zen Simulation Engine (Recurrent Travel)
# -------------------------
class ZenSimulationEngine:
    """
    Runs a 2D particle simulation with Recurrent Travel (Memory-based trajectory).
    """
    def __init__(self, num_particles=200):
        self.num_particles = num_particles
        # State: [x, y, vx, vy, life]
        self.particles = np.zeros((num_particles, 5), dtype=np.float32)
        self.head_idx = 0
        self.t = 0.0
        self.entity_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        # --- Recurrence State ---
        # The 'memory' of travel. Stores accumulated momentum/bias.
        self.recurrence_vec = np.array([0.0, 0.0], dtype=np.float32)
        self.recurrence_decay = 0.12 # How fast history fades
        self.feedback_strength = 0.95 # How much history affects current move
        
        # Default Params
        self.speed = 0.05
        self.decay = 0.95
        self.emit_rate = 2
        self.chaos = 0.02
    
    def get_zen_target(self, t):
        # The "Ideal" Path (Lemniscate)
        scale = 3.0 + np.sin(t * 0.3) * 0.5
        denom = 1 + np.sin(t)**2
        x = scale * np.cos(t) / denom
        y = scale * np.sin(t) * np.cos(t) / denom
        return np.array([x, y], dtype=np.float32)

    def step(self, control_signals=None):
        # 1. Apply Neural Modulation
        if control_signals:
            impulse, emotion, wave_mix = control_signals
            target_speed = 0.05 + (emotion * 0.05) - (impulse * 0.03)
            self.speed = 0.9 * self.speed + 0.1 * target_speed
            
            # Recurrence modulation: High emotion = stronger chaotic memory
            self.feedback_strength = 0.15 + (emotion * 0.1)
            self.recurrence_decay = 0.90 + (impulse * 0.08) # Impulse stabilizes memory
            
            self.chaos = 0.01 + (emotion * 0.05)
            self.decay = 0.90 + (wave_mix * 0.09)

        self.t += self.speed
        
        # 2. Calculate Recurrent Movement
        # A. Where we "should" be based on the Zen formula
        target_pos = self.get_zen_target(self.t)
        
        # B. Calculate the pull towards the target
        ideal_velocity = (target_pos - self.entity_pos) * 0.1
        
        # C. Update Recurrence Vector (The Memory)
        # New Recurrence = (Old Recurrence * Decay) + (Current Ideal Velocity)
        self.recurrence_vec = (self.recurrence_vec * self.recurrence_decay) + ideal_velocity
        
        # D. Apply Recurrence to Position
        # Movement is a blend of immediate target pull AND historical momentum
        # This creates overshoots, swirls, and "orbiting" behavior around the path
        prev_pos = self.entity_pos.copy()
        
        step_move = (ideal_velocity * (1.0 - self.feedback_strength)) + \
                    (self.recurrence_vec * self.feedback_strength * 2.0)
        
        self.entity_pos += step_move
        
        # Calculate actual resulting velocity for particles
        actual_velocity = self.entity_pos - prev_pos
        
        # 3. Emit Particles with Recurrent Curl
        # Particles inherit the *recurrence vector* as well, creating curved trails
        for _ in range(int(self.emit_rate)):
            idx = self.head_idx
            self.particles[idx, 0:2] = self.entity_pos
            
            drift = np.random.normal(0, self.chaos, 2)
            
            # Blend actual velocity with the hidden recurrence field
            # This makes particles curl even after leaving the emitter
            particle_vel = (actual_velocity * 0.4) + (self.recurrence_vec * 10.9) + drift
            
            self.particles[idx, 2:4] = particle_vel
            self.particles[idx, 4] = 1.0
            self.head_idx = (self.head_idx + 1) % self.num_particles
            
        # 4. Physics Update
        self.particles[:, 0:2] += self.particles[:, 2:4]
        self.particles[:, 2:4] *= 0.19 # Drag
        self.particles[:, 4] *= self.decay
        
        return self.get_state_tensor()

    def get_state_tensor(self):
        # We also want to feed the Recurrence Vector to the Neural Net
        # so it can "feel" the momentum.
        
        # Particle Data
        p_data = self.particles[:, :4].copy()
        p_data[:, :2] /= 4.0 
        p_data[:, 2:4] /= 0.1
        
        # Recurrence Data (Append to end or mix in)
        # For simplicity, we just use particle data here, 
        # but the particle motion now implicitly contains recurrence info.
        flat = torch.from_numpy(p_data.flatten())
        return flat.to(device)


# -------------------------
# 2. Network Components
# -------------------------
class NonCommutativeMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B
    @staticmethod
    def backward(ctx, grad_out):
        A, B = ctx.saved_tensors
        return grad_out @ B.transpose(-1, -2), A.transpose(-1, -2) @ grad_out

nc_matmul = NonCommutativeMatMul.apply

class NCLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None: nn.init.uniform_(self.bias, -0.1, 0.1)
    def forward(self, x):
        y = nc_matmul(x, self.weight.transpose(-1, -2))
        return y + self.bias if self.bias is not None else y

class SimulationSensoryCortex(nn.Module):
    def __init__(self, num_particles, output_dim=128):
        super().__init__()
        input_size = num_particles * 4 # x, y, vx, vy per particle
        self.proj = nn.Sequential(
            NCLinear(input_size, 256),
            nn.ReLU(),
            NCLinear(256, output_dim),
            nn.Tanh()
        )
    def forward(self, sim_state):
        if sim_state.dim() == 1:
            sim_state = sim_state.unsqueeze(0)
        return self.proj(sim_state)

class LogicDesignController(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_reward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.to_dist   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.corr_proj = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.impulse_head = nn.Linear(hidden_dim, 1)
        self.emotion_head = nn.Linear(hidden_dim, 1)
        self.unaware_head = nn.Linear(hidden_dim * 2 + hidden_dim + 2, 1)
        self.temp_head = nn.Linear(3, 1)   
        self.mix_head  = nn.Linear(3, 1)   

    def forward(self, input_h):
        # input_h now comes purely from Sim or Text, not Image
        reward_h = self.to_reward(input_h)
        dist_h   = self.to_dist(input_h)
        corr = torch.tanh((reward_h * dist_h).mean(dim=-1, keepdim=True))
        rd = torch.cat([reward_h, dist_h, corr], dim=-1)
        corr_h = torch.tanh(self.corr_proj(rd))
        reward_h = reward_h + 0.35 * corr_h
        dist_h   = dist_h   + 0.35 * corr_h
        
        impulse = torch.sigmoid(self.impulse_head(reward_h))
        emotion = torch.sigmoid(self.emotion_head(dist_h))
        unaware = torch.sigmoid(self.unaware_head(torch.cat([reward_h, dist_h, corr_h, impulse, emotion], dim=-1)))
        
        knobs = torch.cat([emotion, impulse, unaware], dim=-1)
        temp_mult = 0.6 + 1.2 * torch.sigmoid(self.temp_head(knobs))
        wave_mix  = torch.sigmoid(self.mix_head(knobs))
        
        return reward_h, dist_h, impulse, emotion, unaware, temp_mult, wave_mix

# -------------------------
# 3. Main Model
# -------------------------
class ZenFusionNet(nn.Module):
    def __init__(self, vocab_size, num_particles, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Cortices (Only Sim now)
        self.sim_cortex = SimulationSensoryCortex(num_particles, output_dim=hidden_dim)
        
        # Logic
        self.logic = LogicDesignController(hidden_dim)

        # Dynamics
        self.rnn_branch = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.slope_projector = NCLinear(embed_dim, hidden_dim)
        self.slope_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Wave
        self.conv1 = nn.Conv1d(embed_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_wave = NCLinear(128, vocab_size)
        
        self.fc_final = NCLinear(hidden_dim, vocab_size)

    def forward(self, x, h=None, sim_tensor=None):
        emb = self.embedding(x) # (B, T, E)

        # 1. Process Simulation State
        sim_h = None
        if sim_tensor is not None:
            sim_h = self.sim_cortex(sim_tensor) # (B, H)
        
        impulse, emotion, wave_mix, temp_mult = None, None, None, None
        
        # Use Simulation State to drive Logic Controller
        logic_input = sim_h
        
        if logic_input is not None:
            _, _, impulse, emotion, unaware, temp_mult, wave_mix = self.logic(logic_input)
            
            # Seed hidden state with Logic Context (from Simulation)
            if h is None:
                h = logic_input.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # 2. Particle Dynamics (Text)
        slope = emb[:, 1:, :] - emb[:, :-1, :]
        slope = torch.cat([torch.zeros(emb.size(0), 1, emb.size(2)).to(device), slope], dim=1)
        slope_h = torch.tanh(self.slope_projector(slope))
        
        rnn_out, h_next = self.rnn_branch(emb, h)
        
        combined = torch.cat([rnn_out, slope_h], dim=-1)
        gate = torch.sigmoid(self.slope_gate(combined))
        
        # Modulate Gate with Emotion/Impulse
        if emotion is not None:
             gate = gate * (1.0 + emotion.unsqueeze(1) * 0.5)

        fused = rnn_out + (gate * slope_h)
        logits_dyn = self.fc_final(fused[:, -1, :])

        # 3. Wave Branch
        x_perm = emb.transpose(1, 2)
        wave = F.relu(self.conv1(x_perm))
        wave = F.relu(self.conv2(wave))
        wave = self.pool(wave).view(emb.size(0), -1)
        logits_wave = self.fc_wave(wave)

        # 4. Mix
        if wave_mix is not None:
            logits = (1 - wave_mix) * logits_dyn + wave_mix * logits_wave
        else:
            logits = logits_dyn + logits_wave

        # Return control signals to feedback into Simulation
        control_signals = None
        if impulse is not None:
            c_imp = impulse.mean().item()
            c_emo = emotion.mean().item()
            c_mix = wave_mix.mean().item()
            control_signals = (c_imp, c_emo, c_mix)

        return logits, h_next, temp_mult, control_signals

# -------------------------
# 4. Training Loop
# -------------------------
def train(model, optimizer, loader, sim_engine, num_epochs):
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, num_epochs+1):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 1. Step Simulation
            sim_flat = sim_engine.step() 
            sim_batch = sim_flat.unsqueeze(0).expand(x.size(0), -1)
            
            # 2. Forward Pass (No Image)
            logits, _, _, controls = model(x, sim_tensor=sim_batch)
            
            # 3. Feedback
            if controls:
                sim_engine.step(control_signals=controls)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "speed": f"{sim_engine.speed:.3f}"})
            
        print(f"Epoch {epoch} Done. Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), CKPT_PATH)

# -------------------------
# 5. Utils & Main
# -------------------------
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x_list, y_val = self.data[idx]
        return torch.tensor(x_list, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)

if __name__ == "__main__":
    # Load Data
    try:
        with open("xaa", "r", encoding="utf-8") as f: text = f.read().lower()
    except:
        try:
            text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower()
        except:
            text = "hello world " * 1000
        
    words = text.split()[:KB_len]
    vocab = sorted(list(set(words)))
    w2i = {w:i for i,w in enumerate(vocab)}
    i2w = {i:w for w,i in w2i.items()}
    
    raw_samples = []
    for i in range(len(words)-SEQ_LEN):
        raw_samples.append((
            [w2i[w] for w in words[i:i+SEQ_LEN]],
            w2i[words[i+SEQ_LEN]]
        ))
    
    dataset = TextDataset(raw_samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Components (No Image Tensor)
    sim_engine = ZenSimulationEngine(num_particles=200)
    
    model = ZenFusionNet(len(vocab), num_particles=200).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    if os.path.exists(CKPT_PATH):
        print("Loading checkpoint...")
        try:
            model.load_state_dict(torch.load(CKPT_PATH))
        except:
            print("Checkpoint mismatch or error, starting fresh.")

    print("Starting Zen Neural Trainer (Sim Only)...")
    try:
        train(model, optimizer, loader, sim_engine, NUM_EPOCHS)
    except KeyboardInterrupt:
        print("Saving...")
        torch.save(model.state_dict(), CKPT_PATH)

    print("\nInteractive Generator:")
    while True:
        seed = input(">> ")
        if not seed: break
        
        model.eval()
        gen_ids = [w2i.get(w, 0) for w in seed.split()]
        
        print(f"[Sim State] Pos: {sim_engine.entity_pos} | Speed: {sim_engine.speed:.3f}")
        
        for _ in range(500):
            sim_flat = sim_engine.step()
            
            inp = gen_ids[-SEQ_LEN:]
            if len(inp) < SEQ_LEN: inp = [0]*(SEQ_LEN-len(inp)) + inp
            xt = torch.tensor([inp], device=device)
            sim_t = sim_flat.unsqueeze(0)
            
            with torch.no_grad():
                logits, _, temp_mult, ctrls = model(xt, sim_tensor=sim_t)
            
            if ctrls: sim_engine.step(ctrls)

            temp = 0.6 * (temp_mult.item() if temp_mult else 1.0)
            probs = F.softmax(logits[0] / temp, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            gen_ids.append(next_id)
            print(i2w[next_id], end=' ', flush=True)
        print("\n")


# combined_astro_protein_and_ecm_peptide_driven.py
#
# Tab 1: Protein (AstroSolver brute force -> JSON)
# Tab 2: ECM sim where parameters are derived from a peptide sequence ("use the peptide in the ECM")
#
# pip install numpy matplotlib

import json
import itertools
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# Amino acid properties
# =========================
AMINO_ACIDS: Dict[str, Dict[str, Any]] = {
    'A': {'name': 'Alanine',       'hydrophobic':  1.8, 'charge':  0.0, 'pKa': None,  'secondary': 'helix'},
    'R': {'name': 'Arginine',      'hydrophobic': -4.5, 'charge':  1.0, 'pKa': 12.48, 'secondary': 'coil'},
    'N': {'name': 'Asparagine',    'hydrophobic': -3.5, 'charge':  0.0, 'pKa': None,  'secondary': 'turn'},
    'D': {'name': 'Aspartic Acid', 'hydrophobic': -3.5, 'charge': -1.0, 'pKa': 3.65,  'secondary': 'helix'},
    'C': {'name': 'Cysteine',      'hydrophobic':  2.5, 'charge':  0.0, 'pKa': 8.18,  'secondary': 'sheet'},
    'Q': {'name': 'Glutamine',     'hydrophobic': -3.5, 'charge':  0.0, 'pKa': None,  'secondary': 'helix'},
    'E': {'name': 'Glutamic Acid', 'hydrophobic': -3.5, 'charge': -1.0, 'pKa': 4.25,  'secondary': 'helix'},
    'G': {'name': 'Glycine',       'hydrophobic': -0.4, 'charge':  0.0, 'pKa': None,  'secondary': 'turn'},
    'H': {'name': 'Histidine',     'hydrophobic': -3.2, 'charge':  0.1, 'pKa': 6.00,  'secondary': 'coil'},
    'I': {'name': 'Isoleucine',    'hydrophobic':  4.5, 'charge':  0.0, 'pKa': None,  'secondary': 'sheet'},
    'L': {'name': 'Leucine',       'hydrophobic':  3.8, 'charge':  0.0, 'pKa': None,  'secondary': 'helix'},
    'K': {'name': 'Lysine',        'hydrophobic': -3.9, 'charge':  1.0, 'pKa': 10.53, 'secondary': 'helix'},
    'M': {'name': 'Methionine',    'hydrophobic':  1.9, 'charge':  0.0, 'pKa': None,  'secondary': 'helix'},
    'F': {'name': 'Phenylalanine', 'hydrophobic':  2.8, 'charge':  0.0, 'pKa': None,  'secondary': 'sheet'},
    'P': {'name': 'Proline',       'hydrophobic': -1.6, 'charge':  0.0, 'pKa': None,  'secondary': 'turn'},
    'S': {'name': 'Serine',        'hydrophobic': -0.8, 'charge':  0.0, 'pKa': None,  'secondary': 'coil'},
    'T': {'name': 'Threonine',     'hydrophobic': -0.7, 'charge':  0.0, 'pKa': None,  'secondary': 'sheet'},
    'W': {'name': 'Tryptophan',    'hydrophobic': -0.9, 'charge':  0.0, 'pKa': None,  'secondary': 'sheet'},
    'Y': {'name': 'Tyrosine',      'hydrophobic': -1.3, 'charge':  0.0, 'pKa': 10.07, 'secondary': 'sheet'},
    'V': {'name': 'Valine',        'hydrophobic':  4.2, 'charge':  0.0, 'pKa': None,  'secondary': 'sheet'},
}

SS_SCORE = {'helix': 1, 'sheet': 2, 'coil': 3, 'turn': 4}

def residue_contact_energy(hydro: float) -> int:
    return max(1, int((hydro + 5.0) * 5.0))

def peptide_features(seq: str) -> Dict[str, float]:
    seq = seq.strip().upper()
    props = [AMINO_ACIDS[a] for a in seq]
    net_charge = float(sum(p["charge"] for p in props))
    avg_hydro = float(sum(p["hydrophobic"] for p in props) / len(props))
    ss_score = float(sum(SS_SCORE[p["secondary"]] for p in props))
    return {
        "length": float(len(seq)),
        "net_charge": net_charge,
        "avg_hydro": avg_hydro,
        "ss_score": ss_score,
    }

def map_peptide_to_ecm_params(seq: str) -> Dict[str, float]:
    """
    Toy mapping: peptide -> ECM parameters.
    Higher avg hydrophobicity -> higher baseline ECM density and higher fibro deposition.
    Higher |charge| -> more "protease pressure" -> higher cancer degradation.
    """
    f = peptide_features(seq)
    # Normalize hydrophobicity range roughly [-4.5, +4.5] -> [0,1]
    h01 = float(np.clip((f["avg_hydro"] + 4.5) / 9.0, 0.0, 1.0))
    # Normalize charge magnitude (soft)
    c01 = float(np.clip(abs(f["net_charge"]) / max(1.0, f["length"]), 0.0, 1.0))

    # ECM initial density mean and noise
    init_mean = 0.15 + 0.55 * h01
    init_noise = 0.10 + 0.15 * (1.0 - h01)

    # Fibro deposition (fibrosis-like build) and ECM decay
    fibro_deposit = 0.01 + 0.06 * h01
    ecm_decay = 0.005 + 0.02 * (1.0 - h01)

    # Cancer degradation (MMP-like) scales with charge magnitude (toy proxy)
    cancer_degrade = 0.02 + 0.10 * c01

    # Stiffness threshold: denser ECM reduces movement; set higher threshold for low hydro
    stiff_threshold = 0.55 + 0.35 * h01

    # Base move probability (how motile cells are in general)
    base_move = 0.85 - 0.35 * h01 + 0.10 * (1.0 - c01)

    return {
        "h01": h01,
        "c01": c01,
        "init_mean": float(np.clip(init_mean, 0.0, 1.0)),
        "init_noise": float(np.clip(init_noise, 0.0, 0.5)),
        "ecm_decay": float(np.clip(ecm_decay, 0.0, 0.1)),
        "fibro_deposit": float(np.clip(fibro_deposit, 0.0, 0.2)),
        "cancer_degrade": float(np.clip(cancer_degrade, 0.0, 0.3)),
        "stiff_threshold": float(np.clip(stiff_threshold, 0.0, 1.0)),
        "base_move": float(np.clip(base_move, 0.05, 1.0)),
    }


# =========================
# AstroPhysicsSolver (subset-sum mode)
# =========================
class AstroDomain:
    def __init__(self, name, initial_scale=0.5):
        self.name = name
        self.val = float(initial_scale)
        self.velocity = 0.0

    def update_multiplicative(self, factor, dt):
        target_velocity = factor
        self.velocity = (self.velocity * 0.8) + (target_velocity * 0.2)
        step_change = float(np.clip(self.velocity * dt, -0.1, 0.1))
        self.val *= (1.0 + step_change)
        if self.val < 1e-12:
            self.val = 1e-12


class AstroPhysicsSolver:
    def __init__(self):
        self.variables = {}

    def reset(self):
        self.variables = {}

    def create_var(self, name, rough_magnitude):
        self.variables[name] = AstroDomain(name, initial_scale=rough_magnitude)

    def _solve_subset_sum_exact_indices(self, numbers: List[int], target: int) -> Optional[List[int]]:
        if target < 0:
            return None
        if target == 0:
            return []
        n = len(numbers)
        dp = [False] * (target + 1)
        prev = [None] * (target + 1)
        dp[0] = True

        for i in range(n):
            w = numbers[i]
            if w > target:
                continue
            for s in range(target, w - 1, -1):
                if (not dp[s]) and dp[s - w]:
                    dp[s] = True
                    prev[s] = (s - w, i)

        if not dp[target]:
            return None

        idxs = []
        s = target
        while s != 0:
            ps = prev[s]
            if ps is None:
                return None
            s, i = ps
            idxs.append(i)
        idxs.reverse()
        return idxs

    def _solve_subset_sum_annealing_indices(self, numbers: List[int], target: int, steps: int = 50000) -> Optional[List[int]]:
        self.reset()
        n = len(numbers)
        for i in range(n):
            self.create_var(f'incl_{i}', rough_magnitude=0.5)

        for _t in range(steps):
            vals = {k: d.val for k, d in self.variables.items()}
            current_sum = sum(vals[f'incl_{i}'] * numbers[i] for i in range(n))
            if abs(target - current_sum) < 1e-6:
                break

            perturb = 0.01
            for i in range(n):
                name = f'incl_{i}'
                domain = self.variables[name]
                orig = float(np.clip(domain.val, 0.0, 1.0))

                domain.val = min(1.0, orig + perturb)
                up = sum(self.variables[f'incl_{j}'].val * numbers[j] for j in range(n))

                domain.val = max(0.0, orig - perturb)
                down = sum(self.variables[f'incl_{j}'].val * numbers[j] for j in range(n))

                domain.val = orig

                sensitivity = (up - down) / (2.0 * perturb)
                if abs(sensitivity) < 1e-8:
                    sensitivity = float(numbers[i])

                force = (target - current_sum) / sensitivity
                force *= 0.1
                domain.update_multiplicative(force, dt=0.01)
                domain.val = float(np.clip(domain.val, 0.0, 1.0))

        incl = [int(round(float(np.clip(self.variables[f'incl_{i}'].val, 0.0, 1.0)))) for i in range(n)]
        idxs = [i for i, b in enumerate(incl) if b == 1]
        s = sum(numbers[i] for i in idxs)
        return idxs if s == target else None

    def solve_subset_sum(self,
                         numbers: List[int],
                         target: int,
                         steps: int = 50000,
                         max_exact_target: int = 20000) -> Dict[str, Any]:
        if target < 0:
            return {"found": False, "method": "invalid", "indices": None}

        if target <= max_exact_target:
            idxs = self._solve_subset_sum_exact_indices(numbers, target)
            if idxs is not None:
                return {"found": True, "method": "exact_dp", "indices": idxs}

        idxs = self._solve_subset_sum_annealing_indices(numbers, target, steps=steps)
        if idxs is not None:
            return {"found": True, "method": "annealing", "indices": idxs}

        return {"found": False, "method": "failed", "indices": None}


# =========================
# Protein brute-force engine
# =========================
def compute_protein_record(seq: str,
                           solver: AstroPhysicsSolver,
                           target_mode: str,
                           fixed_target: Optional[int],
                           anneal_steps: int,
                           max_exact_target: int) -> Dict[str, Any]:
    seq = seq.strip().upper()
    props = [AMINO_ACIDS[a] for a in seq]

    net_charge = sum(p["charge"] for p in props)
    avg_hydro = sum(p["hydrophobic"] for p in props) / len(props)
    ss_score = sum(SS_SCORE[p["secondary"]] for p in props)
    ionizable = [p["pKa"] for p in props if p["pKa"] is not None]

    contact_energies = [residue_contact_energy(p["hydrophobic"]) for p in props]
    total_contact_energy = int(sum(contact_energies))

    if target_mode == "half":
        target = total_contact_energy // 2
    elif target_mode == "sum":
        target = total_contact_energy
    elif target_mode == "fixed":
        if fixed_target is None:
            raise ValueError("fixed target_mode requires fixed_target")
        target = int(fixed_target)
    else:
        raise ValueError("target_mode must be half|sum|fixed")

    fold = solver.solve_subset_sum(
        numbers=contact_energies,
        target=target,
        steps=anneal_steps,
        max_exact_target=max_exact_target,
    )

    chosen_energies = None
    chosen_sum = None
    if fold["found"]:
        idxs = fold["indices"]
        chosen_energies = [contact_energies[i] for i in idxs]
        chosen_sum = int(sum(chosen_energies))

    return {
        "sequence": seq,
        "composition": [p["name"] for p in props],
        "properties": {
            "length": len(seq),
            "net_charge": round(float(net_charge), 3),
            "avg_hydrophobicity": round(float(avg_hydro), 3),
            "secondary_score": int(ss_score),
            "ionizable_pKas": ionizable,
            "contact_energies": contact_energies,
            "total_contact_energy": total_contact_energy,
        },
        "astro_fold": {
            "target_mode": target_mode,
            "target": int(target),
            "found": bool(fold["found"]),
            "method": fold["method"],
            "chosen_indices": fold["indices"],
            "chosen_energies": chosen_energies,
            "chosen_sum": chosen_sum,
        }
    }


# =========================
# ECM simulation engine (peptide-driven params)
# =========================
@dataclass
class Cell:
    x: int
    y: int
    kind: str  # "fibro" or "cancer"

class ECMWorld:
    def __init__(self, n: int, init_mean: float, init_noise: float):
        self.n = n
        noise = (np.random.rand(n, n) - 0.5) * 2.0 * init_noise
        self.density = np.clip(init_mean + noise, 0.0, 1.0)

    def decay(self, rate: float):
        self.density *= (1.0 - rate)

    def clamp(self):
        np.clip(self.density, 0.0, 1.0, out=self.density)

class ECMSim:
    def __init__(self, grid_n=100, n_fibro=60, n_cancer=40,
                 init_mean=0.3, init_noise=0.15):
        self.grid_n = grid_n
        self.world = ECMWorld(grid_n, init_mean=init_mean, init_noise=init_noise)
        self.cells: List[Cell] = []
        self.step_idx = 0

        cx = grid_n // 2
        cy = grid_n // 2
        for _ in range(n_fibro):
            self.cells.append(Cell(
                x=int(np.clip(cx + np.random.randn() * 6, 0, grid_n - 1)),
                y=int(np.clip(cy + np.random.randn() * 6, 0, grid_n - 1)),
                kind="fibro"
            ))

        for _ in range(n_cancer):
            self.cells.append(Cell(
                x=int(np.clip(10 + np.random.randn() * 6, 0, grid_n - 1)),
                y=int(np.clip(10 + np.random.randn() * 6, 0, grid_n - 1)),
                kind="cancer"
            ))

    def step(self,
             ecm_decay: float,
             fibro_deposit: float,
             cancer_degrade: float,
             stiff_threshold: float,
             base_move: float):
        n = self.grid_n
        d = self.world.density

        for c in self.cells:
            local = float(d[c.x, c.y])

            if c.kind == "fibro":
                d[c.x, c.y] += fibro_deposit
            else:
                d[c.x, c.y] -= cancer_degrade

            # Movement slowed in stiff/dense ECM (toy), reflecting barrier effects. 
            # The dependence of migration on stiffness/density is well known, but here is simplified. 
            # [web:312][web:315][web:320]
            speed = base_move if local <= stiff_threshold else (base_move * 0.2)

            if np.random.rand() < speed:
                dx = np.random.randint(-1, 2)
                dy = np.random.randint(-1, 2)
                c.x = int(np.clip(c.x + dx, 0, n - 1))
                c.y = int(np.clip(c.y + dy, 0, n - 1))

        self.world.decay(ecm_decay)
        self.world.clamp()
        self.step_idx += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.step_idx,
            "grid_n": self.grid_n,
            "ecm_density": self.world.density.tolist(),
            "cells": [{"x": c.x, "y": c.y, "kind": c.kind} for c in self.cells],
        }


# =========================
# GUI
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AstroSolver Protein + Peptide-driven ECM")
        self.geometry("1250x820")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.tab_protein = ttk.Frame(self.nb)
        self.tab_ecm = ttk.Frame(self.nb)

        self.nb.add(self.tab_protein, text="Protein (AstroSolver -> JSON)")
        self.nb.add(self.tab_ecm, text="ECM (peptide-driven)")

        self._build_protein_tab()
        self._build_ecm_tab()

        self._protein_worker = None
        self._stop_protein = threading.Event()

        self.ecm_sim: Optional[ECMSim] = None
        self.ecm_running = False
        self.ecm_params = None  # peptide-derived parameters

    # ---------- Protein tab ----------
    def _build_protein_tab(self):
        frame = self.tab_protein

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Peptide length L:").pack(side="left")
        self.pep_len = tk.StringVar(value="2")
        ttk.Entry(top, textvariable=self.pep_len, width=6).pack(side="left", padx=6)

        ttk.Label(top, text="Target mode:").pack(side="left", padx=(12, 0))
        self.target_mode = tk.StringVar(value="half")
        ttk.Combobox(top, textvariable=self.target_mode, values=["half", "sum", "fixed"], width=8, state="readonly").pack(side="left", padx=6)

        ttk.Label(top, text="Fixed target:").pack(side="left", padx=(12, 0))
        self.fixed_target = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.fixed_target, width=10).pack(side="left", padx=6)

        ttk.Label(top, text="Anneal steps:").pack(side="left", padx=(12, 0))
        self.anneal_steps = tk.StringVar(value="2000")
        ttk.Entry(top, textvariable=self.anneal_steps, width=8).pack(side="left", padx=6)

        ttk.Label(top, text="Max exact target:").pack(side="left", padx=(12, 0))
        self.max_exact_target = tk.StringVar(value="20000")
        ttk.Entry(top, textvariable=self.max_exact_target, width=8).pack(side="left", padx=6)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=10)

        ttk.Button(btns, text="Choose output JSON…", command=self._choose_out).pack(side="left")
        self.out_path = tk.StringVar(value="astro_protein_out.json")
        ttk.Entry(btns, textvariable=self.out_path, width=60).pack(side="left", padx=8)

        ttk.Button(btns, text="Start brute-force", command=self._start_protein).pack(side="left", padx=6)
        ttk.Button(btns, text="Stop", command=self._stop_protein_job).pack(side="left", padx=6)

        mid = ttk.Frame(frame)
        mid.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(mid, text="Send peptide to ECM:").pack(side="left")
        self.peptide_send = tk.StringVar(value="ACDE")
        ttk.Entry(mid, textvariable=self.peptide_send, width=20).pack(side="left", padx=6)
        ttk.Button(mid, text="Apply to ECM tab", command=self._send_peptide_to_ecm).pack(side="left", padx=6)

        self.prog = ttk.Progressbar(frame, mode="determinate")
        self.prog.pack(fill="x", padx=10, pady=10)

        self.protein_log = scrolledtext.ScrolledText(frame, height=24, font=("Consolas", 10))
        self.protein_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._plog("Ready.\n")

    def _plog(self, s: str):
        self.protein_log.insert("end", s)
        self.protein_log.see("end")

    def _choose_out(self):
        path = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.out_path.set(path)

    def _validate_peptide(self, seq: str) -> Tuple[bool, str]:
        seq = seq.strip().upper()
        if not seq:
            return False, "Empty peptide."
        bad = [c for c in seq if c not in AMINO_ACIDS]
        if bad:
            return False, f"Invalid residues: {sorted(set(bad))}. Use only 20 standard AA letters."
        return True, seq

    def _send_peptide_to_ecm(self):
        ok, msg = self._validate_peptide(self.peptide_send.get())
        if not ok:
            messagebox.showerror("Peptide", msg)
            return

        # Compute peptide-driven ECM parameters, then switch to ECM tab.
        self.ecm_peptide_var.set(msg)
        self._ecm_apply_peptide()
        self.nb.select(self.tab_ecm)

    def _start_protein(self):
        if self._protein_worker and self._protein_worker.is_alive():
            messagebox.showinfo("Protein", "Already running.")
            return

        try:
            L = int(self.pep_len.get())
            if L < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Protein", "Peptide length must be an integer >= 1.")
            return

        if L > 5:
            if not messagebox.askyesno("Protein", "L>5 is huge (20^L). Continue anyway?"):
                return

        try:
            steps = int(self.anneal_steps.get())
            max_exact = int(self.max_exact_target.get())
        except ValueError:
            messagebox.showerror("Protein", "Anneal steps and max exact target must be integers.")
            return

        tmode = self.target_mode.get().strip()
        ftarget = self.fixed_target.get().strip()
        ftarget_int = int(ftarget) if (tmode == "fixed" and ftarget) else None
        if tmode == "fixed" and ftarget_int is None:
            messagebox.showerror("Protein", "Target mode fixed requires Fixed target.")
            return

        outp = self.out_path.get().strip()
        if not outp:
            messagebox.showerror("Protein", "Output path required.")
            return

        self._stop_protein.clear()
        self.protein_log.delete("1.0", "end")
        self._plog("Starting brute force...\n")

        total = 20 ** L
        self.prog["value"] = 0
        self.prog["maximum"] = total

        def worker():
            solver = AstroPhysicsSolver()
            codes = list(AMINO_ACIDS.keys())
            records = []

            start = time.time()
            i = 0
            for tup in itertools.product(codes, repeat=L):
                if self._stop_protein.is_set():
                    break

                seq = "".join(tup)
                solver.reset()
                rec = compute_protein_record(
                    seq=seq,
                    solver=solver,
                    target_mode=tmode,
                    fixed_target=ftarget_int,
                    anneal_steps=steps,
                    max_exact_target=max_exact,
                )
                records.append(rec)
                i += 1

                if i % 200 == 0 or i == total:
                    self.after(0, lambda v=i: self.prog.configure(value=v))
                if i % 2000 == 0:
                    self.after(0, lambda v=i, r=rec: self._plog(
                        f"{v}/{total}  sample={r['sequence']}  found={r['astro_fold']['found']}  method={r['astro_fold']['method']}\n"
                    ))

            payload = {
                "meta": {
                    "generated_utc": datetime.utcnow().isoformat() + "Z",
                    "length": L,
                    "count": len(records),
                    "target_mode": tmode,
                    "fixed_target": ftarget_int,
                    "anneal_steps": steps,
                    "max_exact_target": max_exact,
                    "amino_acids": sorted(codes),
                    "stopped_early": bool(self._stop_protein.is_set()),
                    "elapsed_s": round(time.time() - start, 3),
                },
                "records": records,
            }

            with open(outp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.after(0, lambda: self._plog(f"\nDone. Wrote {len(records)} records to:\n{outp}\n"))

        self._protein_worker = threading.Thread(target=worker, daemon=True)
        self._protein_worker.start()

    def _stop_protein_job(self):
        self._stop_protein.set()
        self._plog("\nStop requested...\n")

    # ---------- ECM tab ----------
    def _build_ecm_tab(self):
        frame = self.tab_ecm

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=10, pady=10)

        self.ecm_grid_n = tk.StringVar(value="120")
        self.ecm_fibro = tk.StringVar(value="80")
        self.ecm_cancer = tk.StringVar(value="40")

        ttk.Label(top, text="Grid N:").pack(side="left")
        ttk.Entry(top, textvariable=self.ecm_grid_n, width=6).pack(side="left", padx=6)
        ttk.Label(top, text="Fibro:").pack(side="left", padx=(12, 0))
        ttk.Entry(top, textvariable=self.ecm_fibro, width=6).pack(side="left", padx=6)
        ttk.Label(top, text="Cancer:").pack(side="left", padx=(12, 0))
        ttk.Entry(top, textvariable=self.ecm_cancer, width=6).pack(side="left", padx=6)

        pep = ttk.Frame(frame)
        pep.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(pep, text="Peptide (drives ECM):").pack(side="left")
        self.ecm_peptide_var = tk.StringVar(value="ACDE")
        ttk.Entry(pep, textvariable=self.ecm_peptide_var, width=30).pack(side="left", padx=6)
        ttk.Button(pep, text="Apply peptide -> ECM params", command=self._ecm_apply_peptide).pack(side="left", padx=6)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Init (with peptide)", command=self._ecm_init).pack(side="left", padx=6)
        ttk.Button(btns, text="Start", command=self._ecm_start).pack(side="left", padx=6)
        ttk.Button(btns, text="Stop", command=self._ecm_stop).pack(side="left", padx=6)
        ttk.Button(btns, text="Save snapshot JSON…", command=self._ecm_save_snapshot).pack(side="left", padx=12)

        self.ecm_status = tk.StringVar(value="Not initialized.")
        ttk.Label(frame, textvariable=self.ecm_status).pack(anchor="w", padx=10)

        fig = Figure(figsize=(6, 5), dpi=110)
        self.ecm_ax = fig.add_subplot(111)
        self.ecm_ax.set_title("ECM density (background) + cells")
        self.ecm_ax.set_xticks([])
        self.ecm_ax.set_yticks([])

        self.ecm_im = self.ecm_ax.imshow(np.zeros((10, 10)), vmin=0, vmax=1, cmap="viridis", origin="lower")
        self.ecm_sc_fib = self.ecm_ax.scatter([], [], s=8, c="lime", label="fibro")
        self.ecm_sc_can = self.ecm_ax.scatter([], [], s=8, c="red", label="cancer")
        self.ecm_ax.legend(loc="upper right", fontsize=8)

        self.ecm_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.ecm_canvas.draw()
        self.ecm_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.ecm_log = scrolledtext.ScrolledText(frame, height=10, font=("Consolas", 10))
        self.ecm_log.pack(fill="x", padx=10, pady=(0, 10))
        self._elog("Ready.\n")

        self._ecm_apply_peptide()

    def _elog(self, s: str):
        self.ecm_log.insert("end", s)
        self.ecm_log.see("end")

    def _ecm_apply_peptide(self):
        ok, msg = self._validate_peptide(self.ecm_peptide_var.get())
        if not ok:
            messagebox.showerror("ECM peptide", msg)
            return

        self.ecm_peptide_var.set(msg)
        self.ecm_params = map_peptide_to_ecm_params(msg)
        f = peptide_features(msg)

        self._elog(
            "\nPeptide -> ECM params applied\n"
            f"  peptide={msg}\n"
            f"  avg_hydro={f['avg_hydro']:.3f}  net_charge={f['net_charge']:.3f}\n"
            f"  init_mean={self.ecm_params['init_mean']:.3f}  init_noise={self.ecm_params['init_noise']:.3f}\n"
            f"  fibro_deposit={self.ecm_params['fibro_deposit']:.3f}  cancer_degrade={self.ecm_params['cancer_degrade']:.3f}\n"
            f"  ecm_decay={self.ecm_params['ecm_decay']:.3f}  stiff_threshold={self.ecm_params['stiff_threshold']:.3f}  base_move={self.ecm_params['base_move']:.3f}\n"
        )

        self.ecm_status.set(f"Peptide loaded: {msg} (init_mean={self.ecm_params['init_mean']:.2f}, stiff_thr={self.ecm_params['stiff_threshold']:.2f})")

    def _ecm_init(self):
        try:
            n = int(self.ecm_grid_n.get())
            nf = int(self.ecm_fibro.get())
            nc = int(self.ecm_cancer.get())
        except ValueError:
            messagebox.showerror("ECM", "Grid N / counts must be integers.")
            return

        if self.ecm_params is None:
            self._ecm_apply_peptide()
            if self.ecm_params is None:
                return

        p = self.ecm_params
        self.ecm_sim = ECMSim(
            grid_n=n,
            n_fibro=nf,
            n_cancer=nc,
            init_mean=p["init_mean"],
            init_noise=p["init_noise"],
        )
        self.ecm_running = False
        self.ecm_status.set(f"Initialized with peptide. step=0  N={n}  fibro={nf}  cancer={nc}")
        self._elog(f"Initialized ECM sim with peptide parameters.\n")
        self._ecm_redraw()

    def _ecm_start(self):
        if self.ecm_sim is None:
            self._ecm_init()
        if self.ecm_sim is None:
            return
        if self.ecm_running:
            return
        self.ecm_running = True
        self._elog("Starting...\n")
        self._ecm_tick()

    def _ecm_stop(self):
        self.ecm_running = False
        self._elog("Stopped.\n")

    def _ecm_tick(self):
        if not self.ecm_running or self.ecm_sim is None:
            return

        if self.ecm_params is None:
            self._ecm_apply_peptide()
            if self.ecm_params is None:
                return

        p = self.ecm_params
        self.ecm_sim.step(
            ecm_decay=p["ecm_decay"],
            fibro_deposit=p["fibro_deposit"],
            cancer_degrade=p["cancer_degrade"],
            stiff_threshold=p["stiff_threshold"],
            base_move=p["base_move"],
        )

        if self.ecm_sim.step_idx % 5 == 0:
            self._ecm_redraw()
            self.ecm_status.set(f"Running. step={self.ecm_sim.step_idx}  peptide={self.ecm_peptide_var.get()}")

        self.after(20, self._ecm_tick)

    def _ecm_redraw(self):
        if self.ecm_sim is None:
            return
        sim = self.ecm_sim

        self.ecm_im.set_data(sim.world.density)

        fib = [(c.x, c.y) for c in sim.cells if c.kind == "fibro"]
        can = [(c.x, c.y) for c in sim.cells if c.kind == "cancer"]

        if fib:
            xs, ys = zip(*fib)
            self.ecm_sc_fib.set_offsets(np.c_[ys, xs])
        else:
            self.ecm_sc_fib.set_offsets(np.empty((0, 2)))

        if can:
            xs, ys = zip(*can)
            self.ecm_sc_can.set_offsets(np.c_[ys, xs])
        else:
            self.ecm_sc_can.set_offsets(np.empty((0, 2)))

        self.ecm_canvas.draw_idle()

    def _ecm_save_snapshot(self):
        if self.ecm_sim is None:
            messagebox.showinfo("ECM", "Initialize first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save ECM snapshot JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        snap = self.ecm_sim.snapshot()
        snap["generated_utc"] = datetime.utcnow().isoformat() + "Z"
        snap["peptide"] = self.ecm_peptide_var.get().strip().upper()
        snap["peptide_ecm_params"] = self.ecm_params

        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)

        self._elog(f"Saved snapshot: {path}\n")


if __name__ == "__main__":
    App().mainloop()

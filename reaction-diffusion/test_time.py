#!/usr/bin/env python3
import argparse
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

# ==========================================
# Model & Architecture (Unified)
# ==========================================
def get_act(name: str):
    n = name.lower()
    if n == "relu":  return nn.ReLU()
    if n == "gelu":  return nn.GELU()
    if n == "tanh":  return nn.Tanh()
    if n in ("silu", "swish"): return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")

def fnn(sizes, act="relu", last_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(get_act(act))
        elif last_act is not None:
            layers.append(get_act(last_act))
    return nn.Sequential(*layers)

class MergeDeepONet(dde.nn.pytorch.NN):
    def __init__(self, branch_layers, trunk_layers, merger_layers,
                 activation="relu", trunk_chunk=4096, batch_chunk=0, amp=False):
        super().__init__()
        self.trunk_chunk, self.batch_chunk, self.amp = trunk_chunk, batch_chunk, amp
        self.branch_fnn = fnn(branch_layers, act=activation)
        self.trunk_fnn  = fnn(trunk_layers,  act=activation)
        self.merger = fnn(merger_layers, act=activation, last_act=None)

    def _merge_block(self, b_block, t_block):
        Bb, nC = b_block.shape[0], t_block.shape[0]
        b_exp = b_block[:, None, :].expand(Bb, nC, -1)
        t_exp = t_block[None, :, :].expand(Bb, nC, -1)
        return self.merger(b_exp * t_exp).view(Bb, nC, 1)

    def forward(self, inputs):
        branch_batch, trunk_points = inputs
        B, N = branch_batch.shape[0], trunk_points.shape[0]
        b_all, t_all = self.branch_fnn(branch_batch), self.trunk_fnn(trunk_points)
        
        outs_B = []
        bstep = self.batch_chunk if self.batch_chunk > 0 else B
        for bs in range(0, B, bstep):
            be = min(B, bs + bstep)
            b_block = b_all[bs:be]
            outs_N = []
            nstep = self.trunk_chunk if self.trunk_chunk > 0 else N
            for ns in range(0, N, nstep):
                ne = min(N, ns + nstep)
                outs_N.append(self._merge_block(b_block, t_all[ns:ne]))
            outs_B.append(torch.cat(outs_N, dim=1))
        return torch.cat(outs_B, dim=0)

# ==========================================
# Features & Physics
# ==========================================
def make_trunk_1d(ny, n_harm_y):
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    cols = [np.ones((ny, 1)), y[:, None]]
    for n in range(1, n_harm_y + 1):
        w = 2 * np.pi * n
        cols += [np.cos(w * y)[:, None], np.sin(w * y)[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)

def make_trunk_2d(nx, ny, n_harm_x, n_harm_y, n_cross):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xf, Yf = Xg.flatten(), Yg.flatten()
    cols = [np.ones((Xf.size, 1)), Xf[:, None], Yf[:, None]]
    for n in range(1, n_harm_x + 1):
        w = 2 * np.pi * n
        cols += [np.cos(w * Xf)[:, None], np.sin(w * Xf)[:, None]]
    for n in range(1, n_harm_y + 1):
        w = 2 * np.pi * n
        cols += [np.cos(w * Yf)[:, None], np.sin(w * Yf)[:, None]]
    for k in range(1, n_cross + 1):
        w = 2 * np.pi * k
        z = w * (Xf + Yf)
        cols += [np.cos(z)[:, None], np.sin(z)[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)

def _generate_pores_numpy(p_arr, res, lowval=0.1):
    n = int(math.sqrt(p_arr.size))
    Nblk = res // n
    pores = np.ones((res, res))
    k = 0
    for i in range(n):
        for j in range(n):
            p = p_arr[k]
            half = p * Nblk / 2
            inner = math.floor(half)
            w = np.clip(half - inner, 0.0, 1.0)
            isqa, isqb = Nblk//2 - inner, Nblk//2 + inner
            pores[i*Nblk+isqa : i*Nblk+isqb, j*Nblk+isqa : j*Nblk+isqb] = lowval
            k += 1
    return pores

def compute_kappa_numpy(X_params, T_img, nx, ny):
    res = nx + 1
    iline = res // 2
    pores = _generate_pores_numpy(X_params, res)
    c1 = 0.5 * (pores[:-1, :-1] + pores[1:, :-1])
    dT = T_img[iline, :] - T_img[iline - 1, :]
    return np.sum(c1[iline, :] * dT)

# ==========================================
# Core Timing Logic
# ==========================================
def run_timing_unified(run_dir, device_str="cuda", trials=100):
    device = torch.device(device_str)
    
    with open(os.path.join(run_dir, "args.json"), "r") as f:
        args = json.load(f)
    
    dim = args.get("dim", "2d")
    ny, nx = 38, 840
    
    # 1. Prep Model
    if dim == "2d":
        trunk_full = make_trunk_2d(nx, ny, args["pe_x"], args["pe_y"], args["pe_cross"])
    else:
        trunk_full = make_trunk_1d(ny, args["pe_y"])
    
    branch_in = 25 # Default fourier size
    branch_layers = [branch_in] + args["branch_hidden"]
    trunk_layers  = [trunk_full.shape[1]] + args["trunk_hidden"]
    merger_layers = [branch_layers[-1]] + args["merger_hidden"] + [1]
    
    net = MergeDeepONet(branch_layers, trunk_layers, merger_layers, args["activation"]).to(device).eval()
    
    model_path = os.path.join(run_dir, "models", "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, "models", "final_model.pt")
    
    checkpoint = torch.load(model_path, map_location=device)
    # Handle DDE save format (state_dict is often in ['model_state_dict'])
    s_dict = checkpoint if isinstance(checkpoint, dict) and "branch_fnn.0.weight" in checkpoint else checkpoint.get("model_state_dict", checkpoint)
    net.load_state_dict(s_dict, strict=False)

    # 2. Prep Inputs
    X_sample_np = np.random.randn(branch_in).astype(np.float32)
    X_sample_t = torch.from_numpy(X_sample_np).to(device).unsqueeze(0)
    trunk_full_t = torch.from_numpy(trunk_full).to(device)
    trunk_pt_t = trunk_full_t[:1]

    y_mean = np.array([0.5]).astype(np.float32) if dim == "1d" else None
    y_std  = np.array([0.2]).astype(np.float32) if dim == "1d" else None

    def measure(fn):
        if device_str == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(trials):
            start = time.perf_counter()
            fn()
            if device_str == "cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter() - start)
        return np.mean(ts), np.std(ts)

    # Timing Tasks
    t_pt = measure(lambda: net((X_sample_t, trunk_pt_t)))
    t_line = measure(lambda: net((X_sample_t, trunk_full_t)))
    
    # Physics timing (CPU-bound usually)
    dummy_pred = np.random.randn(ny, nx) if dim == "2d" else np.random.randn(ny)
    def phys_task():
        if dim == "1d":
            p2d = np.repeat(dummy_pred[:, None], nx, axis=1)
            return compute_kappa_numpy(X_sample_np, p2d, nx, ny)
        return compute_kappa_numpy(X_sample_np, dummy_pred, nx, ny)
    t_phys = measure(phys_task)

    # Pipeline timing
    def pipeline_task():
        with torch.no_grad():
            pred = net((X_sample_t, trunk_full_t)).cpu().numpy().reshape(-1)
            if dim == "1d":
                pred = pred * y_std + y_mean
                pred_2d = np.repeat(pred[:, None], nx, axis=1)
            else:
                pred_2d = pred.reshape(ny, nx)
            return compute_kappa_numpy(X_sample_np, pred_2d, nx, ny)
    t_pipe = measure(pipeline_task)

    return t_pt, t_line, t_phys, t_pipe

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", help="Comma-separated paths to run directories")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    for rd in args.run_dirs.split(","):
        rd = rd.strip()
        if not rd: continue
        print(f"\n>>> Timing Results for: {rd}")
        try:
            res = run_timing_unified(rd, args.device, args.trials)
            labels = ["Single Point", "Full Prediction", "Physics (Flux)", "End-to-End Pipe"]
            for label, (mu, sd) in zip(labels, res):
                print(f"{label:<20}: {mu*1000:8.4f} ± {sd*1000:8.4f} ms")
        except Exception as e:
            print(f"Error timing {rd}: {e}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, time, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import deepxde as dde


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
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(get_act(act))
        elif last_act is not None:
            layers.append(get_act(last_act))
    return nn.Sequential(*layers)


def make_trunk_features_2d(nx, ny, n_harm_x=16, n_harm_y=4, n_cross=2):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xf, Yf = Xg.reshape(-1), Yg.reshape(-1)
    N = Xf.shape[0]
    cols = [np.ones((N, 1), np.float32), Xf[:, None], Yf[:, None]]
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

def make_monitor_trunk_2d(nx, ny, monitor_x, n_harm_x=16, n_harm_y=4, n_cross=2):
    xgrid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ygrid = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xf = np.full(ny, xgrid[monitor_x], dtype=np.float32)
    Yf = ygrid.astype(np.float32)
    cols = [np.ones((ny, 1), np.float32), Xf[:, None], Yf[:, None]]
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

def make_single_trunk_2d(nx, ny, monitor_x, y_index, n_harm_x=16, n_harm_y=4, n_cross=2):
    xgrid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ygrid = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xf = np.array([xgrid[monitor_x]], dtype=np.float32)
    Yf = np.array([ygrid[y_index]], dtype=np.float32)
    cols = [np.ones((1, 1), np.float32), Xf[:, None], Yf[:, None]]
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


def make_trunk_features_1d(nx, n_harm_x=16, n_cross=0):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    cols = [np.ones((nx, 1), np.float32), x[:, None]]
    for n in range(1, n_harm_x + 1):
        w = 2 * np.pi * n
        cols += [np.cos(w * x)[:, None], np.sin(w * x)[:, None]]
    for k in range(1, n_cross + 1):
        w = 2 * np.pi * k
        z = w * (x + x)
        cols += [np.cos(z)[:, None], np.sin(z)[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)

def make_monitor_trunk_1d(nx, monitor_x, n_harm_x=16, n_cross=0):
    xgrid = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    xv = np.array([xgrid[monitor_x]], dtype=np.float32)
    cols = [np.ones((1, 1), np.float32), xv[:, None]]
    for n in range(1, n_harm_x + 1):
        w = 2 * np.pi * n
        cols += [np.cos(w * xv)[:, None], np.sin(w * xv)[:, None]]
    for k in range(1, n_cross + 1):
        w = 2 * np.pi * k
        z = w * (xv + xv)
        cols += [np.cos(z)[:, None], np.sin(z)[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)


class MergeDeepONet(dde.nn.pytorch.NN):
    def __init__(self, branch_layers, trunk_layers, merger_layers,
                 activation="relu", trunk_chunk=4096, batch_chunk=0, amp=False):
        super().__init__()
        self.trunk_chunk = int(trunk_chunk)
        self.batch_chunk = int(batch_chunk)
        self.amp = bool(amp)
        self.branch_fnn = fnn(branch_layers, act=activation)
        self.trunk_fnn  = fnn(trunk_layers, act=activation)
        b_out, t_out = branch_layers[-1], trunk_layers[-1]
        if b_out != t_out:
            raise ValueError(f"Branch and trunk output dims must match. Got {b_out} vs {t_out}.")
        if merger_layers[-1] != 2:
            raise ValueError("Merger must end with size 2 (Re, Im).")
        self.merger = fnn(merger_layers, act=activation, last_act=None)

    def _merge_block(self, b_block, t_block):
        Bb, nC = b_block.shape[0], t_block.shape[0]
        b_exp = b_block[:, None, :].expand(Bb, nC, -1)
        t_exp = t_block[None, :, :].expand(Bb, nC, -1)
        fused = b_exp * t_exp
        merged = self.merger(fused.reshape(Bb * nC, -1))
        return merged.view(Bb, nC, 2)

    def forward(self, inputs):
        branch_batch, trunk_points = inputs
        use_amp = self.amp and branch_batch.device.type == 'cuda'
        
        if branch_batch.device.type == 'cuda':
            with torch.cuda.amp.autocast(enabled=use_amp):
                return self._forward_impl(branch_batch, trunk_points)
        else:
            return self._forward_impl(branch_batch, trunk_points)

    def _forward_impl(self, branch_batch, trunk_points):
        b_all = self.branch_fnn(branch_batch)
        t_all = self.trunk_fnn(trunk_points)
        out_blocks = []
        step = max(1, self.trunk_chunk)
        for ns in range(0, t_all.shape[0], step):
            t_block = t_all[ns:ns+step]
            out_blocks.append(self._merge_block(b_all, t_block))
        return torch.cat(out_blocks, dim=1)

def load_state_dict_safely(net, path, map_location):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    state = None
    if isinstance(ckpt, dict):
        for k in ("model","state_dict","net","nn_state_dict","weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
        if state is None and all(isinstance(k,str) for k in ckpt.keys()):
            state = ckpt
    else: state = ckpt
    net.load_state_dict(state, strict=False)

# timing
def measure_stats(fn, trials, device_str):
    """
    Runs `fn()` `trials` times.
    Returns (mean_time, std_time) in seconds.
    """
    for _ in range(10): # Warmup
        _ = fn()
    if device_str == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = fn()
        if device_str == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return float(np.mean(times)), float(np.std(times))

def run_timing_unified(run_dir, device_str, trials):
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    
    device = torch.device(device_str)
    print(f"Loading model on: {device} ...")

    with open(os.path.join(run_dir, "args.json")) as f: 
        a = json.load(f)
    
    # Auto-detect if 2D based on args
    is_2d = "pe_y" in a
    
    # Load Stats for Standardization
    stats = np.load(os.path.join(run_dir,"standardization_stats.npz"))
    X_mean, X_std = stats["X_mean"], stats["X_std"]

    # Load 1 sample from training data
    dtrain_path = a.get("train_file","../data/Maxwell_Train.npz")
    if not os.path.exists(dtrain_path):
        print(f"Warning: {dtrain_path} not found. Creating dummy input.")
        m_in = a["branch_hidden"][0] if a.get("branch_hidden") else 10 
        X_sample = np.random.randn(1, m_in).astype(np.float32)
    else:
        dtrain = np.load(dtrain_path, allow_pickle=True)
        # Handle dicts or npz variants gracefully
        if isinstance(dtrain, np.lib.npyio.NpzFile):
            X_raw = dtrain["X"][:1].astype(np.float32) if "X" in dtrain else dtrain["params"][:1].astype(np.float32)
        else:
            X_raw = dtrain.item()["params"][:1].astype(np.float32)
        X_sample = (X_raw - X_mean)/(X_std+1e-12)

    m = X_sample.shape[1]

    # Rebuild Model Configuration
    nx = 840
    monitor_x = 120 - 1
    pe_x = a.get("pe_x", 16)
    pe_cross = a.get("pe_cross", 2)
    
    if is_2d:
        ny = 38
        pe_y = a["pe_y"]
        trunk_dummy = make_trunk_features_2d(nx, ny, pe_x, pe_y, pe_cross)
    else:
        trunk_dummy = make_trunk_features_1d(nx, pe_x, pe_cross)

    dim_trunk = trunk_dummy.shape[1]
    branch_layers = [m] + a["branch_hidden"]
    trunk_layers = [dim_trunk] + a["trunk_hidden"]
    merger_layers = [branch_layers[-1]] + a["merger_hidden"] + [2]
    
    net = MergeDeepONet(branch_layers, trunk_layers, merger_layers,
                        activation=a["activation"],
                        trunk_chunk=a.get("trunk_chunk", 256),
                        batch_chunk=a.get("batch_chunk", 0),
                        amp=a.get("amp", False))
    
    net.to(device)
    net.eval()

    # Load weights
    ckpt = os.path.join(run_dir,"models","best_model.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir,"models","final_model.pt")
    
    load_state_dict_safely(net, ckpt, device)

    # Prepare Input Tensors
    X_t = torch.from_numpy(X_sample).to(device).float()
    
    results = {}
    print(f"Start benchmarking ({trials} trials)...")
    
    if is_2d:
        # --- 2D Timing ---
        y_mid = ny // 2
        trunk_single_t = torch.from_numpy(make_single_trunk_2d(nx, ny, monitor_x, y_mid, pe_x, pe_y, pe_cross)).to(device)
        trunk_monitor_t = torch.from_numpy(make_monitor_trunk_2d(nx, ny, monitor_x, pe_x, pe_y, pe_cross)).to(device)

        # Warmups are handled inside measure_stats, but precomputing cached array for avg_only is needed
        with torch.no_grad():
            out_monitor_cached = net((X_t, trunk_monitor_t))
        
        def avg_only():
            return torch.mean(out_monitor_cached, dim=1)

        def get_transmission():
            out = net((X_t, trunk_monitor_t))
            return torch.mean(out, dim=1)

        with torch.no_grad():
            results["Single Point"] = measure_stats(lambda: net((X_t, trunk_single_t)), trials, device_str)
            results["Monitor Line (Raw)"] = measure_stats(lambda: net((X_t, trunk_monitor_t)), trials, device_str)
            results["Monitor Averaging Only"] = measure_stats(avg_only, trials, device_str)
            results["Transmission (Full)"] = measure_stats(get_transmission, trials, device_str)
            
    else:
        # --- 1D Timing ---
        trunk_t = torch.from_numpy(make_monitor_trunk_1d(nx, monitor_x, pe_x, pe_cross)).to(device).float()
        with torch.no_grad():
            results["Single x forward pass"] = measure_stats(lambda: net((X_t, trunk_t)), trials, device_str)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", help="Comma-separated run dirs")
    p.add_argument("--trials", type=int, default=100, help="Number of repeats to average")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], 
                   help="Device to run on (cpu or cuda)")
    args = p.parse_args()

    dirs = [d.strip() for d in args.run_dirs.split(",") if d.strip()]

    print(f"Mode: {args.device.upper()}")
    print(f"Trials per model: {args.trials}")

    for rd in dirs:
        print(f"\n--- {rd} ---")
        try:
            results = run_timing_unified(rd, args.device, args.trials)
            
            for metric, (mu, std) in results.items():
                print(f"{metric:<25}: {mu*1000:.4f} ± {std*1000:.4f} ms")
                
        except Exception as e:
            print(f"[ERROR] {rd}: {e}")
            traceback.print_exc()

if __name__=="__main__":
    main()
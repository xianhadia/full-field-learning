import argparse
import os
import csv
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

SEED = 12345
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Global Grid Config
ny, nx = 38, 840
monitor_x = 120 - 1
_monitor_idx = (np.arange(ny, dtype=np.int64) * nx + monitor_x).astype(np.int64)

def _to_torch(y):
    if isinstance(y, np.ndarray):
        return torch.from_numpy(y).to(device).float(), "np"
    if torch.is_tensor(y):
        return y.to(device).float(), "torch"
    raise TypeError(f"Unsupported type: {type(y)}")

def make_run_dir(dim_str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("output", f"{ts}_{dim_str}")
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    return base_dir

# ----------------- Data Loading -----------------
def load_dataset_1d(path, n, assume_already_1d=False):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        X = obj["X"][:n].astype(np.float32)
        y_complex = obj["y_Ez"][:n]
        if y_complex.ndim == 2 and y_complex.shape[1] == ny * nx:
            y_r = np.real(y_complex).reshape(-1, ny, nx)
            y_i = np.imag(y_complex).reshape(-1, ny, nx)
        elif y_complex.ndim == 3 and y_complex.shape[1:] == (ny, nx):
            y_r = np.real(y_complex)
            y_i = np.imag(y_complex)
        else:
            raise ValueError(f"Unexpected y_Ez shape for NPZ: {y_complex.shape}")
        Ez_r = y_r.mean(axis=1).astype(np.float32)
        Ez_i = y_i.mean(axis=1).astype(np.float32)
        y = np.stack([Ez_r, Ez_i], axis=-1)
        return X, y.astype(np.float32)
    
    data = obj.item()
    X = data["params"][:n].astype(np.float32)
    Ez = data["Ez"][:n]
    if assume_already_1d:
        if Ez.ndim == 2:
            Ez_r = Ez.astype(np.float32)
            Ez_i = np.zeros_like(Ez_r, dtype=np.float32)
        elif Ez.ndim == 3 and Ez.shape[-1] == 2:
            Ez_r = Ez[..., 0].astype(np.float32)
            Ez_i = Ez[..., 1].astype(np.float32)
        else:
            raise ValueError(f"assume_already_1d=True but Ez shape is {Ez.shape}")
    else:
        if Ez.ndim == 3:
            Ez_r_2d = Ez.astype(np.float32)
            Ez_i_2d = np.zeros_like(Ez_r_2d, dtype=np.float32)
        elif Ez.ndim == 4 and Ez.shape[-1] == 2:
            Ez_r_2d = Ez[..., 0].astype(np.float32)
            Ez_i_2d = Ez[..., 1].astype(np.float32)
        else:
            raise ValueError(f"Ez shape is {Ez.shape}")
        Ez_r = Ez_r_2d.mean(axis=1)
        Ez_i = Ez_i_2d.mean(axis=1)
    y = np.stack([Ez_r, Ez_i], axis=-1)
    return X, y.astype(np.float32)

def load_dataset_2d(path, n):
    d = np.load(path, allow_pickle=True)
    X = d["X"][:n]
    y_complex = d["y_Ez"][:n]
    y_r = np.real(y_complex).reshape(-1, ny, nx)
    y_i = np.imag(y_complex).reshape(-1, ny, nx)
    y = np.stack([y_r, y_i], axis=-1).reshape(-1, ny * nx, 2)
    return X.astype(np.float32), y.astype(np.float32)

# ----------------- Stats & Standardizing -----------------
def compute_y_channel_stats(y_full):
    # Works for both 1D and 2D since it flattens the spatial dimension
    mean = y_full.reshape(-1, 2).mean(axis=0, keepdims=True)
    std = y_full.reshape(-1, 2).std(axis=0, keepdims=True) + 1e-12
    return mean.reshape(1, 1, 2).astype(np.float32), std.reshape(1, 1, 2).astype(np.float32)

def standardize_y(y, mean, std):
    return (y - mean) / std

def complex_abs_loss(y_true, y_pred):
    yt, _ = _to_torch(y_true)
    yp, _ = _to_torch(y_pred)
    diff2 = (yp[..., 0] - yt[..., 0]).pow(2) + (yp[..., 1] - yt[..., 1]).pow(2)
    return torch.mean(torch.sqrt(diff2 + 1e-20))

# ----------------- Evaluation Metrics -----------------
def fractional_error_1d(y_true_full_1d_phys, y_pred_monitor_phys):
    yt, _ = _to_torch(y_true_full_1d_phys)
    T = yt[:, monitor_x, :]
    P, _ = _to_torch(y_pred_monitor_phys)
    if P.dim() == 3:
        P = P.squeeze(1)
    Pvec = P.reshape(-1)
    Tvec = T.reshape(-1)
    num = torch.linalg.norm(Pvec - Tvec, ord=2)
    den = torch.linalg.norm(Tvec, ord=2) + 1e-12
    return num / den

def fractional_error_2d(y_true_full_phys, y_pred_monitor_phys):
    yt, _ = _to_torch(y_true_full_phys)
    B = yt.shape[0]
    yt = yt.view(B, ny * nx, 2)
    line = yt[:, _monitor_idx, :]
    T = line.mean(dim=1)
    P, _ = _to_torch(y_pred_monitor_phys)
    if P.dim() == 3:
        P = P.mean(dim=1)
    Pvec = P.reshape(-1)
    Tvec = T.reshape(-1)
    num = torch.linalg.norm(Pvec - Tvec, ord=2)
    den = torch.linalg.norm(Tvec, ord=2)
    return num / den

# ----------------- Features -----------------
def get_act(name: str):
    n = name.lower()
    if n == "relu": return nn.ReLU()
    if n == "gelu": return nn.GELU()
    if n == "tanh": return nn.Tanh()
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

# 1D Trunks
def make_trunk_features_1d(nx, n_harm_x=16, n_cross=0):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    cols = [np.ones((nx, 1), dtype=np.float32), x[:, None]]
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

# 2D Trunks
def make_trunk_features_2d(nx, ny, n_harm_x=16, n_harm_y=4, n_cross=2):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    Xf = Xg.reshape(-1)
    Yf = Yg.reshape(-1)
    N = Xf.shape[0]
    cols = [np.ones((N, 1), dtype=np.float32), Xf[:, None], Yf[:, None]]
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

# ----------------- Model -----------------
class MergeDeepONet(dde.nn.pytorch.NN):
    def __init__(self, branch_layers, trunk_layers, merger_layers, 
                 activation="relu", trunk_chunk=4096, batch_chunk=0, amp=False):
        super().__init__()
        self.trunk_chunk = int(trunk_chunk)
        self.batch_chunk = int(batch_chunk)
        self.amp = bool(amp)
        self.branch_fnn = fnn(branch_layers, act=activation)
        self.trunk_fnn = fnn(trunk_layers, act=activation)
        
        b_out = branch_layers[-1]
        t_out = trunk_layers[-1]
        if b_out != t_out:
            raise ValueError(f"Branch and trunk output dims must match. Got branch={b_out}, trunk={t_out}.")
        if merger_layers[0] != b_out:
            raise ValueError(f"Merger input dim must equal branch/trunk output dim {b_out}. Got {merger_layers[0]}.")
        if merger_layers[-1] != 2:
            raise ValueError("Merger must end with size 2 (Re, Im).")
        
        self.merger = fnn(merger_layers, act=activation, last_act=None)

    def _merge_block(self, b_block, t_block):
        Bb = b_block.shape[0]
        nC = t_block.shape[0]
        b_exp = b_block[:, None, :].expand(Bb, nC, -1)
        t_exp = t_block[None, :, :].expand(Bb, nC, -1)
        fused = b_exp * t_exp
        merged = self.merger(fused.reshape(Bb * nC, -1))
        return merged.view(Bb, nC, 2)

    def forward(self, inputs):
        branch_batch, trunk_points = inputs
        with torch.cuda.amp.autocast(enabled=self.amp):
            B = branch_batch.shape[0]
            N = trunk_points.shape[0]
            b_all = self.branch_fnn(branch_batch)
            t_all = self.trunk_fnn(trunk_points)
            outs_B = []
            bstep = max(1, self.batch_chunk) if self.batch_chunk > 0 else B
            for bs in range(0, B, bstep):
                be = min(B, bs + bstep)
                b_block = b_all[bs:be]
                outs_N = []
                nstep = max(1, self.trunk_chunk)
                for ns in range(0, N, nstep):
                    ne = min(N, ns + nstep)
                    t_block = t_all[ns:ne]
                    out_block = self._merge_block(b_block, t_block)
                    outs_N.append(out_block)
                outs_B.append(torch.cat(outs_N, dim=1))
            out = torch.cat(outs_B, dim=0)
        return out

# ----------------- Callback -----------------
class SaveBestCallback(dde.callbacks.Callback):
    def __init__(self, dim, X_train, y_train_full_phys, X_val, y_val_full_phys,
                 trunk_monitor, trunk_full, base_dir, eval_interval=250,
                 batch_bs=32, y_mean=None, y_std=None, start_eval_iter=0):
        super().__init__()
        self.dim_mode = dim
        self.X_train = X_train
        self.y_train_full_phys = y_train_full_phys
        self.X_val = X_val
        self.y_val_full_phys = y_val_full_phys
        self.trunk_monitor = trunk_monitor
        self.trunk_full = trunk_full
        self.eval_interval = eval_interval
        self.best_fe = float("inf")
        self.base_dir = base_dir
        self.model_dir = os.path.join(self.base_dir, "models")
        self.batch_bs = batch_bs
        self.start_eval_iter = start_eval_iter
        os.makedirs(self.model_dir, exist_ok=True)
        self.y_mean = (torch.from_numpy(y_mean).to(device).float() if isinstance(y_mean, np.ndarray) else y_mean)
        self.y_std = (torch.from_numpy(y_std).to(device).float() if isinstance(y_std, np.ndarray) else y_std)
        self.log_csv = os.path.join(self.base_dir, "training_log.csv")
        with open(self.log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "train_FE", "val_FE", "best_mark"])

    def _predict_over_set(self, Xb, trunk_points, batch_bs):
        B = Xb.shape[0]
        preds = []
        with torch.no_grad():
            for s in range(0, B, batch_bs):
                e = min(B, s + batch_bs)
                yhat_std = self.model.predict((Xb[s:e], trunk_points))
                preds.append(yhat_std)
                torch.cuda.empty_cache()
        return np.concatenate(preds, axis=0)

    def _eval_fe_global(self, Xb, ytrue_full_phys):
        y_pred_monitor_std = self._predict_over_set(Xb, self.trunk_monitor, self.batch_bs)
        yp_std_t, _ = _to_torch(y_pred_monitor_std)
        
        if self.dim_mode == "1d":
            y_mean_vec = self.y_mean.view(1, 1, 2)
            y_std_vec = self.y_std.view(1, 1, 2)
            P_phys = yp_std_t * y_std_vec + y_mean_vec
            fe = fractional_error_1d(ytrue_full_phys, P_phys)
        else:
            P_std_mean = yp_std_t.mean(dim=1, keepdim=False)
            y_mean_vec = self.y_mean.view(1, 2)
            y_std_vec = self.y_std.view(1, 2)
            P_phys = P_std_mean * y_std_vec + y_mean_vec
            fe = fractional_error_2d(ytrue_full_phys, P_phys)
            
        return fe.detach().cpu().item()

    def on_epoch_end(self):
        step = int(self.model.train_state.step)
        lt = self.model.train_state.loss_train
        train_loss = float(np.array(lt).reshape(-1)[0])
        if step >= self.start_eval_iter and step % self.eval_interval == 0:
            train_fe = self._eval_fe_global(self.X_train, self.y_train_full_phys)
            val_fe = self._eval_fe_global(self.X_val, self.y_val_full_phys)
            best_mark = ""
            if val_fe < self.best_fe:
                self.best_fe = val_fe
                best_mark = "BEST"
                tmp = os.path.join(self.model_dir, "best_model.pt")
                self.model.save(tmp)
                appended = tmp + f"-{step}.pt"
                if os.path.exists(appended):
                    try:
                        os.replace(appended, os.path.join(self.model_dir, "best_model.pt"))
                    except Exception:
                        pass
            ckpt = os.path.join(self.model_dir, f"model_step_{step}.pt")
            self.model.save(ckpt)
            with open(self.log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([step, f"{train_loss:.6e}", f"{train_fe:.6f}", f"{val_fe:.6f}", best_mark])
            print(f"[Eval @ {step}] train_loss={train_loss:.3e} | train_FE={train_fe:.6f} | val_FE={val_fe:.6f} {best_mark}")
        else:
            with open(self.log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([step, f"{train_loss:.6e}", "", "", ""])

    def on_train_end(self):
        step = int(self.model.train_state.step)
        tmp = os.path.join(self.model_dir, "final_model.pt")
        self.model.save(tmp)
        appended = tmp + f"-{step}.pt"
        if os.path.exists(appended):
            try:
                os.replace(appended, os.path.join(self.model_dir, "final_model.pt"))
            except Exception:
                pass
        print("Training finished. Final model saved.")

# ----------------- Main Flow -----------------
def build_argparser():
    p = argparse.ArgumentParser()
    # Mode selection
    p.add_argument("--dim", type=str, required=True, choices=["1d", "2d"], help="Select model dimensionality.")
    
    # Dataset parsing
    p.add_argument("-train", type=int, required=True)
    p.add_argument("-val", type=int, required=True)
    p.add_argument("-test", type=int, required=True)
    p.add_argument("--train_file", default="../data/Maxwell_Train.npz")
    p.add_argument("--val_file", default="../data/Maxwell_Val.npz")
    p.add_argument("--test_file", default="../data/Maxwell_Test.npz")
    
    # 1D specific
    p.add_argument("--assume_already_1d", action="store_true")
    
    # PE params
    p.add_argument("--pe_x", type=int, default=16) # changed default to 16 to match 1D original, specify overrides in terminal 
    p.add_argument("--pe_y", type=int, default=8)
    p.add_argument("--pe_cross", type=int, default=0)
    
    # Strides
    p.add_argument("--stride_x_train", type=int, default=1)
    p.add_argument("--stride_y_train", type=int, default=2)
    
    # Model
    p.add_argument("--branch_hidden", type=int, nargs="+", default=[512, 512, 512, 1024])
    p.add_argument("--trunk_hidden", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--merger_hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--activation", default="silu", choices=["relu", "gelu", "tanh", "silu"])
    
    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_model", choices=["best", "last"], default="best")
    p.add_argument("--trunk_chunk", type=int, default=256)
    p.add_argument("--batch_chunk", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--start_eval_iter", type=int, default=500)
    return p

def train_and_evaluate(args):
    # Setup dataset processing based on dimension
    if args.dim == "1d":
        X_train_raw, y_train_full_phys = load_dataset_1d(args.train_file, args.train, args.assume_already_1d)
        X_val_raw, y_val_full_phys = load_dataset_1d(args.val_file, args.val, args.assume_already_1d)
        X_test_raw, y_test_full_phys = load_dataset_1d(args.test_file, args.test, args.assume_already_1d)
    else:
        X_train_raw, y_train_full_phys = load_dataset_2d(args.train_file, args.train)
        X_val_raw, y_val_full_phys = load_dataset_2d(args.val_file, args.val)
        X_test_raw, y_test_full_phys = load_dataset_2d(args.test_file, args.test)

    # Standardization
    X_mean = X_train_raw.mean(axis=0, keepdims=True)
    X_std = X_train_raw.std(axis=0, keepdims=True) + 1e-12
    X_train = (X_train_raw - X_mean) / X_std
    X_val = (X_val_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std
    
    y_mean, y_std = compute_y_channel_stats(y_train_full_phys)
    
    m = X_train.shape[1]
    
    # Branch based on Dimension
    if args.dim == "1d":
        trunk_full = make_trunk_features_1d(nx, args.pe_x, args.pe_cross)
        
        x_idx = np.arange(0, nx, args.stride_x_train)
        if monitor_x not in x_idx:
            x_idx = np.unique(np.concatenate([x_idx, np.array([monitor_x])], axis=0))
            
        trunk_train = trunk_full[x_idx]
        y_train_sub_phys = y_train_full_phys[:, x_idx, :]
        
        trunk_monitor = make_monitor_trunk_1d(nx, monitor_x, args.pe_x, args.pe_cross)
        y_test_lite_phys = y_train_full_phys[:1, monitor_x:monitor_x+1, :].copy()
        
    else:
        trunk_full = make_trunk_features_2d(nx, ny, args.pe_x, args.pe_y, args.pe_cross)
        
        y_idx = np.arange(0, ny, args.stride_y_train)
        x_idx = np.arange(0, nx, args.stride_x_train)
        yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
        idx_subsample = (yy * nx + xx).reshape(-1)
        idx_train = np.unique(np.concatenate([idx_subsample, _monitor_idx], axis=0))
        
        trunk_train = trunk_full[idx_train]
        y_train_sub_phys = y_train_full_phys[:, idx_train, :]
        
        trunk_monitor = make_monitor_trunk_2d(nx, ny, monitor_x, args.pe_x, args.pe_y, args.pe_cross)
        y_test_lite_phys = y_train_full_phys[:1, _monitor_idx[:1], :].copy()
    
    dim_trunk = trunk_full.shape[1]
    y_train_sub = standardize_y(y_train_sub_phys, y_mean, y_std)
    
    X_test_lite_branch = X_train[:1].copy()
    X_test_lite_trunk = trunk_monitor[:1].copy()
    y_test_lite = standardize_y(y_test_lite_phys, y_mean, y_std)
    
    # Define DDE Data
    data = dde.data.TripleCartesianProd(
        X_train=(X_train, trunk_train),
        y_train=y_train_sub,
        X_test=(X_test_lite_branch, X_test_lite_trunk),
        y_test=y_test_lite
    )
    data.train_batch_size = args.batch
    
    branch_layers = [m] + args.branch_hidden
    trunk_layers = [dim_trunk] + args.trunk_hidden
    if branch_layers[-1] != trunk_layers[-1]:
        raise ValueError(f"Branch/trunk hidden must end with same dim. Got branch_out={branch_layers[-1]}, trunk_out={trunk_layers[-1]}.")
    merger_layers = [branch_layers[-1]] + args.merger_hidden + [2]
    
    base_dir = make_run_dir(args.dim)
    model_dir = os.path.join(base_dir, "models")
    
    # Save Architecture Setup
    arch_txt = os.path.join(base_dir, "architecture.txt")
    with open(arch_txt, "w") as f:
        f.write(f"MergeDeepONet {args.dim.upper()} Architecture\n")
        f.write(f"Datetime: {datetime.now().isoformat()}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Grid: ny={ny}, nx={nx}, monitor_x={monitor_x}\n")
        f.write(f"Harmonics: pe_x={args.pe_x}, pe_y={args.pe_y}, pe_cross={args.pe_cross}\n")
        f.write(f"Stride: stride_x_train={args.stride_x_train}, stride_y_train={args.stride_y_train}\n\n")
        f.write(f"Branch layers: {branch_layers}\n")
        f.write(f"Trunk layers: {trunk_layers}\n")
        f.write(f"Merger layers: {merger_layers}\n")
        f.write(f"Chunks: trunk_chunk={args.trunk_chunk}, batch_chunk={args.batch_chunk}\n")
        f.write(f"AMP: {args.amp}\n\n")
        f.write(f"Train/Val/Test: {args.train}/{args.val}/{args.test}\n")
        f.write(f"LR={args.lr}, batch={args.batch}, iters={args.iters}, eval_interval={args.eval_interval}\n")
        
    with open(os.path.join(base_dir, "args.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2)
        
    np.savez(
        os.path.join(base_dir, "standardization_stats.npz"),
        X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std
    )
    
    # Initialization
    net = MergeDeepONet(
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        merger_layers=merger_layers,
        activation=args.activation,
        trunk_chunk=args.trunk_chunk,
        batch_chunk=args.batch_chunk,
        amp=args.amp,
    ).to(device)
    model = dde.Model(data, net)
    model.compile("adam", lr=args.lr, loss=complex_abs_loss, metrics=[])
    
    callback = SaveBestCallback(
        dim=args.dim,
        X_train=X_train, y_train_full_phys=y_train_full_phys,
        X_val=X_val, y_val_full_phys=y_val_full_phys,
        trunk_monitor=trunk_monitor,
        trunk_full=trunk_full,
        base_dir=base_dir,
        eval_interval=args.eval_interval,
        batch_bs=args.eval_batch,
        y_mean=y_mean,
        y_std=y_std,
        start_eval_iter=args.start_eval_iter
    )
    
    print(f"Training MergeDeepONet {args.dim.upper()} on device: {device}")
    losshistory, train_state = model.train(
        iterations=args.iters,
        callbacks=[callback],
        batch_size=args.batch,
        display_every=args.eval_interval,
    )
    print("Training complete!")

    # Evaluation phase
    def build_eval_model():
        eval_net = MergeDeepONet(
            branch_layers=branch_layers,
            trunk_layers=trunk_layers,
            merger_layers=merger_layers,
            activation=args.activation,
            trunk_chunk=args.trunk_chunk,
            batch_chunk=args.batch_chunk,
            amp=args.amp,
        ).to(device)
        eval_model = dde.Model(
            dde.data.TripleCartesianProd(
                X_train=(X_train, trunk_train),
                y_train=standardize_y(y_train_sub_phys, y_mean, y_std),
                X_test=(X_val, trunk_full),
                y_test=y_val_full_phys
            ),
            eval_net
        )
        eval_model.compile("adam", lr=args.lr, loss=complex_abs_loss, metrics=[])
        return eval_model

    best_path = os.path.join(model_dir, "best_model.pt")
    final_path = os.path.join(model_dir, "final_model.pt")
    
    if args.eval_model == "best" and os.path.exists(best_path):
        print("Evaluating BEST model on test set...")
        eval_model = build_eval_model()
        eval_model.restore(best_path, verbose=1)
        model_type = "Best"
    else:
        print("Evaluating FINAL model on test set...")
        eval_model = build_eval_model()
        eval_model.restore(final_path, verbose=1)
        model_type = "Final"

    if args.dim == "1d":
        trunk_eval = make_monitor_trunk_1d(nx, monitor_x, args.pe_x, args.pe_cross)
    else:
        trunk_eval = make_monitor_trunk_2d(nx, ny, monitor_x, args.pe_x, args.pe_y, args.pe_cross)

    with torch.no_grad():
        B = X_test.shape[0]
        preds_std = []
        for s in range(0, B, args.eval_batch):
            e = min(B, s + args.eval_batch)
            ypm_std = eval_model.predict((X_test[s:e], trunk_eval))
            preds_std.append(ypm_std)
            torch.cuda.empty_cache()
            
        y_pred_monitor_std = np.concatenate(preds_std, axis=0)
        ypm_std_t, _ = _to_torch(y_pred_monitor_std)
        
        if args.dim == "1d":
            y_mean_t = torch.from_numpy(y_mean).to(device).float()
            y_std_t = torch.from_numpy(y_std).to(device).float()
            P_phys = ypm_std_t * y_std_t + y_mean_t
            test_fe = fractional_error_1d(y_test_full_phys, P_phys).detach().cpu().item()
        else:
            P_std_mean = ypm_std_t.mean(dim=1)
            y_mean_t = torch.from_numpy(y_mean).to(device).float().view(1, 2)
            y_std_t = torch.from_numpy(y_std).to(device).float().view(1, 2)
            P_phys = P_std_mean * y_std_t + y_mean_t
            test_fe = fractional_error_2d(y_test_full_phys, P_phys).detach().cpu().item()

    summary_path = os.path.join(base_dir, "final_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"MergeDeepONet {args.dim.upper()} Results\n")
        f.write(f"Train/Val/Test: {args.train}/{args.val}/{args.test}\n")
        f.write(f"{model_type} Model Test Fractional Error: {test_fe:.6f}\n")
        f.write(f"Training log: {os.path.join(base_dir, 'training_log.csv')}\n")
        f.write(f"Models dir: {model_dir}\n")

    print(f"{model_type} Model Test Fractional Error @ monitor: {test_fe:.6f}")
    print(f"All artifacts saved to: {base_dir}")

def main():
    parser = build_argparser()
    args = parser.parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main()
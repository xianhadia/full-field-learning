import argparse
import os
import csv
import json
import uuid
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import deepxde as dde
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

ny, nx = 38, 840  # Defaults; overwritten upon data load

# ==========================================
# Helpers & Features
# ==========================================
def _to_torch(y):
    if isinstance(y, np.ndarray):
        return torch.from_numpy(y).to(device).float(), "np"
    if torch.is_tensor(y):
        return y.to(device).float(), "torch"
    raise TypeError(f"Unsupported type: {type(y)}")

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

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def make_trunk_features_1d(ny, n_harm_y=16):
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    cols = [np.ones((ny, 1), dtype=np.float32), y[:, None]]
    for n in range(1, n_harm_y + 1):
        w = 2 * np.pi * n
        cols += [np.cos(w * y)[:, None], np.sin(w * y)[:, None]]
    return np.concatenate(cols, axis=1).astype(np.float32)

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

# ==========================================
# Physics & Metrics
# ==========================================
def _generate_pores_numpy(p_arr, resolution, lowval=0.1):
    param_dim = p_arr.size
    n = int(math.sqrt(param_dim))
    Nblk = resolution // n
    pores = np.zeros((resolution, resolution), dtype=float)
    k = 0
    for i in range(n):
        for j in range(n):
            p = p_arr[k]
            half  = p * Nblk / 2
            inner = math.floor(half)
            w     = np.clip(half - inner, 0.0, 1.0)
            if Nblk % 2 == 0:
                isqa = Nblk//2 - inner
                isqb = Nblk//2 + inner
            else:
                isqa = Nblk//2 - inner + 1
                isqb = Nblk//2 + inner
            A1 = np.ones((Nblk, Nblk))
            A1[isqa: isqb, isqa: isqb] = lowval
            if isqa == 0 and w == 0:
                block = A1
            elif isqa == 0:
                block = A1 * (1.0 - w)
            else:
                A2 = np.ones((Nblk, Nblk))
                lo = max(isqa - 1, 0)
                hi = min(isqb + 1, Nblk)
                A2[lo:hi, lo:hi] = lowval
                block = A1 * (1.0 - w) + A2 * w
            pores[i*Nblk:(i+1)*Nblk, j*Nblk:(j+1)*Nblk] = block
            k += 1
    return pores

def compute_kappa_numpy(X_params_batch, T_batch):
    B = T_batch.shape[0]
    resolution = nx + 1
    iline = resolution // 2
    kappas = np.zeros(B, dtype=np.float32)
    for b in range(B):
        pores = _generate_pores_numpy(X_params_batch[b], resolution)
        c1 = 0.5 * (pores[:-1, :-1] + pores[1:, :-1])
        dT = T_batch[b, iline, :] - T_batch[b, iline - 1, :]
        kappas[b] = np.sum(c1[iline, :] * dT)
    return kappas

def mse_loss(y_true, y_pred):
    yt, _ = _to_torch(y_true)
    yp, _ = _to_torch(y_pred)
    if yt.ndim == 2: yt = yt.unsqueeze(-1)
    if yp.ndim == 2: yp = yp.unsqueeze(-1)
    return torch.mean((yp - yt) ** 2)

def global_flux_fe(y_true_full_vec, y_pred, X_params_raw, dim, y_mean=None, y_std=None):
    if y_true_full_vec.ndim == 3:
        Tt = y_true_full_vec[..., 0]
    else:
        Tt = y_true_full_vec
        
    B = Tt.shape[0]
    Tt_img = Tt.reshape(B, ny, nx)

    if torch.is_tensor(y_pred):
        yp = y_pred.detach().cpu().numpy()
    else:
        yp = y_pred

    if dim == "1d":
        if isinstance(y_mean, torch.Tensor):
            y_mean_np = y_mean.detach().cpu().numpy()
            y_std_np  = y_std.detach().cpu().numpy()
        else:
            y_mean_np, y_std_np = y_mean, y_std
            
        P1d_phys = yp * y_std_np + y_mean_np
        P2d_phys = np.repeat(P1d_phys, nx, axis=1).reshape(B, ny, nx)
        flux_p = compute_kappa_numpy(X_params_raw, P2d_phys)
    else:
        if yp.ndim == 3:
            Tp_img = yp[..., 0].reshape(B, ny, nx)
        else:
            Tp_img = yp.reshape(B, ny, nx)
        flux_p = compute_kappa_numpy(X_params_raw, Tp_img)

    flux_t = compute_kappa_numpy(X_params_raw, Tt_img)

    Tvec = torch.from_numpy(flux_t).float()
    Pvec = torch.from_numpy(flux_p).float()
    num = torch.linalg.norm(Pvec - Tvec, ord=2)
    den = torch.linalg.norm(Tvec, ord=2)
    return float((num / den).item())

# ==========================================
# Network Architecture
# ==========================================
class MergeDeepONet(dde.nn.pytorch.NN):
    def __init__(self, branch_layers, trunk_layers, merger_layers,
                 activation="relu", trunk_chunk=4096, batch_chunk=0, amp=False):
        super().__init__()
        self.trunk_chunk = int(trunk_chunk)
        self.batch_chunk = int(batch_chunk)
        self.amp = bool(amp)
        self.branch_fnn = fnn(branch_layers, act=activation)
        self.trunk_fnn  = fnn(trunk_layers,  act=activation)

        b_out = branch_layers[-1]
        t_out = trunk_layers[-1]
        if b_out != t_out:
            raise ValueError(f"Branch and trunk output dims must match. Got branch={b_out}, trunk={t_out}.")
        if merger_layers[0] != b_out:
            raise ValueError(f"Merger input dim must equal branch/trunk output dim {b_out}. Got {merger_layers[0]}.")
        if merger_layers[-1] != 1:
            raise ValueError("Merger must end with size 1.")
        self.merger = fnn(merger_layers, act=activation, last_act=None)

    def _merge_block(self, b_block, t_block):
        Bb = b_block.shape[0]
        nC = t_block.shape[0]
        b_exp = b_block[:, None, :].expand(Bb, nC, -1)
        t_exp = t_block[None, :, :].expand(Bb, nC, -1)
        fused = b_exp * t_exp
        merged = self.merger(fused.reshape(Bb * nC, -1))
        return merged.view(Bb, nC, 1)

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

# ==========================================
# Callback
# ==========================================
class UnifiedSaveBestCallback(dde.callbacks.Callback):
    def __init__(self, X_train_raw, y_train_full2d, X_val_raw, y_val_full2d,
                 trunk_full, base_dir, dim, eval_interval=1000, eval_start_after=10000,
                 X_train_n=None, X_val_n=None, y_mean=None, y_std=None):
        super().__init__()
        self.X_train_raw = X_train_raw
        self.X_val_raw   = X_val_raw
        self.X_train_n   = X_train_n
        self.X_val_n     = X_val_n
        self.y_train_full2d = y_train_full2d
        self.y_val_full2d   = y_val_full2d
        self.trunk_full     = trunk_full
        self.dim = dim
        self.y_mean = y_mean
        self.y_std = y_std
        self.eval_interval = eval_interval
        self.eval_start_after = eval_start_after
        self.best_fe = float('inf')
        self.base_dir = base_dir
        self.model_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_csv = os.path.join(self.base_dir, "training_log.csv")
        with open(self.log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "train_FE", "val_FE", "best_mark"])

    def _predict_full(self, Xb_n, batch_bs=32):
        B = Xb_n.shape[0]
        preds = []
        with torch.no_grad():
            for s in range(0, B, batch_bs):
                e = min(B, s + batch_bs)
                yp = self.model.predict((Xb_n[s:e], self.trunk_full))
                preds.append(yp)
                torch.cuda.empty_cache()
        return np.concatenate(preds, axis=0)

    def on_epoch_end(self):
        step = int(self.model.train_state.step)
        lt = self.model.train_state.loss_train
        train_loss = float(np.array(lt).reshape(-1)[0])
        if step % self.eval_interval == 0:
            if step >= self.eval_start_after:
                Xtr_n = self.X_train_n if self.X_train_n is not None else self.model.data.X_train[0]
                Xva_n = self.X_val_n   if self.X_val_n   is not None else self.model.data.X_test[0]
                
                y_pred_train = self._predict_full(Xtr_n)
                train_fe = global_flux_fe(
                    self.y_train_full2d, y_pred_train, self.X_train_raw, self.dim, self.y_mean, self.y_std
                )
                y_pred_val = self._predict_full(Xva_n)
                val_fe = global_flux_fe(
                    self.y_val_full2d, y_pred_val, self.X_val_raw, self.dim, self.y_mean, self.y_std
                )
                
                best_mark = ""
                if val_fe < self.best_fe:
                    self.best_fe = val_fe
                    best_mark = "BEST"
                    tmp = os.path.join(self.model_dir, "best_model.pt")
                    self.model.save(tmp)
                    appended = tmp + f"-{step}.pt"
                    if os.path.exists(appended):
                        try:
                            os.replace(appended, tmp)
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
                os.replace(appended, tmp)
            except Exception:
                pass
        print("Training finished. Final model saved.")

# ==========================================
# Data IO
# ==========================================
def load_diffusion_data(x_path, y_path):
    loaded_X = np.load(x_path, allow_pickle=True)
    X_raw = loaded_X[loaded_X.files[0]] if isinstance(loaded_X, np.lib.npyio.NpzFile) else loaded_X
    X = X_raw.T.astype(np.float32)
    Y_npz = np.load(y_path)
    T_fourier = Y_npz["T"]
    flux_all  = Y_npz["flux"].astype(np.float32)
    ny_, nx_, total = T_fourier.shape
    T_all = np.transpose(T_fourier, (2, 0, 1)).astype(np.float32)
    return X, T_all, flux_all, ny_, nx_, total

def split_fixed(X, T_all, flux_all, N):
    return (X[N:2*N], X[0:N], X[-N:]), (T_all[N:2*N], T_all[0:N], T_all[-N:]), (flux_all[N:2*N], flux_all[0:N], flux_all[-N:])

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=str, choices=["1d", "2d"], default="2d", help="Train 1D or 2D model")
    p.add_argument("--x_path", default="../data/X_fourier25.npz")
    p.add_argument("--y_path", default="../data/Y_fourier25.npz")
    p.add_argument("--N", type=int, default=1088)
    
    # Harmonics and strides
    p.add_argument("--pe_x", type=int, default=4)
    p.add_argument("--pe_y", type=int, default=4)
    p.add_argument("--pe_cross", type=int, default=1)
    p.add_argument("--stride_x_train", type=int, default=4)
    p.add_argument("--stride_y_train", type=int, default=2)
    
    # Network
    p.add_argument("--branch_hidden", type=int, nargs="+", default=[512*2, 512*2, 512*2])
    p.add_argument("--trunk_hidden",  type=int, nargs="+", default=[256*2, 256*2, 256*2, 512*2])
    p.add_argument("--merger_hidden", type=int, nargs="+", default=[512*2, 256*2])
    p.add_argument("--activation", default="relu", choices=["relu","gelu","tanh","silu"])
    
    # Training Params
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_start_after", type=int, default=10000)
    p.add_argument("--trunk_chunk", type=int, default=4096)
    p.add_argument("--batch_chunk", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    return p

def main():
    global ny, nx
    args = build_argparser().parse_args()

    # Create run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    shortid = uuid.uuid4().hex[:6]
    run_id = f"{ts}_{args.dim.upper()}__N{args.N}__{shortid}"
    base_dir = os.path.join("output", run_id)
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load & Split
    X_all_raw, T_all, flux_all, ny, nx, total_samples = load_diffusion_data(args.x_path, args.y_path)
    (X_train_raw, X_val_raw, X_test_raw), (T_train, T_val, T_test), _ = split_fixed(X_all_raw, T_all, flux_all, args.N)

    # Normalize X
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std  = X_train_raw.std(axis=0, keepdims=True) + 1e-12
    X_train_n = (X_train_raw - mean) / std
    X_val_n   = (X_val_raw   - mean) / std
    X_test_n  = (X_test_raw  - mean) / std

    # Full 2D targets (for Evaluation logic consistency across both modes)
    Npts = ny * nx
    y_train_full2d = T_train.reshape((-1, Npts, 1)).astype(np.float32)
    y_val_full2d   = T_val.reshape((-1, Npts, 1)).astype(np.float32)
    y_test_full2d  = T_test.reshape((-1, Npts, 1)).astype(np.float32)

    # Setup 1D vs 2D Target Variables & Trunk
    y_mean, y_std = None, None
    if args.dim == "2d":
        trunk_full = make_trunk_features_2d(nx, ny, args.pe_x, args.pe_y, args.pe_cross)
        
        y_train_full = y_train_full2d
        y_val_full   = y_val_full2d
        y_test_full  = y_test_full2d

        # Subsample along both axes
        y_idx = np.arange(0, ny, args.stride_y_train)
        x_idx = np.arange(0, nx, args.stride_x_train)
        yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
        idx_train = (yy * nx + xx).reshape(-1)
        trunk_train = trunk_full[idx_train]
        y_train_sub = y_train_full[:, idx_train, :]

    else: # 1D
        trunk_full = make_trunk_features_1d(ny, args.pe_y)
        
        # 1D targets
        y_train_1d = T_train.mean(axis=2).astype(np.float32)[:, :, None]
        y_val_1d   = T_val.mean(axis=2).astype(np.float32)[:, :, None]
        y_test_1d  = T_test.mean(axis=2).astype(np.float32)[:, :, None]

        y_mean = y_train_1d.reshape(-1, 1).mean(axis=0, keepdims=True).astype(np.float32).reshape(1,1,1)
        y_std  = y_train_1d.reshape(-1, 1).std(axis=0, keepdims=True).astype(np.float32).reshape(1,1,1) + 1e-12

        y_train_full = (y_train_1d - y_mean) / y_std
        y_val_full   = (y_val_1d   - y_mean) / y_std
        y_test_full  = (y_test_1d  - y_mean) / y_std

        # Subsample along y only
        y_idx = np.arange(0, ny, args.stride_y_train)
        if y_idx.size == 0: y_idx = np.arange(ny)
        trunk_train = trunk_full[y_idx]
        y_train_sub = y_train_full[:, y_idx, :]

    dim_trunk = trunk_full.shape[1]

    # Data object
    data = dde.data.TripleCartesianProd(
        X_train=(X_train_n, trunk_train),
        y_train=y_train_sub,
        X_test=(X_train_n[:1].copy(), trunk_full[:1].copy()), # Dummy test block
        y_test=y_train_sub[:1, :1, :].copy(),
    )
    data.train_batch_size = args.batch

    # Network init
    branch_layers = [X_train_n.shape[1]] + args.branch_hidden
    trunk_layers  = [dim_trunk]  + args.trunk_hidden
    merger_layers = [branch_layers[-1]] + args.merger_hidden + [1]
    
    net = MergeDeepONet(
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        merger_layers=merger_layers,
        activation=args.activation,
        trunk_chunk=args.trunk_chunk,
        batch_chunk=args.batch_chunk,
        amp=args.amp,
    ).to(device)

    # Save hyperparams mapping to args.json to ensure test.py compatibility
    with open(os.path.join(base_dir, "args.json"), "w") as jf:
        json.dump(vars(args), jf, indent=2)

    # Train
    model = dde.Model(data, net)
    model.compile("adam", lr=args.lr, loss=mse_loss, metrics=[])

    callback = UnifiedSaveBestCallback(
        X_train_raw=X_train_raw, y_train_full2d=y_train_full2d,
        X_val_raw=X_val_raw,     y_val_full2d=y_val_full2d,
        trunk_full=trunk_full, dim=args.dim, base_dir=base_dir,
        eval_interval=args.eval_interval, eval_start_after=args.eval_start_after,
        X_train_n=X_train_n, X_val_n=X_val_n, y_mean=y_mean, y_std=y_std
    )

    print(f"Training MergeDeepONet ({args.dim.upper()} diffusion) on device: {device}")
    model.train(iterations=args.iters, callbacks=[callback], batch_size=args.batch, display_every=args.eval_interval)

    # Evaluate using DDE test object logic natively
    best_plain = os.path.join(model_dir, "best_model.pt")
    candidates = glob.glob(os.path.join(model_dir, "best_model.pt-*.pt"))
    def step_from_name(path): return int(os.path.basename(path).split("-", 1)[1].rsplit(".", 1)[0])
    
    if os.path.exists(best_plain): restore_path, model_type = best_plain, "Best"
    elif candidates: restore_path, model_type = sorted(candidates, key=step_from_name)[-1], "Best"
    else: restore_path, model_type = os.path.join(model_dir, "final_model.pt"), "Final"

    # Evaluation Model
    eval_net = MergeDeepONet(branch_layers, trunk_layers, merger_layers, args.activation, args.trunk_chunk, args.batch_chunk, args.amp).to(device)
    eval_model = dde.Model(dde.data.TripleCartesianProd(X_train=(X_train_n, trunk_full), y_train=y_train_full, X_test=(X_val_n, trunk_full), y_test=y_val_full), eval_net)
    eval_model.compile("adam", lr=args.lr, loss=mse_loss, metrics=[])
    eval_model.restore(restore_path, verbose=1)

    with torch.no_grad():
        B = X_test_n.shape[0]
        preds = []
        for s in range(0, B, 32):
            e = min(B, s + 32)
            preds.append(eval_model.predict((X_test_n[s:e], trunk_full)))
            torch.cuda.empty_cache()
        ypred_test = np.concatenate(preds, axis=0)

    test_fe = global_flux_fe(y_test_full2d, ypred_test, X_test_raw, args.dim, y_mean, y_std)

    with open(os.path.join(base_dir, "final_results.txt"), "w") as f:
        f.write(f"Model: MergeDeepONet ({args.dim.upper()} diffusion)\n")
        f.write(f"{model_type} Model Test FE: {test_fe:.6f}\n")

    print(f"{model_type} Model Test FE : {test_fe:.6f}")
    print(f"All artifacts saved to: {base_dir}")

if __name__ == "__main__":
    main()

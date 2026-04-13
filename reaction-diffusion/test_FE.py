import argparse
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Ensure Determinisim & Precision matches training script ---
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------- Features -----------------
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

# ----------------- Physics Logic -----------------
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

def compute_kappa_numpy(X_params_batch, T_batch, nx):
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

def global_flux_fe(y_true_full_vec, y_pred, X_params_raw, dim, ny, nx, y_mean=None, y_std=None):
    # Matches Train Script exactly
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
        flux_p = compute_kappa_numpy(X_params_raw, P2d_phys, nx)
    else:
        if yp.ndim == 3:
            Tp_img = yp[..., 0].reshape(B, ny, nx)
        else:
            Tp_img = yp.reshape(B, ny, nx)
        flux_p = compute_kappa_numpy(X_params_raw, Tp_img, nx)

    flux_t = compute_kappa_numpy(X_params_raw, Tt_img, nx)

    # Use torch norm to replicate training script's exact byte behavior
    Tvec = torch.from_numpy(flux_t).float()
    Pvec = torch.from_numpy(flux_p).float()
    num = torch.linalg.norm(Pvec - Tvec, ord=2)
    den = torch.linalg.norm(Tvec, ord=2)
    return float((num / den).item())

# ----------------- Model -----------------
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
        self.trunk_chunk = int(trunk_chunk)
        self.batch_chunk = int(batch_chunk)
        self.amp = bool(amp)
        self.branch_fnn = fnn(branch_layers, act=activation)
        self.trunk_fnn  = fnn(trunk_layers,  act=activation)
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


# ----------------- Main -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--eval_model", type=str, choices=["best", "final"], default="best")
    cmd_args = p.parse_args()

    with open(os.path.join(cmd_args.dir, "args.json"), "r") as f:
        m = json.load(f)

    X_dat = np.load(m["x_path"], allow_pickle=True)
    X_raw = (X_dat[X_dat.files[0]] if isinstance(X_dat, np.lib.npyio.NpzFile) else X_dat).T.astype(np.float32)
    Y_dat = np.load(m["y_path"])
    T_all = np.transpose(Y_dat["T"], (2, 0, 1)).astype(np.float32)
    ny, nx = Y_dat["T"].shape[0], Y_dat["T"].shape[1]

    N = m["N"]
    X_tr_raw, X_test_raw = X_raw[N:2*N], X_raw[-N:]
    T_test = T_all[-N:]

    # Match train script exactly for normalization
    X_m = X_tr_raw.mean(axis=0, keepdims=True)
    X_s = X_tr_raw.std(axis=0, keepdims=True) + 1e-12
    X_test_n = (X_test_raw - X_m) / X_s

    y_mean, y_std = None, None
    if m["dim"] == "1d":
        y_train_1d = T_all[N:2*N].mean(axis=2).astype(np.float32)[:, :, None]
        y_mean = y_train_1d.reshape(-1, 1).mean(axis=0, keepdims=True).astype(np.float32).reshape(1,1,1)
        y_std  = y_train_1d.reshape(-1, 1).std(axis=0, keepdims=True).astype(np.float32).reshape(1,1,1) + 1e-12
        trunk = make_trunk_features_1d(ny, m.get("pe_y", 4)) # Ensure fallback matches default
    else:
        trunk = make_trunk_features_2d(nx, ny, m["pe_x"], m["pe_y"], m["pe_cross"])

    branch_layers = [X_test_n.shape[1]] + m["branch_hidden"]
    trunk_layers = [trunk.shape[1]] + m["trunk_hidden"]
    merger_layers = [branch_layers[-1]] + m["merger_hidden"] + [1]

    net = MergeDeepONet(
        branch_layers=branch_layers, 
        trunk_layers=trunk_layers, 
        merger_layers=merger_layers, 
        activation=m["activation"], 
        trunk_chunk=m.get("trunk_chunk", 4096),
        batch_chunk=m.get("batch_chunk", 0),
        amp=m.get("amp", False)
    ).to(device)
    
    dummy_data = dde.data.TripleCartesianProd((X_test_n[:1], trunk), np.zeros((1, trunk.shape[0], 1)), (X_test_n[:1], trunk), np.zeros((1, trunk.shape[0], 1)))
    model = dde.Model(dummy_data, net)
    model.compile("adam", lr=m["lr"], loss=lambda y_t, y_p: torch.mean((y_t - y_p)**2))
    model.restore(os.path.join(cmd_args.dir, "models", f"{cmd_args.eval_model}_model.pt"), verbose=1)

    print("Evaluating Test FE...")
    with torch.no_grad():
        sub_batch = 32 
        preds = []
        for i in range(0, X_test_n.shape[0], sub_batch):
            preds.append(model.predict((X_test_n[i:i+sub_batch], trunk)))
        ypred = np.concatenate(preds, axis=0)

        # RESHAPE T_TEST TO MATCH TRAINING SCRIPT TARGET FORMAT
        Npts = ny * nx
        y_test_full2d = T_test.reshape((-1, Npts, 1)).astype(np.float32)

        fe = global_flux_fe(y_test_full2d, ypred, X_test_raw, m["dim"], ny, nx, y_mean, y_std)

    print(f"\nFinal Test Global Flux FE: {fe:.6f}")

if __name__ == "__main__":
    main()

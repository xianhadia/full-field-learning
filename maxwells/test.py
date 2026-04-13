import argparse
import os
import json

import numpy as np
import torch
import torch.nn as nn
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ny, nx = 38, 840
monitor_x = 120 - 1
_monitor_idx = (np.arange(ny, dtype=np.int64) * nx + monitor_x).astype(np.int64)

def _to_torch(y):
    if isinstance(y, np.ndarray):
        return torch.from_numpy(y).to(device).float(), "np"
    if torch.is_tensor(y):
        return y.to(device).float(), "torch"
    raise TypeError(f"Unsupported type: {type(y)}")

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


def main():
    parser = argparse.ArgumentParser(description="Test script for trained MergeDeepONet models.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing args.json, models, and stats.")
    parser.add_argument("--eval_model", type=str, choices=["best", "final"], default="best", help="Which saved model to evaluate.")
    
    parser.add_argument("--test_file", type=str, default=None, help="Override test file path in args.json")
    parser.add_argument("--test_size", type=int, default=None, help="Override test set size in args.json")
    cmd_args = parser.parse_args()

    args_path = os.path.join(cmd_args.dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Cannot find args.json in {cmd_args.dir}")
    
    with open(args_path, "r") as f:
        args_dict = json.load(f)
    
    args = argparse.Namespace(**args_dict)
    
 
    if not hasattr(args, "dim"):
        if hasattr(args, "pe_y"):
            args.dim = "2d"
        else:
            args.dim = "1d"
            
    if cmd_args.test_file is not None:
        args.test_file = cmd_args.test_file
    if cmd_args.test_size is not None:
        args.test = cmd_args.test_size
        
    print(f"Loading configuration from: {cmd_args.dir}")
    print(f"Inferred/Provided dimension mode: {args.dim}")
    print(f"Testing on {args.test} samples from {args.test_file}...")

    # Load test data
    if args.dim == "1d":
        assume_1d = getattr(args, "assume_already_1d", False)
        X_test_raw, y_test_full_phys = load_dataset_1d(args.test_file, args.test, assume_1d)
    else:
        X_test_raw, y_test_full_phys = load_dataset_2d(args.test_file, args.test)

    # Load standardization data from training
    stats_path = os.path.join(cmd_args.dir, "standardization_stats.npz")
    stats = np.load(stats_path)
    X_mean = stats["X_mean"]
    X_std = stats["X_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    # standardize data
    X_test = (X_test_raw - X_mean) / X_std
    m = X_test.shape[1]

    # Generate Trunk Features
    if args.dim == "1d":
        trunk_full = make_trunk_features_1d(nx, args.pe_x, args.pe_cross)
        trunk_eval = make_monitor_trunk_1d(nx, monitor_x, args.pe_x, args.pe_cross)
    else:
        trunk_full = make_trunk_features_2d(nx, ny, args.pe_x, args.pe_y, args.pe_cross)
        trunk_eval = make_monitor_trunk_2d(nx, ny, monitor_x, args.pe_x, args.pe_y, args.pe_cross)
    
    dim_trunk = trunk_full.shape[1]
    
    # create network
    branch_layers = [m] + args.branch_hidden
    trunk_layers = [dim_trunk] + args.trunk_hidden
    merger_layers = [branch_layers[-1]] + args.merger_hidden + [2]

    eval_net = MergeDeepONet(
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        merger_layers=merger_layers,
        activation=args.activation,
        trunk_chunk=args.trunk_chunk,
        batch_chunk=args.batch_chunk,
        amp=getattr(args, "amp", False), # default False if not found
    ).to(device)

    # uses dummy data to satisfy wrapper
    class DummyData(dde.data.Data):
        def __init__(self):
            super().__init__()
            # Use real shapes so the model initializes correctly
            self.train_x = (X_test[:1], trunk_eval)
            self.train_y = np.zeros((1, trunk_eval.shape[0], 2), dtype=np.float32)
            self.test_x = self.train_x
            self.test_y = self.train_y

        def losses(self, targets, outputs, loss_fn, inputs, model, weight=None):
            return [torch.tensor(0.0, device=device)]

        def train_next_batch(self, batch_size=None):
            return self.train_x, self.train_y

        def test(self):
            return self.test_x, self.test_y
    # ------------------------------------

    dummy_data = DummyData()
    eval_model = dde.Model(dummy_data, eval_net)
    eval_model.compile("adam", lr=args.lr, loss=complex_abs_loss)
    


    # loads weights
    model_name = f"{cmd_args.eval_model}_model.pt"
    model_path = os.path.join(cmd_args.dir, "models", model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Restoring weights from: {model_path}")
    eval_model.restore(model_path, verbose=1)
    print("Running inference...")
    eval_batch_size = getattr(args, "eval_batch", 16)
    
    with torch.no_grad():
        B = X_test.shape[0]
        preds_std = []
        for s in range(0, B, eval_batch_size):
            e = min(B, s + eval_batch_size)
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

    print("="*50)
    print(f"Successfully evaluated {cmd_args.eval_model} model from {cmd_args.dir}")
    print(f"Test Fractional Error @ monitor: {test_fe:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
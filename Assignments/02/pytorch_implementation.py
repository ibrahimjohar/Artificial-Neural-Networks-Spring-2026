#ibrahim johar farooqi
#23K-0074
#BAI-6A
#ANN Assignment 2

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from datasets import generate_toy1, generate_toy2, f_toy1, f_toy2

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")

#activation functions
def sigma_mc(x: torch.Tensor) -> torch.Tensor:
    #softplus - convex AND monotone non-decreasing. used for x- and y-branches.
    return F.softplus(x)                      #log(1 + exp(x))

def sigma_m(x: torch.Tensor) -> torch.Tensor:
    #sigmoid - monotone non-decreasing (not necessarily convex). used for t-branch.
    return torch.sigmoid(x)

def sigma_a(x: torch.Tensor) -> torch.Tensor:
    #sigmoid - used for unconstrained z-branch (any non-linear activation works).
    return torch.sigmoid(x)

#non-negative linear layer
class PosLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.raw_weight = nn.Parameter(
            torch.abs(torch.randn(out_features, in_features)) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        return torch.abs(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.raw_weight.shape[1]}, "
                f"out={self.raw_weight.shape[0]}, "
                f"bias={self.bias is not None}, non_neg=True")


#FFIN - unconstrained baseline
class FFNN(nn.Module):
    #Standard feed-forward neural network.
    #3 hidden layers × 30 neurons → 2041 trainable parameters.
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 30), nn.Sigmoid(),
            nn.Linear(30, 30), nn.Sigmoid(),
            nn.Linear(30, 30), nn.Sigmoid(),
            nn.Linear(30, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


#ISNN-1 - type 1 input specific neural network
class ISNN1(nn.Module):
    def __init__(self, n_hidden: int = 10, n_layers: int = 2):
        super().__init__()
        H = n_hidden

        #y-branch : σ_mc, non-negative weights
        self.y_layers = nn.ModuleList()
        d = 1
        for _ in range(n_layers):
            self.y_layers.append(PosLinear(d, H))
            d = H

        #z-branch : σ_a, unconstrained weights
        self.z_layers = nn.ModuleList()
        d = 1
        for _ in range(n_layers):
            self.z_layers.append(nn.Linear(d, H))
            d = H

        #t-branch : σ_m, non-negative weights
        self.t_layers = nn.ModuleList()
        d = 1
        for _ in range(n_layers):
            self.t_layers.append(PosLinear(d, H))
            d = H

        #x-branch coupling layer
        #W^[xx]_0 : unconstrained (x0 → x1)
        #W^[xy] : non-negative (y_Hy → x1)
        #W^[xz] : unconstrained (z_Hz → x1)
        #W^[xt] : non-negative (t_Ht → x1)
        #bias is in W_xx0; all cross-branch matrices are bias-free
        self.W_xx0 = nn.Linear(1, H, bias=True)
        self.W_xy = PosLinear(H, H, bias=False)
        self.W_xz = nn.Linear(H, H, bias=False)
        self.W_xt = PosLinear(H, H, bias=False)

        #x-branch subsequent layers (h = 1 … Hx-1, eq. 5)
        #all non-negative; last layer maps H → 1 (scalar output)
        self.x_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            out_dim = 1 if (i == n_layers - 2) else H
            self.x_layers.append(PosLinear(H, out_dim))

    #forward pass
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        #cols: [x, y, t, z]
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        #y-branch (σ_mc, eq. 1)
        y = y0
        for layer in self.y_layers:
            y = sigma_mc(layer(y))  #y_Hy after loop

        #z-branch (σ_a, eq. 2)
        z = z0
        for layer in self.z_layers:
            z = sigma_a(layer(z))   #z_Hz

        #t-branch (σ_m, eq. 3)
        t = t0
        for layer in self.t_layers:
            t = sigma_m(layer(t))   #t_Ht

        #x-branch: coupling layer (eq. 4)
        x = sigma_mc(
            self.W_xx0(x0)   #unconstrained (includes bias b^[x]_0)
            + self.W_xy(y)   #non-negative (y_Hy)
            + self.W_xz(z)   #unconstrained (z_Hz)
            + self.W_xt(t)   #non-negative (t_Ht)
        )

        #x-branch: subsequent layers (eq. 5)
        #applying σ_mc to all but the last layer
        n = len(self.x_layers)
        for i, layer in enumerate(self.x_layers):
            if i < n - 1:
                x = sigma_mc(layer(x))
            else:
                x = layer(x)   #linear output, no activation

        return x.squeeze(-1)

#ISNN-2 - type 2 input specific neural network (eqs. 6-10)
class ISNN2(nn.Module):
    def __init__(self, n_hidden: int = 15, H: int = 2):
        super().__init__()
        Nh  = n_hidden
        self.H  = H
        self.Nh = Nh

        #y-branch : H-1 layers, σ_mc, non-negative
        self.y_layers = nn.ModuleList()
        d = 1
        for _ in range(H - 1):
            self.y_layers.append(PosLinear(d, Nh))
            d = Nh

        #z-branch : H-1 layers, σ_a, unconstrained
        self.z_layers = nn.ModuleList()
        d = 1
        for _ in range(H - 1):
            self.z_layers.append(nn.Linear(d, Nh))
            d = Nh

        #t-branch : H-1 layers, σ_m, non-negative
        self.t_layers = nn.ModuleList()
        d = 1
        for _ in range(H - 1):
            self.t_layers.append(PosLinear(d, Nh))
            d = Nh

        #x-branch first layer (h=0, eq. 9)
        #mixes original inputs - W^[xx]_0 is NOT constrained non-neg
        self.W_xx_first = nn.Linear(1, Nh, bias=True)   #x0→x1, unconstrained+bias
        self.W_xy_first = PosLinear(1, Nh, bias=False)  #y0→x1, non-neg
        self.W_xz_first = nn.Linear(1, Nh, bias=False)  #z0→x1, unconstrained
        self.W_xt_first = PosLinear(1, Nh, bias=False)  #t0→x1, non-neg

        #x-branch subsequent layers (h=1,...,H-1, eq. 10)
        #W^[xx]_h : non-negative (x_h → next)
        #W^[xx0]_h : unconstrained (x0 skip → next, includes bias b^[x]_h)
        #W^[xy]_h : non-negative (y_h → next)
        #W^[xz]_h : unconstrained (z_h → next)
        #W^[xt]_h : non-negative (t_h → next)
        self.W_xx = nn.ModuleList()
        self.W_xx0s = nn.ModuleList()   #skip connections from x0
        self.W_xy = nn.ModuleList()
        self.W_xz = nn.ModuleList()
        self.W_xt = nn.ModuleList()

        for h in range(1, H):                   #h = 1 … H-1
            out = 1 if h == H - 1 else Nh       #last x-layer outputs scalar
            self.W_xx.append(PosLinear(Nh, out, bias=False))
            self.W_xx0s.append(nn.Linear(1, out, bias=True))  #skip + bias
            self.W_xy.append(PosLinear(Nh, out, bias=False))
            self.W_xz.append(nn.Linear(Nh, out, bias=False))
            self.W_xt.append(PosLinear(Nh, out, bias=False))

    #forward pass
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        #cols: [x, y, t, z]
        x0 = X[:, 0:1]
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        #run y/z/t branches; store ALL intermediate activations
        #y_acts[k] = activation after k layers  (y_acts[0] = y0)
        y_acts = [y0]
        for layer in self.y_layers:
            y_acts.append(sigma_mc(layer(y_acts[-1])))

        z_acts = [z0]
        for layer in self.z_layers:
            z_acts.append(sigma_a(layer(z_acts[-1])))

        t_acts = [t0]
        for layer in self.t_layers:
            t_acts.append(sigma_m(layer(t_acts[-1])))

        #x-branch first layer (eq. 9): uses x0, y0, z0, t0
        x = sigma_mc(
            self.W_xx_first(x0)
            + self.W_xy_first(y0)
            + self.W_xz_first(z0)
            + self.W_xt_first(t0)
        )
        x_acts = [x0, x]   #x_acts[0]=x0, x_acts[1]=x1

        #x-branch subsequent layers (eq. 10): uses x_h, x0, y_h, z_h, t_h
        for idx in range(len(self.W_xx)):
            h      = idx + 1                    #1-based layer index
            x_prev = x_acts[-1]
            y_h    = y_acts[min(h, len(y_acts)-1)]
            z_h    = z_acts[min(h, len(z_acts)-1)]
            t_h    = t_acts[min(h, len(t_acts)-1)]

            lin = (
                self.W_xx[idx](x_prev)     #non-neg
                + self.W_xx0s[idx](x0)     #skip from x0 (unconstrained + bias)
                + self.W_xy[idx](y_h)      #non-neg
                + self.W_xz[idx](z_h)      #unconstrained
                + self.W_xt[idx](t_h)      #non-neg
            )

            if h < self.H - 1:
                x = sigma_mc(lin)
            else:
                x = lin                    #no activation on output
            x_acts.append(x)

        return x.squeeze(-1)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float64, device=DEVICE)

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30_000,
    lr: float = 1e-3,
    eval_every: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    Xtr, ytr = to_tensor(X_train), to_tensor(y_train)
    Xte, yte = to_tensor(X_test),  to_tensor(y_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    test_checkpoints: list[tuple[int, float]] = []   #(epoch_idx, loss)

    for epoch in range(epochs):
        #training forward + backward
        optimizer.zero_grad()
        pred_tr = model(Xtr)
        loss_tr = F.mse_loss(pred_tr, ytr)
        loss_tr.backward()
        optimizer.step()
        train_losses.append(loss_tr.item())

        #test evaluation (every eval_every steps)
        if epoch % eval_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                loss_te = F.mse_loss(model(Xte), yte).item()
            test_checkpoints.append((epoch, loss_te))

    #interpolate test losses to full epoch resolution
    ck_ep  = np.array([c[0] for c in test_checkpoints], dtype=float)
    ck_val = np.array([c[1] for c in test_checkpoints], dtype=float)
    test_losses = np.interp(np.arange(epochs, dtype=float), ck_ep, ck_val)

    return np.array(train_losses), test_losses


def run_seeds(
    ModelClass,
    model_kwargs: dict,
    X_train, y_train,
    X_test, y_test,
    epochs: int,
    n_seeds: int = 10,
    name: str = "",
) -> dict:
    all_tr, all_te, models = [], [], []

    print(f"\n  ── {name:6s}  ({n_seeds} seeds × {epochs} epochs) ──")

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ModelClass(**model_kwargs).to(DEVICE)
        t0 = time.time()
        tr, te = train_model(model, X_train, y_train, X_test, y_test, epochs=epochs)
        elapsed = time.time() - t0

        all_tr.append(tr)
        all_te.append(te)
        models.append(model)

        print(f"     seed {seed+1:02d}/{n_seeds}  |  "
              f"train MSE={tr[-1]:.3e}  test MSE={te[-1]:.3e}  "
              f"[{elapsed:.1f}s]")

    return {
        "train_losses": np.array(all_tr),   #(n_seeds, epochs)
        "test_losses" : np.array(all_te),
        "models"      : models,
    }

#plotting
MODEL_COLORS = {"FFNN": "tab:red", "ISNN-1": "tab:green", "ISNN-2": "tab:blue"}

def _mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr.mean(axis=0), arr.std(axis=0)

def plot_losses(
    results: dict[str, dict],
    title: str,
    save_path: str,
) -> None:
    epochs = results[next(iter(results))]["train_losses"].shape[1]
    ep_axis = np.arange(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, key, sub in [
        (axes[0], "train_losses", "(a) Training Loss"),
        (axes[1], "test_losses",  "(b) Test Loss"),
    ]:
        for name, res in results.items():
            mu, sd = _mean_std(res[key])
            c = MODEL_COLORS.get(name, "black")
            ax.semilogy(ep_axis, mu, color=c, lw=1.5, label=name)
            ax.fill_between(ep_axis,
                            np.maximum(mu - sd, 1e-9),
                            mu + sd,
                            color=c, alpha=0.15)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss", fontsize=11)
        ax.set_title(sub, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_behavior(
    results: dict[str, dict],
    true_func,
    train_end: float,
    test_end: float,
    title: str,
    save_path: str,
) -> None:
    n_pts  = 300
    vals   = np.linspace(0.0, test_end, n_pts)
    diag   = np.column_stack([vals] * 4)           #(n_pts, 4)
    true_y = true_func(diag)

    split  = np.searchsorted(vals, train_end)       #index where train ends

    model_names = list(results.keys())
    n_models    = len(model_names)
    fig, axes   = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5), sharey=False)
    if n_models == 1:
        axes = [axes]

    Xd = to_tensor(diag)

    for ax, name in zip(axes, model_names):
        res = results[name]

        #collect predictions from every seed
        preds = []
        for m in res["models"]:
            m.eval()
            with torch.no_grad():
                p = m(Xd).cpu().numpy()
            preds.append(p)
        preds = np.array(preds)              # (n_seeds, n_pts)
        mu, sd = preds.mean(0), preds.std(0)

        #true response
        ax.plot(vals, true_y, "k--", lw=1.8, label="True response", zorder=5)

        #interpolated region  (in training domain)
        ax.plot(vals[:split], mu[:split], color="steelblue", lw=1.5,
                label="Interpolated response")
        ax.fill_between(vals[:split],
                        mu[:split] - sd[:split],
                        mu[:split] + sd[:split],
                        color="steelblue", alpha=0.25)

        #extrapolated region  (beyond training domain)
        ax.plot(vals[split:], mu[split:], color="tomato", lw=1.5,
                label="Extrapolated response")
        ax.fill_between(vals[split:],
                        mu[split:] - sd[split:],
                        mu[split:] + sd[split:],
                        color="tomato", alpha=0.25)

        ax.axvline(train_end, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("x = y = t = z", fontsize=11)
        ax.set_ylabel("f", fontsize=11)
        ax.set_title(f"({chr(96 + model_names.index(name) + 1)}) {name}",
                     fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(epochs: int = 30_000, n_seeds: int = 10) -> None:

    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Model parameter counts")
    print("=" * 65)
    for Cls, kw, nm in [
        (FFNN,  {}, "FFNN  "),
        (ISNN1, {"n_hidden": 10, "n_layers": 2}, "ISNN-1"),
        (ISNN2, {"n_hidden": 15, "H": 2}, "ISNN-2"),
    ]:
        n = count_params(Cls(**kw))
        print(f"  {nm} : {n:>5} trainable parameters")
    print("=" * 65)

    MODEL_DEFS = [
        ("FFNN", FFNN, {}),
        ("ISNN-1", ISNN1, {"n_hidden": 10, "n_layers": 2}),
        ("ISNN-2", ISNN2, {"n_hidden": 15, "H": 2}),
    ]

    #toy dataset 1 - additive function
    print("\n" + "=" * 65)
    print("  Toy Dataset 1  -  Additive function  (eq. 12)")
    print("=" * 65)

    X_tr1, y_tr1, X_te1, y_te1 = generate_toy1()

    results1 = {}
    for name, Cls, kw in MODEL_DEFS:
        results1[name] = run_seeds(
            Cls, kw,
            X_tr1, y_tr1, X_te1, y_te1,
            epochs=epochs, n_seeds=n_seeds, name=name,
        )

    print("\n  Generating loss plot …")
    plot_losses(
        results1,
        title="Toy Dataset 1 - Additive function",
        save_path=f"{out_dir}/fig3_toy1_losses.png",
    )
    print("  Generating behavior plot …")
    plot_behavior(
        results1, true_func=f_toy1,
        train_end=4.0, test_end=6.0,
        title="Toy Dataset 1 — Behavioral response along diagonal",
        save_path=f"{out_dir}/fig4_toy1_behavior.png",
    )

    #toy dataset 2 - multiplicative function
    print("\n" + "=" * 65)
    print("  Toy Dataset 2  —  Multiplicative function  (eq. 13)")
    print("=" * 65)

    X_tr2, y_tr2, X_te2, y_te2 = generate_toy2()

    results2 = {}
    for name, Cls, kw in MODEL_DEFS:
        results2[name] = run_seeds(
            Cls, kw,
            X_tr2, y_tr2, X_te2, y_te2,
            epochs=epochs, n_seeds=n_seeds, name=name,
        )

    print("\n  Generating loss plot …")
    plot_losses(
        results2,
        title="Toy Dataset 2 - Multiplicative function",
        save_path=f"{out_dir}/fig5_toy2_losses.png",
    )
    print("  Generating behavior plot …")
    plot_behavior(
        results2, true_func=f_toy2,
        train_end=4.0, test_end=10.0,
        title="Toy Dataset 2 - Behavioral response along diagonal",
        save_path=f"{out_dir}/fig6_toy2_behavior.png",
    )

    print("\n" + "=" * 65)
    print("Figures written to:", out_dir)
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch FFNN / ISNN-1 / ISNN-2 implementation"
    )
    parser.add_argument(
        "--epochs", type=int, default=30_000,
        help=(
            "Training epochs per seed (paper uses 30000). "
            "CPU estimate: ~90 min for 30000 epochs across all runs. "
            "Use --epochs 2000 for a quick smoke-test."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of random initialisations (default 10)",
    )
    args = parser.parse_args()

    main(epochs=args.epochs, n_seeds=args.seeds)

#ibrahim johar farooqi
#23K-0074
#BAI-6A
#ANN Assignment 2

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

#latin hypercube sampling (LHS) - statistical method for generating a sample of possible collections of parameter vals from a 
#multidimensional distribution.
def lhs_sample(
    n_samples: int,
    n_dims: int,
    low: float,
    high: float,
    seed: int = 42,
) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    unit_samples = sampler.random(n=n_samples)           #vals in [0, 1]
    return qmc.scale(unit_samples, l_bounds=low, u_bounds=high)
    
    #generate latin hypercube samples uniformly scaled to [low, high]^n_dims.
    #n_samples: num of sample points
    #n_dims: dimensionality of each sample
    #low, high: bounds applied identically to every dimension
    #seed: random seed for reproducibility
    #returns -> ndarray of shape (n_samples, n_dims)

#toy dataset 1 - additive function
def f_toy1(data: np.ndarray) -> np.ndarray:
    x, y, t, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    #np.log1p(u) = log(1+u), numerically more stable than np.log(1+np.exp(...))
    return (
        np.exp(-0.5 * x)
        + np.log1p(np.exp(0.4 * y))   #softplus component → convex + monotone
        + np.tanh(t)                 #monotone
        + np.sin(z)                  #arbitrary
        - 0.4
    )

def generate_toy1(
    n_train: int = 500,
    n_test: int = 5000,
    train_range: tuple = (0.0, 4.0),
    test_range: tuple = (0.0, 6.0),
    seed: int = 42,
) -> tuple:
    X_train = lhs_sample(n_train, 4, train_range[0], train_range[1], seed=seed)
    X_test = lhs_sample(n_test, 4, test_range[0], test_range[1], seed=seed + 1000)
    return X_train, f_toy1(X_train), X_test, f_toy1(X_test)

#toy dataset 2 - multiplicative function
def f_toy2(data: np.ndarray) -> np.ndarray:
    x, y, t, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    fx = np.exp(-0.3 * x)                    #convex in x
    fy = (0.15 * y) ** 2                     #convex + monotone in y
    ft = np.tanh(0.3 * t)                    #monotone in t
    fz = 0.2 * np.sin(0.5 * z + 2) + 0.5     #arbitrary in z

    return fx * fy * ft * fz

def generate_toy2(
    n_train: int = 500,
    n_test: int = 5000,
    train_range: tuple = (0.0, 4.0),
    test_range: tuple = (0.0, 10.0),
    seed: int = 42,
) -> tuple:
    X_train = lhs_sample(n_train, 4, train_range[0], train_range[1], seed=seed)
    X_test  = lhs_sample(n_test, 4, test_range[0], test_range[1], seed=seed + 1000)
    return X_train, f_toy2(X_train), X_test, f_toy2(X_test)

def save_datasets(out_dir: str = "data") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("  generating ISNN toy datasets")
    print("=" * 60)

    #toy1 - additive function
    print("\n[1/2]  Additive function  f(x,y,t,z) = exp(-0.5x) + log(1+exp(0.4y)) + tanh(t) + sin(z) - 0.4")
    X_tr1, y_tr1, X_te1, y_te1 = generate_toy1()

    np.save(f"{out_dir}/toy1_X_train.npy", X_tr1)
    np.save(f"{out_dir}/toy1_y_train.npy", y_tr1)
    np.save(f"{out_dir}/toy1_X_test.npy", X_te1)
    np.save(f"{out_dir}/toy1_y_test.npy", y_te1)

    print(f"Train: {X_tr1.shape} range [0,4]^4 y ∈ [{y_tr1.min():.3f}, {y_tr1.max():.3f}]")
    print(f"Test: {X_te1.shape} range [0,6]^4 y ∈ [{y_te1.min():.3f}, {y_te1.max():.3f}]")

    #toy 2 - multiplicative function
    print("\n[2/2]  Multiplicative function  g(x,y,t,z) = exp(-0.3x)·(0.15y)²·tanh(0.3t)·(0.2sin(0.5z+2)+0.5)")
    X_tr2, y_tr2, X_te2, y_te2 = generate_toy2()

    np.save(f"{out_dir}/toy2_X_train.npy", X_tr2)
    np.save(f"{out_dir}/toy2_y_train.npy", y_tr2)
    np.save(f"{out_dir}/toy2_X_test.npy", X_te2)
    np.save(f"{out_dir}/toy2_y_test.npy", y_te2)

    print(f"Train: {X_tr2.shape} range [0,4]^4 y ∈ [{y_tr2.min():.3f}, {y_tr2.max():.3f}]")
    print(f"Test: {X_te2.shape} range [0,10]^4 y ∈ [{y_te2.min():.3f}, {y_te2.max():.3f}]")

    print(f"\nall datasets saved to '{out_dir}/'")

    return dict(
        toy1_X_train=X_tr1, toy1_y_train=y_tr1,
        toy1_X_test=X_te1, toy1_y_test=y_te1,
        toy2_X_train=X_tr2, toy2_y_train=y_tr2,
        toy2_X_test=X_te2, toy2_y_test=y_te2,
    )

#visualization 
def _diagonal_response(func, t_end: float, n_pts: int = 300) -> tuple:
    #evaluate func along the diagonal x=y=t=z from 0 to t_end
    vals = np.linspace(0, t_end, n_pts)
    pts  = np.column_stack([vals, vals, vals, vals])
    return vals, func(pts)


def plot_true_functions(out_dir: str = "data") -> None:
    #quick diagnostic: plot each true function along the diagonal
    #x = y = t = z over both the training and test domains.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, func, title, t_train, t_test in [
        (axes[0], f_toy1, "Toy 1 – Additive", 4.0, 6.0),
        (axes[1], f_toy2, "Toy 2 – Multiplicative", 4.0, 10.0),
    ]:
        xs_all, ys_all = _diagonal_response(func, t_test)
        xs_tr, ys_tr = _diagonal_response(func, t_train)

        ax.plot(xs_all, ys_all, "k--", lw=1.8, label="True response")
        ax.fill_betweenx(
            [ys_all.min() - 0.1, ys_all.max() + 0.1],
            0, t_train, alpha=0.08, color="royalblue", label=f"Train range [0,{t_train}]"
        )
        ax.axvline(t_train, color="royalblue", lw=1.2, ls="--")
        ax.set_xlim(0, t_test)
        ax.set_xlabel("Input value  (x = y = t = z)", fontsize=11)
        ax.set_ylabel("Function output", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{out_dir}/true_functions_diagonal.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot saved → {save_path}")


if __name__ == "__main__":
    datasets = save_datasets(out_dir="data")
    plot_true_functions(out_dir="data")
    #test the functions at the origin 
    print("\nSanity check — f evaluated at origin [0,0,0,0]:")
    origin = np.zeros((1, 4))
    print(f"Toy1 f(0,0,0,0) = {f_toy1(origin)[0]:.4f} (expected ≈ 0.6931 + 1 + 0 + 0 - 0.4 ≈ 1.293)")
    print(f"Toy2 g(0,0,0,0) = {f_toy2(origin)[0]:.4f} (expected = 0 because fy=(0.15*0)²=0)")
    
    
"""
Generates the two toy datasets from Section 3.1 of the ISNN paper.

Toy Dataset 1 -> Additive function (eq.12)

f(x, y, t, z) = exp(-0.5x) + log(1+exp(0.4y)) + tanh(t) + sin(z) - 0.4
structural properties:
- convex in x
- convex + monotonically non-decreasing in y
- monotonically non-decreasing in t
- arbitrary (unconstrained) in z

Toy Dataset 2 -> Multiplicative function (eqs.13-14)
    g(x, y, t, z) = fx * fy * ft * fz   
    where
        fx = exp(-0.3x)
        fy = (0.15y)²
        ft = tanh(0.3t)
        fz = 0.2·sin(0.5z + 2) + 0.5
same structural properties as dataset 1.

Sampling strategy (matching the paper):
- Training : 500 samples via Latin Hypercube Sampling (LHS) in [0, 4]⁴
- Testing  : 5000 samples via LHS
                   Dataset 1 → [0, 6]⁴
                   Dataset 2 → [0, 10]⁴
"""

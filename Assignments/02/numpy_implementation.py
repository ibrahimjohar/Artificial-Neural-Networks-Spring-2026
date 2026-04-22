#ibrahim johar farooqi
#23K-0074
#BAI-6A
#ANN Assignment 2

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from datasets import generate_toy1, generate_toy2, f_toy1, f_toy2

#activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    #sigmoid: 1 / (1 + exp(-x))
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def sigmoid_d(x: np.ndarray) -> np.ndarray:
    #derivative of sigmoid: σ(x)(1 - σ(x))
    s = sigmoid(x)
    return s * (1.0 - s)


def softplus(x: np.ndarray) -> np.ndarray:
    #softplus: log(1 + exp(x))
    #σ_mc from the paper - convex AND monotone non-decreasing.
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -500.0, 30.0))))


def softplus_d(x: np.ndarray) -> np.ndarray:
    #derivative of softplus = sigmoid(x)
    return sigmoid(x)

#adam optimizer
class Adam:
    def __init__(self, params: dict, lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> None:
        #in-place Adam update for every key in params
        self.t += 1
        t, b1, b2, eps, lr = self.t, self.b1, self.b2, self.eps, self.lr
        for k in params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g ** 2
            m_hat = self.m[k] / (1.0 - b1 ** t)
            v_hat = self.v[k] / (1.0 - b2 ** t)
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

#FFNN (unconstrained baseline, matches pytorch FFNN)
class FFNN:
    #standard feed-forward network
    #architecture: 4 → 30 (σ) → 30 (σ) → 30 (σ) → 1 (linear)
    #total parameters: 4·30+30 + 30·30+30 + 30·30+30 + 30·1+1 = 2041
    #activation: sigmoid (σ) on hidden, linear on output.

    def __init__(self, lr: float = 1e-3, seed: int = 0):
        np.random.seed(seed)
        s = 0.1

        #weight shapes: (out, in) - same convention as PyTorch nn.Linear
        self.params = {
            'W1': np.random.randn(30, 4) * s, 'b1': np.zeros(30),
            'W2': np.random.randn(30, 30) * s, 'b2': np.zeros(30),
            'W3': np.random.randn(30, 30) * s, 'b3': np.zeros(30),
            'W4': np.random.randn(1, 30) * s, 'b4': np.zeros( 1),
        }
        self.opt = Adam(self.params, lr=lr)

    def count_params(self) -> int:
        return sum(v.size for v in self.params.values())

    
    #forward pass
    def forward(self, X: np.ndarray) -> tuple:
        p = self.params

        #layer 1: linear → sigmoid
        z1 = X  @ p['W1'].T + p['b1']   #(N, 30)
        a1 = sigmoid(z1)
        #layer 2: linear → sigmoid
        z2 = a1 @ p['W2'].T + p['b2']   #(N, 30)
        a2 = sigmoid(z2)
        #layer 3: linear → sigmoid
        z3 = a2 @ p['W3'].T + p['b3']   #(N, 30)
        a3 = sigmoid(z3)
        #output: Linear (no activation)
        z4 = a3 @ p['W4'].T + p['b4']   #(N, 1)

        cache = {'X': X, 'z1': z1, 'a1': a1,
                          'z2': z2, 'a2': a2,
                          'z3': z3, 'a3': a3, 'z4': z4}
        return z4.ravel(), cache   #(N,)

    #backward pass
    def backward(self, grad_out: np.ndarray, cache: dict) -> dict:
        #grad_out : (N,) - dL/d(out) = (2/N)(pred - target)
        #returns grads dict with same keys as self.params.

        #from output layer backwards:
        #  δ_k = ∇_{y_k}L ⊙ f'(z_k)
        #  ∇_W L = δ_k.T @ y_{k-1}
        #  ∇_b L = δ_k.sum(0)
        #  ∇_{y_{k-1}} L = δ_k @ W_k
        
        p = self.params
        g = {}              #gradient dict (same keys as params)

        d = grad_out.reshape(-1, 1)   #(N, 1)

        #layer 4 (linear, no activation → f'=1, so δ = d)
        #δ4 = d  (upstream gradient, no activation to multiply)
        g['W4'] = d.T @ cache['a3']    #(1, 30)
        g['b4'] = d.sum(axis=0)        #(1,)
        d = d @ p['W4']                #(N, 30) - propagate back

        #layer 3 (sigmoid activation)
        #δ3 = ∇_{a3}L ⊙ σ'(z3)
        d = d * sigmoid_d(cache['z3'])    #(N, 30)
        g['W3'] = d.T @ cache['a2']       #(30, 30)
        g['b3'] = d.sum(axis=0)           #(30,)
        d = d @ p['W3']                   #(N, 30)

        #layer 2 (sigmoid)
        d = d * sigmoid_d(cache['z2'])
        g['W2'] = d.T @ cache['a1']
        g['b2'] = d.sum(axis=0)
        d = d @ p['W2']

        #layer 1 (sigmoid)
        d = d * sigmoid_d(cache['z1'])
        g['W1'] = d.T @ cache['X']
        g['b1'] = d.sum(axis=0)

        return g

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out   #(N,)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        N = y.shape[0]
        pred, cache = self.forward(X)
        loss = float(np.mean((pred - y) ** 2))
        grad_out = (2.0 / N) * (pred - y)        #(N,)
        grads = self.backward(grad_out, cache)
        self.opt.step(self.params, grads)
        return loss

#ISSN-1
class ISNN1:
    def __init__(self, n_hidden: int = 10, n_layers: int = 2,
                 lr: float = 1e-3, seed: int = 0):
        np.random.seed(seed)
        H, s = n_hidden, 0.1

        self.params = {
            #y branch (PosLinear: use abs in forward)
            'W_yy0': np.abs(np.random.randn(H, 1)) * s, 'b_y0': np.zeros(H),   #layer 0: 1→H
            'W_yy1': np.abs(np.random.randn(H, H)) * s, 'b_y1': np.zeros(H),   #layer 1: H→H

            #z branch (Linear: unconstrained)
            'W_zz0': np.random.randn(H, 1) * s, 'b_z0': np.zeros(H),
            'W_zz1': np.random.randn(H, H) * s, 'b_z1': np.zeros(H),

            #t branch (PosLinear)
            'W_tt0': np.abs(np.random.randn(H, 1)) * s, 'b_t0': np.zeros(H),
            'W_tt1': np.abs(np.random.randn(H, H)) * s, 'b_t1': np.zeros(H),

            #x coupling (eq. 4)
            #W_xx0: x0 → x1, unconstrained + bias
            'W_xx0': np.random.randn(H, 1) * s, 'b_xx0': np.zeros(H),
            #W_xy: y_Hy → x1, non-negative, no bias
            'W_xy': np.abs(np.random.randn(H, H)) * s,
            #W_xz: z_Hz → x1, unconstrained, no bias
            'W_xz': np.random.randn(H, H) * s,
            #W_xt: t_Ht → x1, non-negative, no bias
            'W_xt': np.abs(np.random.randn(H, H)) * s,

            #x output (PosLinear(H → 1), linear, no activation)
            'W_xout': np.abs(np.random.randn(1, H)) * s, 'b_xout': np.zeros(1),
        }
        self.opt = Adam(self.params, lr=lr)

    def count_params(self) -> int:
        return sum(v.size for v in self.params.values())

    #forward pass
    def forward(self, X: np.ndarray) -> tuple:
        p = self.params

        x0 = X[:, 0:1]   #(N,1)
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        #y branch: 2 layers (σ_mc = softplus, W ≥ 0)
        #eq. 1:  y_{h+1} = σ_mc( y_h @ |W_yy_h|.T + b_y_h )
        zy0 = y0 @ np.abs(p['W_yy0']).T + p['b_y0']   # (N,H)
        ay0 = softplus(zy0)
        zy1 = ay0 @ np.abs(p['W_yy1']).T + p['b_y1']  # (N,H)
        y_Hy = softplus(zy1)

        #z branch: 2 layers (σ_a = sigmoid, unconstrained)
        #eq. 2:  z_{h+1} = σ_a( z_h @ W_zz_h.T + b_z_h )
        zz0 = z0 @ p['W_zz0'].T + p['b_z0']           #(N,H)
        az0 = sigmoid(zz0)
        zz1 = az0 @ p['W_zz1'].T + p['b_z1']          #(N,H)
        z_Hz = sigmoid(zz1)

        #t branch: 2 layers (σ_m = sigmoid, W ≥ 0)
        #eq. 3:  t_{h+1} = σ_m( t_h @ |W_tt_h|.T + b_t_h )
        zt0 = t0 @ np.abs(p['W_tt0']).T + p['b_t0']   # (N,H)
        at0 = sigmoid(zt0)
        zt1 = at0 @ np.abs(p['W_tt1']).T + p['b_t1']  # (N,H)
        t_Ht = sigmoid(zt1)

        #x coupling layer: eq. 4
        #x1 = σ_mc( x0·W_xx0 + y_Hy·|W_xy| + z_Hz·W_xz + t_Ht·|W_xt| + b )
        pre_x1 = (x0  @ p['W_xx0'].T  + p['b_xx0']
                + y_Hy @ np.abs(p['W_xy']).T
                + z_Hz @ p['W_xz'].T
                + t_Ht @ np.abs(p['W_xt']).T)          #(N,H)
        x1 = softplus(pre_x1)

        #x output layer: eq. 5 (linear, no activation)
        #out = x1 @ |W_xout|.T + b_xout
        out = x1 @ np.abs(p['W_xout']).T + p['b_xout']  #(N,1)

        cache = {
            'X': X, 'x0': x0, 'y0': y0, 't0': t0, 'z0': z0,
            #y branch
            'zy0': zy0, 'ay0': ay0, 'zy1': zy1, 'y_Hy': y_Hy,
            #z branch
            'zz0': zz0, 'az0': az0, 'zz1': zz1, 'z_Hz': z_Hz,
            #t branch
            'zt0': zt0, 'at0': at0, 'zt1': zt1, 't_Ht': t_Ht,
            #x branch
            'pre_x1': pre_x1, 'x1': x1,
        }
        return out.ravel(), cache   #(N,)

    #backward pass
    def backward(self, grad_out: np.ndarray, cache: dict) -> dict:
        #for PosLinear weights: ∇_{W_raw} = ∇_{W_eff} * sign(W_raw)
        
        p = self.params
        g = {k: np.zeros_like(v) for k, v in p.items()}

        #x output layer (linear, δ = upstream, no f' needed)
        d = grad_out.reshape(-1, 1)             #(N, 1)
        #∇_W = δ.T @ x1; correct for abs
        g['W_xout'] = (d.T @ cache['x1']) * np.sign(p['W_xout'])   #(1, H)
        g['b_xout'] = d.sum(axis=0)                                #(1,)
        #propagate: d_{x1} = δ @ |W_xout|
        d = d @ np.abs(p['W_xout'])             #(N, H)

        #x coupling layer (softplus activation)
        #δ = d ⊙ softplus'(pre_x1)
        d = d * softplus_d(cache['pre_x1'])     #(N, H)

        #gradients for each source in the coupling sum
        #x0 → unconstrained
        g['W_xx0'] = d.T @ cache['x0']          #(H, 1)
        g['b_xx0'] = d.sum(axis=0)              #(H,)
        #y_Hy → non-negative (sign correction)
        g['W_xy'] = (d.T @ cache['y_Hy']) * np.sign(p['W_xy'])   #(H, H)
        #z_Hz → unconstrained
        g['W_xz'] = d.T @ cache['z_Hz']                           #(H, H)
        #t_Ht → non-negative
        g['W_xt'] = (d.T @ cache['t_Ht']) * np.sign(p['W_xt'])   #(H, H)

        #propagate gradient to each satellite branch final output
        dy_Hy = d @ np.abs(p['W_xy'])    #(N, H) → into y branch backward
        dz_Hz = d @ p['W_xz']            #(N, H)
        dt_Ht = d @ np.abs(p['W_xt'])    #(N, H)

        #y branch backward (layer 1 → layer 0)
        #layer 1: y_Hy = softplus(zy1),  zy1 = ay0 @ |W_yy1|.T + b_y1
        d = dy_Hy * softplus_d(cache['zy1'])               #(N, H)
        g['W_yy1'] = (d.T @ cache['ay0']) * np.sign(p['W_yy1'])   #(H, H)
        g['b_y1'] = d.sum(axis=0)                                #(H,)
        d = d @ np.abs(p['W_yy1'])                         #(N, H)

        #layer 0: ay0 = softplus(zy0),  zy0 = y0 @ |W_yy0|.T + b_y0
        d = d * softplus_d(cache['zy0'])                   #(N, H)
        g['W_yy0'] = (d.T @ cache['y0']) * np.sign(p['W_yy0'])    #(H, 1)
        g['b_y0'] = d.sum(axis=0)       #(H,)

        #z branch backward (layer 1 → layer 0, unconstrained)
        #layer 1: z_Hz = sigmoid(zz1)
        d = dz_Hz * sigmoid_d(cache['zz1'])  #(N, H)
        g['W_zz1'] = d.T @ cache['az0']      #(H, H)
        g['b_z1'] = d.sum(axis=0)
        d = d @ p['W_zz1']                   #(N, H)

        #layer 0: az0 = sigmoid(zz0)
        d = d * sigmoid_d(cache['zz0'])
        g['W_zz0'] = d.T @ cache['z0']       #(H, 1)
        g['b_z0'] = d.sum(axis=0)

        #t branch backward (layer 1 → layer 0, PosLinear)
        #layer 1: t_Ht = sigmoid(zt1)
        d = dt_Ht * sigmoid_d(cache['zt1'])
        g['W_tt1'] = (d.T @ cache['at0']) * np.sign(p['W_tt1'])   #(H, H)
        g['b_t1'] = d.sum(axis=0)
        d = d @ np.abs(p['W_tt1'])

        #layer 0: at0 = sigmoid(zt0)
        d = d * sigmoid_d(cache['zt0'])
        g['W_tt0'] = (d.T @ cache['t0']) * np.sign(p['W_tt0'])    #(H, 1)
        g['b_t0'] = d.sum(axis=0)

        return g

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        N = y.shape[0]
        pred, cache = self.forward(X)
        loss = float(np.mean((pred - y) ** 2))
        grad_out = (2.0 / N) * (pred - y)
        grads = self.backward(grad_out, cache)
        self.opt.step(self.params, grads)
        return loss


#ISNN-2
class ISNN2:
    def __init__(self, n_hidden: int = 15, H: int = 2,
                 lr: float = 1e-3, seed: int = 0):
        assert H == 2, "this implementation is for H=2 (paper default)"
        np.random.seed(seed)
        Nh, s = n_hidden, 0.1

        self.params = {
            #y branch: 1 layer (σ_mc, W≥0)
            'W_yy0': np.abs(np.random.randn(Nh, 1)) * s, 'b_y0': np.zeros(Nh),

            #z branch: 1 layer (σ_a, unconstrained)
            'W_zz0': np.random.randn(Nh, 1) * s, 'b_z0': np.zeros(Nh),

            #t branch: 1 layer (σ_m, W≥0)
            'W_tt0': np.abs(np.random.randn(Nh, 1)) * s, 'b_t0': np.zeros(Nh),

            #x first layer (eq.9): raw scalar inputs → x1
            #W_xx_first: x0 → x1, unconstrained + bias
            'W_xx_first': np.random.randn(Nh, 1) * s, 'b_xx_first': np.zeros(Nh),
            #W_xy_first: y0 → x1, non-neg, no bias
            'W_xy_first': np.abs(np.random.randn(Nh, 1)) * s,
            #W_xz_first: z0 → x1, unconstrained, no bias
            'W_xz_first': np.random.randn(Nh, 1) * s,
            #W_xt_first: t0 → x1, non-neg, no bias
            'W_xt_first': np.abs(np.random.randn(Nh, 1)) * s,

            #x last layer (Eq.10): processed states → out
            #W_xx_last: x1 → out, non-neg, no bias
            'W_xx_last': np.abs(np.random.randn(1, Nh)) * s,
            #W_xx0_skip: x0 → out, unconstrained + bias
            'W_xx0_skip': np.random.randn(1, 1) * s, 'b_skip': np.zeros(1),
            #W_xy_last: y1 → out, non-neg, no bias
            'W_xy_last': np.abs(np.random.randn(1, Nh)) * s,
            #W_xz_last: z1 → out, unconstrained, no bias
            'W_xz_last': np.random.randn(1, Nh) * s,
            #W_xt_last: t1 → out, non-neg, no bias
            'W_xt_last': np.abs(np.random.randn(1, Nh)) * s,
        }
        self.opt = Adam(self.params, lr=lr)

    def count_params(self) -> int:
        return sum(v.size for v in self.params.values())

    #forward pass
    def forward(self, X: np.ndarray) -> tuple:
        """
        X : (N, 4)  columns = [x, y, t, z]
        Returns out (N,), cache.
        """
        p = self.params

        x0 = X[:, 0:1]   #(N,1)
        y0 = X[:, 1:2]
        t0 = X[:, 2:3]
        z0 = X[:, 3:4]

        #satellite branches (1 layer each)
        #y branch: eq.6  →  y1 = σ_mc(y0 @ |W_yy0|.T + b_y0)
        zy0 = y0 @ np.abs(p['W_yy0']).T + p['b_y0']   #(N, Nh)
        y1  = softplus(zy0)

        #z branch: eq.7  →  z1 = σ_a(z0 @ W_zz0.T + b_z0)
        zz0 = z0 @ p['W_zz0'].T + p['b_z0']           #(N, Nh)
        z1  = sigmoid(zz0)

        #t branch: eq.8  →  t1 = σ_m(t0 @ |W_tt0|.T + b_t0)
        zt0 = t0 @ np.abs(p['W_tt0']).T + p['b_t0']   #(N, Nh)
        t1  = sigmoid(zt0)

        #x first layer: eq.9 (raw scalar inputs)
        pre_x1 = (x0 @ p['W_xx_first'].T  + p['b_xx_first']
                + y0 @ np.abs(p['W_xy_first']).T
                + z0 @ p['W_xz_first'].T
                + t0 @ np.abs(p['W_xt_first']).T)      #(N, Nh)
        x1 = softplus(pre_x1)

        #x last layer: eq.10 (processed states, linear out)
        out = (x1 @ np.abs(p['W_xx_last']).T
             + x0 @ p['W_xx0_skip'].T  + p['b_skip']
             + y1 @ np.abs(p['W_xy_last']).T
             + z1 @ p['W_xz_last'].T
             + t1 @ np.abs(p['W_xt_last']).T)          #(N, 1)

        cache = {
            'x0': x0, 'y0': y0, 't0': t0, 'z0': z0,
            #satellite branches
            'zy0': zy0, 'y1': y1,
            'zz0': zz0, 'z1': z1,
            'zt0': zt0, 't1': t1,
            #x branch
            'pre_x1': pre_x1, 'x1': x1,
        }
        return out.ravel(), cache   #(N,)

    #backward pass
    def backward(self, grad_out: np.ndarray, cache: dict) -> dict:
        p = self.params
        g = {k: np.zeros_like(v) for k, v in p.items()}

        #step 1: x last layer (linear, no activation)
        d = grad_out.reshape(-1, 1)           #(N, 1)

        #weight gradients for x last layer
        g['W_xx_last'] = (d.T @ cache['x1']) * np.sign(p['W_xx_last'])   #(1,Nh)
        g['W_xx0_skip'] = d.T @ cache['x0']                              #(1, 1)
        g['b_skip'] = d.sum(axis=0)                                      #(1,)
        g['W_xy_last'] = (d.T @ cache['y1']) * np.sign(p['W_xy_last'])   #(1,Nh)
        g['W_xz_last'] = d.T @ cache['z1']                               #(1,Nh)
        g['W_xt_last'] = (d.T @ cache['t1']) * np.sign(p['W_xt_last'])   #(1,Nh)

        #propagate to inputs of x last layer
        dx1 = d @ np.abs(p['W_xx_last'])            #(N, Nh) - into x first layer
        dy1 = d @ np.abs(p['W_xy_last'])            #(N, Nh) - into y branch
        dz1 = d @ p['W_xz_last']                    #(N, Nh) - into z branch
        dt1 = d @ np.abs(p['W_xt_last'])            #(N, Nh) - into t branch
        #dx0_skip goes into the raw input x0 - no weight to update there

        #step 2: x first layer (softplus)
        #δ_x1 = dx1 ⊙ softplus'(pre_x1)
        d = dx1 * softplus_d(cache['pre_x1'])        #(N, Nh)

        #weight gradients for x first layer
        g['W_xx_first'] = d.T @ cache['x0']                                 #(Nh, 1)
        g['b_xx_first'] = d.sum(axis=0)                                     #(Nh,)
        g['W_xy_first'] = (d.T @ cache['y0']) * np.sign(p['W_xy_first'])    #(Nh, 1)
        g['W_xz_first'] = d.T @ cache['z0']                                 #(Nh, 1)
        g['W_xt_first'] = (d.T @ cache['t0']) * np.sign(p['W_xt_first'])    #(Nh, 1)
        #note: x0 and raw y0/z0/t0 are inputs - don't propagate further.

        #step 3: satellite branches backward
        #y branch: y1 = softplus(zy0),  zy0 = y0 @ |W_yy0|.T + b_y0
        #δ_y = dy1 ⊙ softplus'(zy0)
        d = dy1 * softplus_d(cache['zy0']) #(N, Nh)
        g['W_yy0'] = (d.T @ cache['y0']) * np.sign(p['W_yy0']) #(Nh, 1)
        g['b_y0'] = d.sum(axis=0) #(Nh,)

        #z branch: z1 = sigmoid(zz0),  zz0 = z0 @ W_zz0.T + b_z0
        d = dz1 * sigmoid_d(cache['zz0']) #(N, Nh)
        g['W_zz0'] = d.T @ cache['z0'] #(Nh, 1)
        g['b_z0']  = d.sum(axis=0)

        #t branch: t1 = sigmoid(zt0),  zt0 = t0 @ |W_tt0|.T + b_t0
        d = dt1 * sigmoid_d(cache['zt0'])  #(N, Nh)
        g['W_tt0'] = (d.T @ cache['t0']) * np.sign(p['W_tt0']) #(Nh, 1)
        g['b_t0']  = d.sum(axis=0)

        return g

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        N = y.shape[0]
        pred, cache = self.forward(X)
        loss = float(np.mean((pred - y) ** 2))
        grad_out = (2.0 / N) * (pred - y)
        grads = self.backward(grad_out, cache)
        self.opt.step(self.params, grads)
        return loss

#gradient check
def gradient_check(ModelClass, model_kwargs: dict, tol: float = 1e-4) -> bool:
    #Compares analytic gradients from backward() against numerical finite differences. Prints result per parameter key.
    np.random.seed(999)
    model = ModelClass(**model_kwargs, seed=999)
    N = 8
    X = np.random.randn(N, 4) * 0.5 + 1.0   #keep +ve for softplus stability
    y = np.random.randn(N)
    eps = 1e-5
    all_ok = True

    pred, cache = model.forward(X)
    loss0 = float(np.mean((pred - y) ** 2))
    grad_out = (2.0 / N) * (pred - y)
    grads = model.backward(grad_out, cache)

    print(f"  Gradient check for {ModelClass.__name__}:")
    for key in list(model.params.keys())[:6]:   #check first 6 params only (speed)
        param = model.params[key]
        grad = grads[key]
        orig = param.copy()
        max_err = 0.0

        for idx in np.ndindex(param.shape):
            param[idx] = orig[idx] + eps
            Lp = float(np.mean((model.predict(X) - y) ** 2))
            param[idx] = orig[idx] - eps
            Lm = float(np.mean((model.predict(X) - y) ** 2))
            param[idx] = orig[idx]

            num = (Lp - Lm) / (2 * eps)
            ana = grad[idx]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-10)
            max_err = max(max_err, rel)

        ok = max_err < tol
        print(f"{key:20s}  max_rel_err={max_err:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    return all_ok


#training loop
def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30_000,
    eval_every: int = 50,
) -> tuple:
    train_losses: list[float] = []
    test_checkpoints: list[tuple] = []   #(epoch_idx, loss)

    for ep in range(epochs):
        loss_tr = model.train_step(X_train, y_train)
        train_losses.append(loss_tr)

        if ep % eval_every == 0 or ep == epochs - 1:
            pred_te = model.predict(X_test)
            loss_te = float(np.mean((pred_te - y_test) ** 2))
            test_checkpoints.append((ep, loss_te))

    ck_ep = np.array([c[0] for c in test_checkpoints], dtype=float)
    ck_val = np.array([c[1] for c in test_checkpoints], dtype=float)
    test_losses = np.interp(np.arange(epochs, dtype=float), ck_ep, ck_val)

    return np.array(train_losses), test_losses


def run_seeds(
    ModelClass,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30_000,
    n_seeds: int = 10,
    name: str = "",
) -> dict:
    all_tr, all_te, models = [], [], []

    print(f"\n  ── {name:6s}  ({n_seeds} seeds × {epochs} epochs) ──")

    for seed in range(n_seeds):
        np.random.seed(seed)
        model = ModelClass(**model_kwargs, seed=seed)

        t0 = time.time()
        tr, te = train_model(model, X_train, y_train, X_test, y_test,
                              epochs=epochs)
        elapsed = time.time() - t0

        all_tr.append(tr)
        all_te.append(te)
        models.append(model)

        print(f"     seed {seed+1:02d}/{n_seeds}  |  "
              f"train MSE={tr[-1]:.3e}  test MSE={te[-1]:.3e}  "
              f"[{elapsed:.1f}s]")

    return {
        "train_losses": np.array(all_tr),   #(n_seeds, epochs)
        "test_losses":  np.array(all_te),
        "models":       models,
    }


#plotting
MODEL_COLORS = {"FFNN": "tab:red", "ISNN-1": "tab:green", "ISNN-2": "tab:blue"}

def _mean_std(arr: np.ndarray) -> tuple:
    return arr.mean(axis=0), arr.std(axis=0)

def plot_losses(
    results: dict,
    title: str,
    save_path: str,
) -> None:
    #loss curves - identical to pytorch_implementation version
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_behavior(
    results: dict,
    true_func,
    train_end: float,
    test_end:  float,
    title: str,
    save_path: str,
) -> None:
    n_pts  = 300
    vals   = np.linspace(0.0, test_end, n_pts)
    diag   = np.column_stack([vals] * 4)       # (n_pts, 4)
    true_y = true_func(diag)

    split = np.searchsorted(vals, train_end)

    model_names = list(results.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        res   = results[name]
        preds = []

        for m in res["models"]:
            p = m.predict(diag)   #(n_pts,) - numpy array
            preds.append(p)

        preds = np.array(preds)       #(n_seeds, n_pts)
        mu, sd = preds.mean(0), preds.std(0)

        ax.plot(vals, true_y, "k--", lw=1.8, label="True response", zorder=5)

        ax.plot(vals[:split], mu[:split], color="steelblue", lw=1.5,
                label="Interpolated response")
        ax.fill_between(vals[:split],
                        mu[:split] - sd[:split],
                        mu[:split] + sd[:split],
                        color="steelblue", alpha=0.25)

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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def main(epochs: int = 30_000, n_seeds: int = 10) -> None:

    out_dir = "figures_numpy"
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "=" * 65)
    print("  Model parameter counts  (NumPy implementation)")
    print("=" * 65)
    for Cls, kw, nm in [
        (FFNN, {}, "FFNN  "),
        (ISNN1, {"n_hidden": 10, "n_layers": 2}, "ISNN-1"),
        (ISNN2, {"n_hidden": 15, "H": 2}, "ISNN-2"),
    ]:
        n = Cls(**kw).count_params()
        print(f"  {nm} : {n:>5} trainable parameters")
    print("=" * 65)

    #gradient checks (verifies backprop is correct)
    print("\n" + "=" * 65)
    print("Gradient checks (analytic vs finite-difference)")
    print("=" * 65)
    gradient_check(FFNN,{})
    gradient_check(ISNN1, {"n_hidden": 10, "n_layers": 2})
    gradient_check(ISNN2, {"n_hidden": 15, "H": 2})

    MODEL_DEFS = [
        ("FFNN", FFNN,  {}),
        ("ISNN-1", ISNN1, {"n_hidden": 10, "n_layers": 2}),
        ("ISNN-2", ISNN2, {"n_hidden": 15, "H": 2}),
    ]

    #toy dataset 1 - additive function (eq. 12)
    print("\n" + "=" * 65)
    print("Toy Dataset 1  -  Additive function  (eq. 12)")
    print("=" * 65)

    X_tr1, y_tr1, X_te1, y_te1 = generate_toy1()

    results1 = {}
    for name, Cls, kw in MODEL_DEFS:
        results1[name] = run_seeds(
            Cls, kw,
            X_tr1, y_tr1, X_te1, y_te1,
            epochs=epochs, n_seeds=n_seeds, name=name,
        )

    print("\nGenerating loss plot …")
    plot_losses(
        results1,
        title="Toy Dataset 1 – Additive function [NumPy]",
        save_path=f"{out_dir}/fig3_toy1_losses.png",
    )
    print("Generating behavior plot …")
    plot_behavior(
        results1, true_func=f_toy1,
        train_end=4.0, test_end=6.0,
        title="Toy Dataset 1 - Behavioral response along diagonal [NumPy]",
        save_path=f"{out_dir}/fig4_toy1_behavior.png",
    )

    #toy dataset 2 - multiplicative function (eq. 13)
    print("\n" + "=" * 65)
    print("Toy Dataset 2  -  Multiplicative function  (eq. 13)")
    print("=" * 65)

    X_tr2, y_tr2, X_te2, y_te2 = generate_toy2()

    results2 = {}
    for name, Cls, kw in MODEL_DEFS:
        results2[name] = run_seeds(
            Cls, kw,
            X_tr2, y_tr2, X_te2, y_te2,
            epochs=epochs, n_seeds=n_seeds, name=name,
        )

    print("\nGenerating loss plot …")
    plot_losses(
        results2,
        title="Toy Dataset 2 – Multiplicative function [NumPy]",
        save_path=f"{out_dir}/fig5_toy2_losses.png",
    )
    print("Generating behavior plot …")
    plot_behavior(
        results2, true_func=f_toy2,
        train_end=4.0, test_end=10.0,
        title="Toy Dataset 2 - Behavioral response along diagonal [NumPy]",
        save_path=f"{out_dir}/fig6_toy2_behavior.png",
    )

    print("\n" + "=" * 65)
    print("All figures written to:", out_dir)
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NumPy FFNN / ISNN-1 / ISNN-2 with manual backpropagation"
    )
    parser.add_argument(
        "--epochs", type=int, default=30_000,
        help=(
            "Training epochs per seed (paper uses 30000). "
            "CPU estimate: ~2-4 min/seed for 30000 epochs. "
            "Use --epochs 2000 for a quick smoke-test (~10s/seed)."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of random initialisations (default 10)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run gradient checks only, then exit (no training)",
    )
    args = parser.parse_args()

    if args.check_only:
        print("\nRunning gradient checks only …\n")
        gradient_check(FFNN,  {})
        gradient_check(ISNN1, {"n_hidden": 10, "n_layers": 2})
        gradient_check(ISNN2, {"n_hidden": 15, "H": 2})
    else:
        main(epochs=args.epochs, n_seeds=args.seeds)

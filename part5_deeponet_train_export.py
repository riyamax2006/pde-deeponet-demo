import numpy as np
import torch
import torch.nn as nn
import json

np.random.seed(0)
torch.manual_seed(0)

def solve_ade(a, k, f, N, u0=0, u1=1):
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    u = np.zeros(N)
    alpha = -a/2/h - k/h**2
    beta = 2*k/h**2
    gamma = a/2/h - k/h**2
    A = np.diag(np.full(N-2, beta), k=0) + \
        np.diag(np.full(N-3, alpha), k=-1) + \
        np.diag(np.full(N-3, gamma), k=1)
    RHS = f(x[1:-1]).reshape(-1, 1)
    RHS[0] -= alpha * u0
    RHS[-1] -= gamma * u1
    u[1:-1] = np.linalg.solve(A, RHS).flatten()
    u[0] = u0
    u[-1] = u1
    return u

kappa, a_coef = 0.1, 1.0

def sample_grf(n_samples, grid, length_scale=0.1):
    diff = grid.reshape(-1,1) - grid.reshape(1,-1)
    cov = np.exp(-(diff**2) / (2*length_scale**2))
    L = np.linalg.cholesky(cov + 1e-10*np.eye(len(grid)))
    z = np.random.randn(len(grid), n_samples)
    samples = L @ z
    samples = samples - samples.min(axis=0, keepdims=True)
    samples = samples / samples.max(axis=0, keepdims=True)
    samples = 2*samples - 1
    return samples.T

N_grid = 101
x_grid = np.linspace(0, 1, N_grid)
N_sensors = 21
x_sensors = np.linspace(0, 1, N_sensors)
N_train = 1500
p = 10  # using the good value found in the sweep

f_samples = sample_grf(N_train, x_grid, length_scale=0.1)
U_train = np.zeros((N_train, N_grid))
F_train = np.zeros((N_train, N_sensors))
for n in range(N_train):
    f_vals = f_samples[n]
    f_interp = lambda xx, f_vals=f_vals: np.interp(xx, x_grid, f_vals)
    U_train[n] = solve_ade(a=a_coef, k=kappa, f=f_interp, N=N_grid)
    F_train[n] = f_interp(x_sensors)

class Branch(nn.Module):
    def __init__(self, n_sensors, p, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p)
        )
    def forward(self, F): return self.net(F)

class Trunk(nn.Module):
    def __init__(self, p, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p)
        )
    def forward(self, x): return self.net(x)

F_t = torch.tensor(F_train, dtype=torch.float32)
U_t = torch.tensor(U_train, dtype=torch.float32)
x_t = torch.tensor(x_grid, dtype=torch.float32).reshape(-1,1)

branch = Branch(N_sensors, p)
trunk = Trunk(p)
optimizer = torch.optim.Adam(list(branch.parameters()) + list(trunk.parameters()), lr=1e-3)
for epoch in range(4000):
    optimizer.zero_grad()
    B = branch(F_t); T = trunk(x_t)
    U_pred = B @ T.T
    loss = torch.mean((U_pred - U_t)**2)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(epoch, loss.item())

print("final loss:", loss.item())

# ---- export weights to JSON for JS reimplementation ----
def export_seq(seq):
    layers = []
    for layer in seq:
        if isinstance(layer, nn.Linear):
            layers.append({
                "type": "linear",
                "W": layer.weight.detach().numpy().tolist(),  # (out, in)
                "b": layer.bias.detach().numpy().tolist()
            })
        elif isinstance(layer, nn.Tanh):
            layers.append({"type": "tanh"})
    return layers

model_json = {
    "kappa": kappa,
    "a": a_coef,
    "x_sensors": x_sensors.tolist(),
    "branch_layers": export_seq(branch.net),
    "trunk_layers": export_seq(trunk.net),
}

with open('deeponet_weights.json', 'w') as fjson:
    json.dump(model_json, fjson)

print("Exported weights to JSON")
print("File size:", __import__('os').path.getsize('deeponet_weights.json'), "bytes")

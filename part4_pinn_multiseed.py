import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

kappa = 0.1
a_coef = 1.0
f_np = lambda x: np.sin(2*np.pi*x)
f_torch = lambda x: torch.sin(2*np.pi*x)

class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_pinn(N_int, lambda_b=10.0, epochs=6000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x0 = torch.tensor([[0.0]])
    x1 = torch.tensor([[1.0]])

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_int = torch.rand(N_int, 1, requires_grad=True)

        u = model(x_int)
        du_dx = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_int, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

        residual = -kappa * d2u_dx2 + a_coef * du_dx - f_torch(x_int)
        loss_res = torch.mean(residual**2)

        loss_b = (model(x0) - 0.0)**2 + (model(x1) - 1.0)**2
        loss_b = loss_b.squeeze()

        loss = loss_res + lambda_b * loss_b
        loss.backward()
        optimizer.step()

    return model

# ---------- fine FD reference ----------
x_fine = np.linspace(0, 1, 1000)
u_fine = solve_ade(a=1, k=0.1, f=f_np, N=1000)
x_fine_t = torch.tensor(x_fine, dtype=torch.float32).reshape(-1, 1)

# ---------- run each N_int across multiple seeds, average the error ----------
N_ints = [10, 20, 100]
seeds = [0, 1, 2, 3, 4]

results = {}   # N_int -> list of (mean_err, max_err) per seed
best_models = {}  # keep one representative model per N_int (seed=0) for plotting

for N_int in N_ints:
    errs_mean, errs_max = [], []
    for seed in seeds:
        model = train_pinn(N_int, lambda_b=10.0, seed=seed)
        with torch.no_grad():
            u_pred = model(x_fine_t).numpy().flatten()
        err = np.abs(u_pred - u_fine)
        errs_mean.append(err.mean())
        errs_max.append(err.max())
        if seed == 0:
            best_models[N_int] = model
        print(f"N_int={N_int}, seed={seed}: mean_err={err.mean():.5f}, max_err={err.max():.5f}")
    results[N_int] = (errs_mean, errs_max)

print()
print(f"{'N_int':>6} | {'avg mean err':>12} | {'std mean err':>12} | {'avg max err':>12}")
for N_int in N_ints:
    errs_mean, errs_max = results[N_int]
    print(f"{N_int:>6} | {np.mean(errs_mean):>12.5f} | {np.std(errs_mean):>12.5f} | {np.mean(errs_max):>12.5f}")

# ---------- plot: error bars across seeds ----------
plt.figure(figsize=(6,5))
means = [np.mean(results[N][0]) for N in N_ints]
stds = [np.std(results[N][0]) for N in N_ints]
plt.errorbar(N_ints, means, yerr=stds, fmt='o-', capsize=5)
plt.xlabel('N_int (collocation points)')
plt.ylabel('Mean |error| (averaged over 5 seeds)')
plt.title('PINN accuracy vs N_int, averaged over 5 random seeds')
plt.xscale('log')
plt.savefig('/home/claude/pinn_multiseed_result.png', dpi=100)
print("saved plot")

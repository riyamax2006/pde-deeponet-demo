import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        x_int = torch.rand(N_int, 1, requires_grad=True)  # resample each epoch

        u = model(x_int)
        du_dx = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_int, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

        # NOTE: using -kappa (matches the given FD solver's actual convention)
        residual = -kappa * d2u_dx2 + a_coef * du_dx - f_torch(x_int)
        loss_res = torch.mean(residual**2)

        loss_b = (model(x0) - 0.0)**2 + (model(x1) - 1.0)**2
        loss_b = loss_b.squeeze()

        loss = loss_res + lambda_b * loss_b
        loss.backward()
        optimizer.step()

    return model

N_ints = [10, 20, 100]
models = {}
for N_int in N_ints:
    models[N_int] = train_pinn(N_int, lambda_b=10.0)
    print(f"Trained PINN with N_int={N_int}")

x_fine = np.linspace(0, 1, 1000)
u_fine = solve_ade(a=1, k=0.1, f=f_np, N=1000)
x_fine_t = torch.tensor(x_fine, dtype=torch.float32).reshape(-1, 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_fine, u_fine, 'k-', label='FD reference (N=1000)', linewidth=2)
for N_int in N_ints:
    with torch.no_grad():
        u_pred = models[N_int](x_fine_t).numpy().flatten()
    plt.plot(x_fine, u_pred, '--', label=f'PINN N_int={N_int}')
plt.xlabel('x'); plt.ylabel('u(x)'); plt.legend()
plt.title('PINN predictions vs FD reference')

plt.subplot(1, 2, 2)
for N_int in N_ints:
    with torch.no_grad():
        u_pred = models[N_int](x_fine_t).numpy().flatten()
    error = np.abs(u_pred - u_fine)
    print(f"N_int={N_int}: mean error = {error.mean():.5f}, max error = {error.max():.5f}")
    plt.plot(x_fine, error, label=f'PINN N_int={N_int}')
plt.xlabel('x'); plt.ylabel('|error|'); plt.legend()
plt.title('Pointwise error vs FD reference')
plt.tight_layout()
plt.savefig('outputs/part4_pinn_result.png', dpi=150)
plt.show()
print("done")

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

f = lambda x: np.sin(2*np.pi*x)

Ns = [10, 20, 100]
datasets = {}
for N in Ns:
    x = np.linspace(0, 1, N)
    u = solve_ade(a=1, k=0.1, f=f, N=N)
    datasets[N] = (x, u)

class MLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(x_train, u_train, epochs=3000, lr=1e-3):
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_t = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1)
    u_t = torch.tensor(u_train, dtype=torch.float32).reshape(-1, 1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(x_t)
        loss = loss_fn(u_pred, u_t)
        loss.backward()
        optimizer.step()
    return model

models = {}
for N in Ns:
    x_train, u_train = datasets[N]
    models[N] = train_mlp(x_train, u_train)
    print(f"Trained MLP on N={N} dataset")

x_fine = np.linspace(0, 1, 1000)
u_fine = solve_ade(a=1, k=0.1, f=f, N=1000)
x_fine_t = torch.tensor(x_fine, dtype=torch.float32).reshape(-1, 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_fine, u_fine, 'k-', label='FD reference (N=1000)', linewidth=2)
for N in Ns:
    with torch.no_grad():
        u_pred = models[N](x_fine_t).numpy().flatten()
    plt.plot(x_fine, u_pred, '--', label=f'NN trained on N={N}')
plt.xlabel('x'); plt.ylabel('u(x)'); plt.legend()
plt.title('Predicted solutions vs FD reference')

plt.subplot(1, 2, 2)
for N in Ns:
    with torch.no_grad():
        u_pred = models[N](x_fine_t).numpy().flatten()
    error = np.abs(u_pred - u_fine)
    print(f"N={N}: mean error = {error.mean():.5f}, max error = {error.max():.5f}")
    plt.plot(x_fine, error, label=f'NN trained on N={N}')
plt.xlabel('x'); plt.ylabel('|error|'); plt.legend()
plt.title('Pointwise error vs FD reference')

plt.tight_layout()
import os
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/part3_result.png', dpi=150)
plt.show()

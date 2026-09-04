[README.md](https://github.com/user-attachments/files/31819153/README.md)
# Neural Networks for the Advection-Diffusion Equation

Exploring three deep learning approaches to solving a 1D advection-diffusion
equation (ADE), compared against a classical finite-difference (FD) baseline:

1. **Data-driven regression** — a plain MLP trained on labeled FD solutions
2. **Physics-Informed Neural Networks (PINN)** — trained using only the PDE
   residual and boundary conditions, no labeled data
3. **DeepONet (operator learning)** — trained to solve the equation for *any*
   source function `f(x)`, generalizing without retraining

**[Live interactive demo →](#)** *https://riyamax2006.github.io/pde-deeponet-demo/*
Type any source function `f(x)` and see the trained DeepONet's prediction
compared live against a fresh finite-difference solve — runs entirely in
your browser, no backend.

## The equation

```
κ u''(x) + a u'(x) = f(x),   x ∈ [0, 1]
u(0) = 0,  u(1) = 1
```

with `κ = 0.1`, `a = 1` fixed throughout. Parts 3–4 use a fixed
`f(x) = sin(2πx)`; Part 5 trains across many random `f`.

> **Note on sign convention:** the FD solver's coefficients were confirmed
> (by algebraic expansion) to implement `-κu'' + au' = f`, not `+κu'' + au' = f`.
> The PINN residual in `part4_pinn.py` uses `-κ` to stay consistent with the
> FD solver, since it is the reference solution used throughout.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running each part

```bash
python part3_regression_nn.py          # data-driven regression NN
python part4_pinn.py                   # PINN, single run
python part4_pinn_multiseed.py         # PINN, averaged over 5 seeds (see RESULTS.md)
python part5_deeponet_train_export.py  # DeepONet, also writes deeponet_weights.json
```

Each script prints progress and saves a result plot into `outputs/`.

## Files

| File | Description |
|---|---|
| `part3_regression_nn.py` | FD solver + data-driven MLP regression, compared across training grid resolutions (N=10, 20, 100) |
| `part4_pinn.py` | PINN trained via PDE residual + boundary loss, single seeded run |
| `part4_pinn_multiseed.py` | Same PINN setup, averaged over 5 random seeds — see [RESULTS.md](RESULTS.md) for why this matters |
| `part5_deeponet_train_export.py` | DeepONet trained on 1500 random source functions, exports trained weights to `deeponet_weights.json` |
| `index.html` | Standalone interactive demo — the trained DeepONet + a live FD solver, both reimplemented in plain JavaScript. Just open in a browser. |
| `build_explorer_html.py` | Regenerates `index.html` from `deeponet_weights.json` |
| `RESULTS.md` | Full write-up of findings across all three methods |

## Using your own trained DeepONet weights in the demo

```bash
python part5_deeponet_train_export.py   # writes a fresh deeponet_weights.json
python build_explorer_html.py           # rebuilds index.html using it
```

## Summary of findings

See [RESULTS.md](RESULTS.md) for full details, plots, and tables. Headline points:

- **Data-driven regression:** accuracy improves smoothly and reproducibly with
  more training data (N=10 → N=100 cut mean error roughly 5-6x).
- **PINN:** achieves comparable or better accuracy with *zero* labeled data,
  using only the PDE residual. Unlike Part 3, accuracy showed **no
  statistically significant dependence on the number of collocation points**
  once averaged over multiple random seeds — a single run can misleadingly
  suggest a trend that doesn't hold up.
- **DeepONet:** trades a small amount of per-instance accuracy for the
  ability to solve for entirely new source functions instantly, without
  retraining — the key advantage of operator learning.

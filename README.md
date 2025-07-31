# vPET-ABC-JAX  
Fast, likelihood‑free PET kinetic modelling implemented in **JAX**

---

## 1 · What is this repository?

`vpetabc` is a pure‑Python re‑implementation of the **vPET‑ABC** framework&nbsp;[(Grazian *et&nbsp;al.*, 2021)](https://ieeexplore.ieee.org/document/9875446/, peer-reviewed paper coming soon) for large‑scale dynamic PET kinetic modelling, written from the ground up in **[JAX](https://github.com/google/jax)**.

Compared with the earlier CuPy version, the JAX rewrite

* removes CUDA‑specific boiler‑plate – the same code runs on CPU, multi‑GPU, or TPU via XLA;  
* exposes a clean, PyTorch‑like API centred on an abstract `KineticModel`;  
* relies on vectorised primitives (`vmap`, `lax.scan`) so that even > 40 M‑voxel datasets fit into a single JIT‑compiled graph;  
* delivers further speed‑ups;  
* depends only on `jax`, `pandas`, and `tqdm` – no CuPy, no manual builds.

---

## 2 · Repository layout

```
.
├── data/
│   ├── sample_2TCM.csv
│   └── sample_lpntPET.csv
├── dist/                     # wheels/sdists created by `python -m build`
├── example_usage.ipynb
├── pyproject.toml
├── README.md
└── src/
    └── vpetabc/
        ├── __init__.py       # package namespace
        ├── engine.py         # ABC engine + helpers
        ├── models.py         # TwoTissueModel, lpntPETModel, …
        ├── priors.py         # prior samplers
        └── utilities.py      # I/O + posterior utilities
```

| Module | Description |
|--------|-------------|
| `engine.py` | `ABCRejection`, the fully vectorised, JIT‑compiled rejection‑ABC driver |
| `models.py` | `KineticModel` base‑class + `TwoTissueModel`, `lpntPETModel` implementations |
| `priors.py` | Uniform × Bernoulli prior samplers (`TwoTissuePrior`, `lpntPETPrior`) |
| `utilities.py` | `preprocess_table`, `get_conditional_posterior_mean`, misc. helpers |

---

## 3 · Installation

> **TL;DR** `pip install vpet-abc[cuda]`

### 3.1  Stable release from PyPI

```bash
# create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# CPU‑only:
pip install vpet-abc

# NVIDIA GPUs
pip install vpet-abc[cuda]
```

`jax[cuda]` wheels already bundle matching CUDA/cuDNN libraries; you only need a driver on Linux / Windows. For TPU, Metal (macOS), or ROCm see the official  
[JAX installation guide](https://github.com/google/jax#installation).

### 3.2  Tested environments

| OS | Python | `jax` / `jaxlib` | Accelerator |
|----|--------|------------------|-------------|
| macOS 15.5 (arm64) | 3.11.12 | 0.6.1 (CPU) | Apple M2 |
| Rocky Linux 8.10 | 3.9.2 | 0.4.30 [cuda12] | NVIDIA V100 |

---

## 4 · Quick start

See **`example_usage.ipynb`** for an executable walkthrough, or run:

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from vpetabc import *

df = pd.read_csv("data/sample_2TCM.csv", index_col=0)
Cp_fine, A, TACs, _ = preprocess_table(df)

engine = ABCRejection(
    TwoTissueModel(),
    prior_sampler = TwoTissuePrior,
    num_sims      = 200_000,
    accept_frac   = 0.005,
)

post = engine.run(jr.PRNGKey(0), TACs, Cp_fine, A, batch_size=50_000)
means, chosen = get_conditional_posterior_mean(post)
print(means[:3])            # first 3 voxels
```

---

## 5 · Extending the framework

1. **Define your kinetic model**

```python
class MyModel(KineticModel):
    @partial(jax.jit, static_argnums=(0,))
    def simulate(self, θ, Cp, dt):
        # return Ct(t) as a (T_fine,) array
        ...
```

2. **Write a prior**

```python
def MyPrior(key, n, lows, highs):
    return jr.uniform(key, (n, P), lows, highs)
```

3. **Pass `MyModel` and `MyPrior` to `ABCRejection`.**  
Batching, GPU kernels, and distance evaluation are handled automatically.

---

## 6 · Benchmarks

*TBD*
Current estimates believe inference on 4.4 million voxels for a simulation size of 10,000,000 takes no more than 2 hours on 4 A100 GPUs, and 11.8 hours on one V100 GPU.

---

## 7 · Citation

*TBA soon.*

---

## 8 · Licence

`vpetabc` is released under the MIT Licence (see `LICENCE`).  
The sample dataset is provided for non‑commercial research use only.

# vPET-ABC-JAX  
Fast, likelihood‑free PET kinetic modelling implemented in **JAX**

---

## 1. What is this repository?

`vpetabc` is a pure‑Python re‑implementation of the **vPET‑ABC** framework&nbsp;[(Grazian *et&nbsp;al.*, 2021)](https://ieeexplore.ieee.org/document/9875446/, peer-reviewed full paper coming out soon.) for large‑scale dynamic PET kinetic modelling, written from the ground up in **[JAX](https://github.com/google/jax)**.  
Compared with the older CuPy code‑base, the new design

* removes all CUDA‑specific boiler‑plate – the same code runs on CPU, multi‑GPU, or TPU via XLA;  
* exposes a clean, PyTorch‑like API centred on an abstract `KineticModel`;  
* uses *single‑kernel* vectorised primitives (`vmap`, `lax.scan`) so that even >40 M‑voxel whole‑body datasets fit into a single JIT‑compiled graph;  
* delivers further speed‑ups;  
* depends only on `jax`, `pandas` and `tqdm` – no CuPy, no manual builds.

---

## 2. Repository layout

```
.
├── abc_kinetic.py       # core models & ABC engine
├── examples/
│   └── run_2tcm.py      # minimal end‑to‑end script
└── data/
    └── FDG_sample.csv   # toy dataset (2‑TCM, 3 voxels)
```

| Module | Description |
|--------|-------------|
| `KineticModel` | Abstract base; implement `simulate()` once, get batched GPU kernels for free |
| `TwoTissueModel` | 5‑parameter irreversible/reversible 2‑TCM (4‑stage RK integration) |
| `lpntPETModel` | 7‑parameter lp‑ntPET activation model |
| `ABCRejection` | Vectorised, JIT‑compiled rejection ABC with fixed simulation budget |
| `preprocess_table`, `get_conditional_posterior_mean` | I/O helpers for TAC tables |
| `TwoTissuePrior`, `lpntPETPrior` | Example uniform × Bernoulli priors |

---

## 3. Installation

### 3.1 Quick recipe (CUDA 12, Linux)
```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install vpet-abc-jax
```

Alternatively, install manually:

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install "jax[cuda]"
pip install pandas tqdm
```
And then Git Clone the repo.

The JAX wheel already bundles the correct CUDA & cuDNN libraries; you only need a proper NVIDIA driver.  
For other accelerators (CPU‑only, TPU, Metal, ROCm) see the [JAX installation guide](https://github.com/google/jax#installation).

### 3.2 Tested environments

| OS | Python | `jax` / `jaxlib` | Accelerator |
|----|--------|------------------|-------------|
| MacOS 15.5 | 3.11.12 | 0.6.1 | Apple M2 CPU |
| Rocky Linux release 8.10 (Green Obsidian) | 3.9.2 | 0.4.30[cuda] | NVIDIA V100 |
---

## 4. Quick start

See example usage.ipynb.

---

## 5. Extending the framework

1. **Define your own kinetic model**

```python
class MyModel(KineticModel):
    @partial(jax.jit, static_argnums=(0,))
    def simulate(self, θ, Cp, dt):
        # return Ct(t) as (T_fine,) array
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

## 6. Benchmarks

TBD

---

## 7. Citation

```text
TBA soon.
```

---

## 8. Licence

`vpetabc` is released under the MIT Licence (see `LICENCE`).  
The sample dataset is provided for non‑commercial research use only.
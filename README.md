# SQGEN — Synergic Quantum Generative Machine Learning

Qiskit implementations of the quantum circuits from the paper
["Synergic quantum generative machine learning" (arXiv:2112.13255v2)](https://arxiv.org/abs/2112.13255).

The example task is training both recognition and generation of an *n*-qubit GHZ entangled state.

## Contents

| File | Description |
|------|-------------|
| `sqgen.py` | Shared module: generator, discriminator, cost functions (Qiskit path) |
| `sqgen_fast.py` | Numba JIT fast simulator + optimized training (2652x faster) |
| `multiqubit_n5seed103.py` | SQGEN and QGAN training on a statevector simulator (*n* = 5) |
| `singlequbit_n1.ipynb` | SQGEN training for a single qubit on a real/fake IBM backend |
| `figure_n5seed103.py` | Plotting learning curves from saved `.npy` data |
| `test_sqgen.py` | Test suite for all ansatze and training loops |
| `benchmark.py` | Full benchmark across seeds, qubit counts, and ansatze |

## Installation

```bash
pip install -r requirements.txt
```

Requires **Qiskit >= 1.0** with `qiskit-aer` and `qiskit-ibm-runtime`.
The fast simulator (`sqgen_fast.py`) additionally requires `numba`.

## Usage

### Multi-qubit simulator experiment

```bash
python multiqubit_n5seed103.py
```

Trains SQGEN and QGAN for a 5-qubit GHZ state (seed 103) and saves
probability / fidelity traces as `.npy` files.

### Single-qubit real-hardware experiment

Open `singlequbit_n1.ipynb` in Jupyter and follow the cells.
Requires an IBM Quantum account configured via `qiskit-ibm-runtime`.

### Plotting

```bash
python figure_n5seed103.py
```

Reads `.npy` files produced by the multi-qubit script and generates
SVG/PDF figures.

## Ansatze

Four variational ansatze are available for the generator (and discriminator sub-circuit):

| Ansatz | Params | Description |
|--------|--------|-------------|
| `mcmt` (default) | 4*n* | Original from the paper — multi-controlled RZ/RY gates. Expressive but expensive: O(*n*) CNOTs per gate. |
| `hardware_efficient` | 2*n*×*L* | RY + RZ per qubit + linear CNOT ladder per layer. Nearest-neighbour friendly. |
| `strongly_entangling` | 3*n*×*L* | RX + RY + RZ per qubit + CNOT ring with varying offset per layer. High expressivity. |
| `circular` | 2*n*×*L* | RY + RZ per qubit + circular CNOT ring per layer. Good balance of cost and entanglement. |

Set `ANSATZ` and `N_LAYERS` in the scripts or notebook to switch.

## Fast simulator

`sqgen_fast.py` provides a Numba JIT-compiled statevector simulator that
bypasses Qiskit circuit overhead entirely (2652x speedup on n=3).
It also implements the **direct fidelity cost** J = 1 - F, which eliminates
the discriminator circuit and halves the parameter-space dimension.

```python
from sqgen_fast import train_sqgen_fast

fid, evals, params, trace = train_sqgen_fast(
    n_qubits=5,
    ansatz='hardware_efficient',
    n_layers=3,
    max_evals=1500,
    seed=42,
)
print(f"Fidelity: {fid:.6f} in {evals} evaluations")
```

Key optimizations (based on results from the SQGEN research):
- **Numba JIT** gate primitives (RX, RY, RZ, CNOT) on split real/imaginary arrays
- **Pre-allocated work arrays** — zero allocation in the hot loop
- **L-BFGS-B with maxiter=5** per epoch (vs BFGS maxiter=1)
- **Direct fidelity cost** — optimizes the generator alone, no discriminator needed

## Benchmark

Mean fidelity (± std) over 5 random seeds at a fixed budget of 1500 cost-function evaluations, for *n* = 1–5 qubits generating GHZ states.

### QGAN (Qiskit path)

| Ansatz | L | p(n=5) | n=1 | n=2 | n=3 | n=4 | n=5 |
|--------|---|--------|-----|-----|-----|-----|-----|
| `mcmt` | 1 | 40 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.996±0.009 | 0.708±0.266 |
| `hardware_efficient` | 3 | 60 | 1.000±0.000 | 1.000±0.000 | 0.998±0.003 | **0.998±0.002** | **0.969±0.055** |
| `strongly_entangling` | 3 | 90 | 1.000±0.000 | 0.999±0.001 | 0.974±0.021 | 0.652±0.274 | 0.388±0.049 |
| `circular` | 3 | 60 | 1.000±0.000 | 1.000±0.000 | 0.998±0.002 | 0.926±0.157 | 0.957±0.088 |

### SQGEN — Qiskit path (BFGS, circuit-based cost)

| Ansatz | L | p(n=5) | n=1 | n=2 | n=3 | n=4 | n=5 |
|--------|---|--------|-----|-----|-----|-----|-----|
| `mcmt` | 1 | 40 | 1.000±0.000 | 0.898±0.223 | 0.737±0.242 | 0.541±0.097 | 0.588±0.196 |
| `hardware_efficient` | 3 | 60 | 1.000±0.000 | 0.997±0.003 | 0.716±0.236 | 0.579±0.320 | 0.235±0.199 |
| `strongly_entangling` | 3 | 90 | 1.000±0.000 | 0.921±0.048 | 0.733±0.082 | 0.188±0.070 | 0.118±0.075 |
| `circular` | 3 | 60 | 1.000±0.000 | 0.993±0.006 | 0.860±0.072 | 0.501±0.230 | 0.160±0.085 |

### SQGEN — fast path (L-BFGS-B, direct fidelity, JIT)

| Ansatz | L | p(n=5) | n=1 | n=2 | n=3 | n=4 | n=5 |
|--------|---|--------|-----|-----|-----|-----|-----|
| `hardware_efficient` | 3 | 30 | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** |
| `strongly_entangling` | 3 | 45 | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** | 0.920±0.160 | 0.961±0.077 |
| `circular` | 3 | 30 | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** | **1.000±0.000** |

### Key findings

- **Fast path achieves F=1.000 for n=1..5** with `hardware_efficient` and `circular` (L=3) — up from F=0.235/0.160 at n=5 on the Qiskit path.
- The improvement comes from three factors: (1) direct fidelity cost eliminates the discriminator, halving the parameter space; (2) L-BFGS-B with 5 inner iterations converges faster; (3) JIT simulator is 2652x faster, so more effective work per evaluation.
- **QGAN** with Qiskit path: `hardware_efficient` (L=3) performs best at n=4–5.
- **`strongly_entangling`** has too many parameters (45 at n=5) and doesn't fully converge within budget for n=4–5.
- For hardware deployment, `hardware_efficient` and `circular` avoid multi-controlled gates, making them suitable for near-term devices.

Reproduce with: `python benchmark.py` (Qiskit path) or `python test_sqgen.py` (smoke tests).

## References

- K. Bartkiewicz *et al.*, "Synergic quantum generative machine learning",
  [arXiv:2112.13255v2](https://arxiv.org/abs/2112.13255)

## License

MIT

# SQGEN — Synergic Quantum Generative Machine Learning

Qiskit implementations of the quantum circuits from the paper
["Synergic quantum generative machine learning" (arXiv:2112.13255v2)](https://arxiv.org/abs/2112.13255).

The example task is training both recognition and generation of an *n*-qubit GHZ entangled state.

## Contents

| File | Description |
|------|-------------|
| `sqgen.py` | Shared module: generator, discriminator, cost functions |
| `multiqubit_n5seed103.py` | SQGEN and QGAN training on a statevector simulator (*n* = 5) |
| `singlequbit_n1.ipynb` | SQGEN training for a single qubit on a real/fake IBM backend |
| `figure_n5seed103.py` | Plotting learning curves from saved `.npy` data |

## Installation

```bash
pip install -r requirements.txt
```

Requires **Qiskit >= 1.0** with `qiskit-aer` and `qiskit-ibm-runtime`.

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

## Benchmark

Mean fidelity (± std) over 5 random seeds at a fixed budget of 1500 cost-function evaluations, for *n* = 1–5 qubits generating GHZ states.

### QGAN

| Ansatz | L | p(n=5) | n=1 | n=2 | n=3 | n=4 | n=5 |
|--------|---|--------|-----|-----|-----|-----|-----|
| `mcmt` | 1 | 40 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.996±0.009 | 0.708±0.266 |
| `hardware_efficient` | 3 | 60 | 1.000±0.000 | 1.000±0.000 | 0.998±0.003 | **0.998±0.002** | **0.969±0.055** |
| `strongly_entangling` | 3 | 90 | 1.000±0.000 | 0.999±0.001 | 0.974±0.021 | 0.652±0.274 | 0.388±0.049 |
| `circular` | 3 | 60 | 1.000±0.000 | 1.000±0.000 | 0.998±0.002 | 0.926±0.157 | 0.957±0.088 |

### SQGEN

| Ansatz | L | p(n=5) | n=1 | n=2 | n=3 | n=4 | n=5 |
|--------|---|--------|-----|-----|-----|-----|-----|
| `mcmt` | 1 | 40 | 1.000±0.000 | 0.898±0.223 | 0.737±0.242 | 0.541±0.097 | **0.588±0.196** |
| `hardware_efficient` | 3 | 60 | 1.000±0.000 | **0.997±0.003** | 0.716±0.236 | 0.579±0.320 | 0.235±0.199 |
| `strongly_entangling` | 3 | 90 | 1.000±0.000 | 0.921±0.048 | 0.733±0.082 | 0.188±0.070 | 0.118±0.075 |
| `circular` | 3 | 60 | 1.000±0.000 | 0.993±0.006 | **0.860±0.072** | 0.501±0.230 | 0.160±0.085 |

### Key findings

- **QGAN**: `hardware_efficient` (L=3) achieves the best fidelity at n=4–5 within the same evaluation budget, outperforming the original `mcmt` ansatz.
- **SQGEN**: Joint optimization is harder; `mcmt` retains an edge at n=5 thanks to fewer parameters (more epochs per budget). `circular` leads at n=3.
- **`strongly_entangling`** does not scale well — too many parameters (90 at n=5) leaves too few epochs within a fixed budget.
- For hardware deployment, `hardware_efficient` and `circular` avoid multi-controlled gates entirely, making them suitable for near-term devices.

Reproduce with: `python benchmark.py` (full run) or `python test_sqgen.py` (quick smoke tests).

## References

- K. Bartkiewicz *et al.*, "Synergic quantum generative machine learning",
  [arXiv:2112.13255v2](https://arxiv.org/abs/2112.13255)

## License

MIT

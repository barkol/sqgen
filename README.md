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

## References

- K. Bartkiewicz *et al.*, "Synergic quantum generative machine learning",
  [arXiv:2112.13255v2](https://arxiv.org/abs/2112.13255)

## License

MIT

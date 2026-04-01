"""
Shared circuits and cost functions for SQGEN / QGAN training.

Paper: "Synergic quantum generative machine learning" (arXiv:2112.13255v2)
"""

import numpy as np
from numpy import pi, sqrt

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import MCMT, RZGate, RYGate, Initialize
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------------
# Ansatz registry
# ---------------------------------------------------------------------------

ANSATZE = ('mcmt', 'hardware_efficient', 'strongly_entangling', 'circular')


def generator_num_params(n, ansatz='mcmt', n_layers=1):
    """Return the number of variational parameters for the chosen ansatz."""
    if ansatz == 'mcmt':
        return 4 * n
    elif ansatz == 'hardware_efficient':
        return 2 * n * n_layers
    elif ansatz == 'strongly_entangling':
        return 3 * n * n_layers
    elif ansatz == 'circular':
        return 2 * n * n_layers
    raise ValueError(f"Unknown ansatz '{ansatz}'. Choose from {ANSATZE}.")


# ---------------------------------------------------------------------------
# Ansatz builders  (each returns a QuantumCircuit, not yet an instruction)
# ---------------------------------------------------------------------------

def _ansatz_mcmt(n, x):
    """Original MCMT ansatz from the paper — 4*n params."""
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name='G')

    for i in range(n):
        if i < n - 1:
            gate = MCMT(RZGate(2 * pi * x[i]), n - i - 1, 1)
            qc.append(gate, list(range(n - i)))
        else:
            qc.rz(2 * pi * x[i], 0)

    for i in range(n):
        if i < n - 1:
            gate = MCMT(RYGate(2 * pi * x[i + n]), n - i - 1, 1)
            qc.append(gate, list(range(n - i)))
        else:
            qc.ry(2 * pi * x[i + n], 0)

    for i in reversed(range(n)):
        if i < n - 1:
            gate = MCMT(RYGate(2 * pi * x[i + 3 * n]), n - i - 1, 1)
            qc.append(gate, list(range(n - i)))
        else:
            qc.ry(2 * pi * x[i + 3 * n], 0)

    for i in reversed(range(n)):
        if i < n - 1:
            gate = MCMT(RZGate(2 * pi * x[i + 2 * n]), n - i - 1, 1)
            qc.append(gate, list(range(n - i)))
        else:
            qc.rz(2 * pi * x[i + 2 * n], 0)

    return qc


def _ansatz_hardware_efficient(n, x, n_layers):
    """
    Hardware-efficient ansatz — 2*n*n_layers params.

    Each layer: RY(x) + RZ(x) on every qubit, then a linear CNOT ladder.
    Friendly to nearest-neighbour topologies.
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name='G')
    p = 0
    for _ in range(n_layers):
        for q in range(n):
            qc.ry(2 * pi * x[p], q);     p += 1
            qc.rz(2 * pi * x[p], q);     p += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def _ansatz_strongly_entangling(n, x, n_layers):
    """
    Strongly entangling layers — 3*n*n_layers params.

    Each layer: RX + RY + RZ on every qubit, then CNOT ring with
    offset = layer mod (n-1) + 1  (varies entanglement range per layer).
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name='G')
    p = 0
    for layer in range(n_layers):
        for q in range(n):
            qc.rx(2 * pi * x[p], q);     p += 1
            qc.ry(2 * pi * x[p], q);     p += 1
            qc.rz(2 * pi * x[p], q);     p += 1
        if n > 1:
            offset = layer % (n - 1) + 1
            for q in range(n):
                qc.cx(q, (q + offset) % n)
    return qc


def _ansatz_circular(n, x, n_layers):
    """
    Circular entanglement ansatz — 2*n*n_layers params.

    Each layer: RY + RZ on every qubit, then circular CNOT ring
    (last qubit connects back to first).
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name='G')
    p = 0
    for _ in range(n_layers):
        for q in range(n):
            qc.ry(2 * pi * x[p], q);     p += 1
            qc.rz(2 * pi * x[p], q);     p += 1
        if n > 1:
            for q in range(n):
                qc.cx(q, (q + 1) % n)
    return qc


# ---------------------------------------------------------------------------
# Generator / discriminator
# ---------------------------------------------------------------------------

def generator_g(n, x_g, ansatz='mcmt', n_layers=1):
    """Trained (variational) generator circuit on *n* qubits."""
    n_params = generator_num_params(n, ansatz, n_layers)
    x = ParameterVector('x', n_params)

    if ansatz == 'mcmt':
        qc = _ansatz_mcmt(n, x)
    elif ansatz == 'hardware_efficient':
        qc = _ansatz_hardware_efficient(n, x, n_layers)
    elif ansatz == 'strongly_entangling':
        qc = _ansatz_strongly_entangling(n, x, n_layers)
    elif ansatz == 'circular':
        qc = _ansatz_circular(n, x, n_layers)
    else:
        raise ValueError(f"Unknown ansatz '{ansatz}'. Choose from {ANSATZE}.")

    return qc.to_instruction({x: x_g})


def ghz_statevector(n):
    """Return the *n*-qubit GHZ state as a list of amplitudes."""
    return [1 / sqrt(2)] + (2**n - 2) * [0] + [1 / sqrt(2)]


def generator_r(n):
    """Real-data generator: prepares the *n*-qubit GHZ state."""
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name='R')
    qc.initialize(ghz_statevector(n), list(range(n)))
    return qc.to_instruction()


def generator_r_dagger(n):
    """Conjugate transpose of the real-data generator."""
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name='Rdg')
    qi = Initialize(ghz_statevector(n)).gates_to_uncompute().to_instruction()
    qc.append(qi, qr)
    return qc.to_instruction()


def discriminator(n, x_d, testing=False, ansatz='mcmt', n_layers=1):
    """Trained discriminator circuit on *n* + 1 qubits."""
    n_g = generator_num_params(n, ansatz, n_layers)
    qr = QuantumRegister(n + 1)
    qc = QuantumCircuit(qr, name='D')

    x_d1 = ParameterVector('xD1', n_g)
    sub_inst = generator_g(n, x_d1, ansatz=ansatz, n_layers=n_layers)
    qc.append(sub_inst, list(range(n)))

    shift = pi if testing else pi / 2
    gate = MCMT(RYGate(shift), n, 1)
    qc.append(gate, list(range(n + 1)))

    qc.append(sub_inst.inverse(), list(range(n)))
    return qc.to_instruction({x_d1: x_d[0:n_g]})


# ---------------------------------------------------------------------------
# Statevector helpers  (simulator path — no shots)
# ---------------------------------------------------------------------------

def _run_sv(circuit, backend=None):
    """Compute the statevector of a parameterised circuit (all params bound)."""
    return np.asarray(Statevector(circuit))


def real_true_sv(n, x, circuits, backend, prt=None):
    """P(real recognised as real) on statevector simulator."""
    if prt is None:
        prt = []
    n_g = circuits['n_g']
    qr_circ, x_d = circuits['qR'], circuits['xD']
    b = {x_d: x[:n_g]}
    qr_bound = qr_circ.assign_parameters(b)
    sv = _run_sv(qr_bound, backend)
    p = np.abs(sv[0]) ** 2
    prt.append(p)
    return 1 - p


def fake_true_sv(n, x, circuits, backend, pft=None):
    """P(fake recognised as true) on statevector simulator."""
    if pft is None:
        pft = []
    n_g = circuits['n_g']
    qg_circ, x_d, x_g = circuits['qG'], circuits['xD'], circuits['xG']
    b = {x_d: x[:n_g], x_g: x[n_g:2 * n_g]}
    qg_bound = qg_circ.assign_parameters(b)
    sv = _run_sv(qg_bound, backend)
    p = np.abs(sv[0]) ** 2
    pft.append(p)
    return 1 - p


def fidelity_rg_sv(n, x, circuits, backend, fid=None):
    """Fidelity between real and generated states (statevector)."""
    if fid is None:
        fid = []
    n_g = circuits['n_g']
    qrg_circ, x_g = circuits['qRG'], circuits['xG']
    b = {x_g: x[n_g:2 * n_g]}
    qrg_bound = qrg_circ.assign_parameters(b)
    sv = _run_sv(qrg_bound, backend)
    result = np.abs(sv[0]) ** 2
    fid.append(result)
    return result


def cost_new_sv(n, x, circuits, backend, counter, verbose=False,
                prt=None, pft=None, fid=None):
    """Total SQGEN cost function (statevector simulator)."""
    n_g = circuits['n_g']
    q_circ, x_d, x_g = circuits['q'], circuits['xD'], circuits['xG']
    counter[0] += 1
    b = {x_d: x[:n_g], x_g: x[n_g:2 * n_g]}
    qb = q_circ.assign_parameters(b)
    sv = _run_sv(qb, backend)
    n_reg = 1e-8 * np.linalg.norm(x)
    result = 1 - np.abs(sv[0]) ** 2 + n_reg
    if verbose:
        print(np.array([
            result,
            real_true_sv(n, x, circuits, backend, prt),
            fake_true_sv(n, x, circuits, backend, pft),
            fidelity_rg_sv(n, x, circuits, backend, fid),
        ]))
    return result


def disc_cost_sv(n, x, circuits, backend, counter, prt=None, pft=None):
    """Discriminator cost function (QGAN, statevector)."""
    n_g = circuits['n_g']
    counter[0] += 1
    q = fake_true_sv(n, x, circuits, backend, pft)
    p = real_true_sv(n, x, circuits, backend, prt)
    d = np.abs(p - q)
    dp = np.abs(p)
    n_reg = 1e-8 * np.linalg.norm(x[:n_g])
    return np.abs(1 - d) - dp + n_reg


def gen_cost_sv(n, x, circuits, backend, counter, fid=None):
    """Generator cost function (QGAN, statevector)."""
    n_g = circuits['n_g']
    counter[0] += 1
    n_reg = 1e-8 * np.linalg.norm(x[n_g:2 * n_g])
    d = 1 - fidelity_rg_sv(n, x, circuits, backend, fid) + n_reg
    return np.log(d)


# ---------------------------------------------------------------------------
# Shot-based helpers  (real / fake hardware path)
# ---------------------------------------------------------------------------

def _count_key(counts, key, default=0):
    return float(counts.get(key, default))


def _run_shots(circuit, backend, shots, coupling_map=None, basis_gates=None):
    """Transpile, run with shots, return counts dict."""
    tc = transpile(circuit, backend)
    job = backend.run(tc, shots=shots)
    return job.result().get_counts()


def real_true_shots(n, x, circuits, backend, shots, prt=None,
                    coupling_map=None, basis_gates=None):
    """P(real recognised as real) — shot-based."""
    if prt is None:
        prt = []
    n_g = circuits['n_g']
    qr_circ, x_d = circuits['qR'], circuits['xD']
    b = {x_d: x[:n_g]}
    qr_bound = qr_circ.assign_parameters(b)
    counts = _run_shots(qr_bound, backend, shots, coupling_map, basis_gates)
    c00 = _count_key(counts, '00')
    result = c00 / float(shots)
    prt.append(result)
    return 1 - result


def fake_true_shots(n, x, circuits, backend, shots, pft=None,
                    coupling_map=None, basis_gates=None):
    """P(fake recognised as true) — shot-based."""
    if pft is None:
        pft = []
    n_g = circuits['n_g']
    qg_circ, x_d, x_g = circuits['qG'], circuits['xD'], circuits['xG']
    b = {x_d: x[:n_g], x_g: x[n_g:2 * n_g]}
    qg_bound = qg_circ.assign_parameters(b)
    counts = _run_shots(qg_bound, backend, shots, coupling_map, basis_gates)
    c00 = _count_key(counts, '00')
    c10 = _count_key(counts, '10')
    c01 = _count_key(counts, '01')
    c11 = _count_key(counts, '11')
    total = float(c00 + c01 + c10 + c11)
    result = c00 / total
    pft.append(result)
    return 1 - result


def fidelity_rg_shots(n, x, circuits, backend, shots, fid=None,
                      coupling_map=None, basis_gates=None):
    """Fidelity — shot-based."""
    if fid is None:
        fid = []
    n_g = circuits['n_g']
    qrg_circ, x_g = circuits['qRG'], circuits['xG']
    b = {x_g: x[n_g:2 * n_g]}
    qrg_bound = qrg_circ.assign_parameters(b)
    counts = _run_shots(qrg_bound, backend, shots, coupling_map, basis_gates)
    c0 = _count_key(counts, '0')
    result = c0 / float(shots)
    fid.append(result)
    return result


def cost_new_shots(n, x, circuits, backend, shots, counter, verbose=False,
                   prt=None, pft=None, fid=None,
                   coupling_map=None, basis_gates=None):
    """Total SQGEN cost — shot-based."""
    n_g = circuits['n_g']
    q_circ, x_d, x_g = circuits['q'], circuits['xD'], circuits['xG']
    counter[0] += 1
    b = {x_d: x[:n_g], x_g: x[n_g:2 * n_g]}
    qb = q_circ.assign_parameters(b)
    counts = _run_shots(qb, backend, shots, coupling_map, basis_gates)
    c00 = _count_key(counts, '00')
    result = 1 - c00 / float(shots)
    if verbose:
        print(np.array([
            result,
            real_true_shots(n, x, circuits, backend, shots, prt),
            fake_true_shots(n, x, circuits, backend, shots, pft),
            fidelity_rg_shots(n, x, circuits, backend, shots, fid),
        ]), end="\r")
    return result


# ---------------------------------------------------------------------------
# Circuit assembly
# ---------------------------------------------------------------------------

def build_circuits(n, backend, x_d, x_g, x_r=None, measure=False,
                   ansatz='mcmt', n_layers=1):
    """
    Build and transpile all training circuits.

    Parameters
    ----------
    ansatz : str
        Variational ansatz for the generator (and discriminator sub-circuit).
        One of: 'mcmt', 'hardware_efficient', 'strongly_entangling', 'circular'.
    n_layers : int
        Number of variational layers (ignored for 'mcmt').

    Returns a dict with keys: 'q', 'qR', 'qG', 'qRG', 'xD', 'xG', 'n_g'
    and transpiled circuit depths: 'dq', 'dqR', 'dqG', 'dqRG'.
    """
    n_g = generator_num_params(n, ansatz, n_layers)
    D = discriminator(n, x_d, ansatz=ansatz, n_layers=n_layers)
    Dt = discriminator(n, x_d, testing=True, ansatz=ansatz, n_layers=n_layers)
    G = generator_g(n, x_g, ansatz=ansatz, n_layers=n_layers)
    R = generator_r(n)
    Rdg = generator_r_dagger(n)

    # Main SQGEN circuit
    qr = QuantumRegister(n + 1, 'q')
    qc = QuantumCircuit(qr)
    qc.append(R, list(range(n)))
    qc.append(D, list(range(n + 1)))
    qc.x(qr[n])
    qc.append(D.inverse(), list(range(n + 1)))
    qc.append(G.inverse(), list(range(n)))
    if measure:
        qc.measure_all()
    q = transpile(qc, backend)

    # Real-true circuit
    qr2 = QuantumRegister(n + 1, 'q')
    qc2 = QuantumCircuit(qr2)
    qc2.append(R, list(range(n)))
    qc2.append(Dt, list(range(n + 1)))
    qc2.append(Rdg, list(range(n)))
    if measure:
        qc2.measure_all()
    qR = transpile(qc2, backend)

    # Fake-true circuit
    qr3 = QuantumRegister(n + 1, 'q')
    qc3 = QuantumCircuit(qr3)
    qc3.append(G, list(range(n)))
    qc3.append(Dt, list(range(n + 1)))
    qc3.append(G.inverse(), list(range(n)))
    if measure:
        qc3.measure_all()
    qG = transpile(qc3, backend)

    # Fidelity circuit
    qr4 = QuantumRegister(n, 'q')
    qc4 = QuantumCircuit(qr4)
    qc4.append(R, list(range(n)))
    qc4.append(G.inverse(), list(range(n)))
    if measure:
        qc4.measure_all()
    qRG = transpile(qc4, backend)

    return {
        'q': q, 'qR': qR, 'qG': qG, 'qRG': qRG,
        'xD': x_d, 'xG': x_g, 'n_g': n_g,
        'dq': q.depth(), 'dqR': qR.depth(),
        'dqG': qG.depth(), 'dqRG': qRG.depth(),
    }

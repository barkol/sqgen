"""Continue benchmark — circular QGAN + all SQGEN."""
import csv
import sys
import time

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from sqgen import (
    ANSATZE, generator_num_params, build_circuits,
    fidelity_rg_sv,
    cost_new_sv, disc_cost_sv, gen_cost_sv,
)

backend = AerSimulator(method='statevector')

MAX_EVALS = 1500
SEEDS = [42, 103, 7, 256, 999]
N_QUBITS = [1, 2, 3, 4, 5]
N_LAYERS = 3
CSV_FILE = "benchmark_results.csv"


def run_qgan(n, seed, ansatz, n_layers):
    n_g = generator_num_params(n, ansatz, n_layers)
    xG = ParameterVector('xG', n_g)
    xD = ParameterVector('xD', n_g)
    circ = build_circuits(n, backend, xD, xG, ansatz=ansatz, n_layers=n_layers)
    np.random.seed(seed)
    x0 = np.random.rand(2 * n_g)
    xD_val, xG_val = x0[:n_g].copy(), x0[n_g:].copy()
    dc, gc = [0], [0]
    pa, pf, fa = [], [], []
    epoch = 0
    while dc[0] + gc[0] < MAX_EVALS:
        solD = minimize(
            lambda xd: disc_cost_sv(
                n, xd.tolist() + xG_val.tolist(), circ, backend, dc, pa, pf),
            xD_val, method='BFGS', options={'maxiter': 1, 'disp': False})
        xD_val = solD.x
        if dc[0] + gc[0] >= MAX_EVALS:
            break
        solG = minimize(
            lambda xg: gen_cost_sv(
                n, xD_val.tolist() + xg.tolist(), circ, backend, gc, fa),
            xG_val, method='BFGS', options={'maxiter': 1, 'disp': False})
        xG_val = solG.x
        epoch += 1
    f = fidelity_rg_sv(n, xD_val.tolist() + xG_val.tolist(), circ, backend)
    return f, dc[0] + gc[0], epoch


def run_sqgen(n, seed, ansatz, n_layers):
    n_g = generator_num_params(n, ansatz, n_layers)
    xG = ParameterVector('xG', n_g)
    xD = ParameterVector('xD', n_g)
    circ = build_circuits(n, backend, xD, xG, ansatz=ansatz, n_layers=n_layers)
    np.random.seed(seed)
    x0 = np.random.rand(2 * n_g)
    counter = [0]
    prt, pft, fid = [], [], []
    epoch = 0
    while counter[0] < MAX_EVALS:
        cost_new_sv(n, x0, circ, backend, counter)
        if counter[0] >= MAX_EVALS:
            break
        sol = minimize(
            lambda x: cost_new_sv(n, x, circ, backend, counter, False,
                                  prt, pft, fid),
            x0, method='BFGS', options={'maxiter': 1, 'disp': False})
        x0 = sol.x
        epoch += 1
    fi = []
    fidelity_rg_sv(n, x0, circ, backend, fi)
    return fi[0], counter[0], epoch


# Jobs to run: circular QGAN + all SQGEN
jobs = []
# circular QGAN
for nq in N_QUBITS:
    for seed in SEEDS:
        jobs.append(('QGAN', 'circular', 3, nq, seed, run_qgan))
# all SQGEN
for ansatz in ANSATZE:
    nl = 1 if ansatz == 'mcmt' else N_LAYERS
    for nq in N_QUBITS:
        for seed in SEEDS:
            jobs.append(('SQGEN', ansatz, nl, nq, seed, run_sqgen))

total = len(jobs)
for i, (method, ansatz, nl, nq, seed, runner) in enumerate(jobs, 1):
    n_g = generator_num_params(nq, ansatz, nl)
    t1 = time.time()
    f, evals, epochs = runner(nq, seed, ansatz, nl)
    dt = time.time() - t1
    with open(CSV_FILE, 'a', newline='') as fh:
        csv.writer(fh).writerow([
            method, ansatz, nl, nq, seed,
            2 * n_g, evals, epochs, f'{f:.6f}', f'{dt:.1f}'])
    print(f"[{i:3d}/{total}] {method:5s} {ansatz:25s} n={nq} seed={seed:3d} "
          f"F={f:.4f} ({dt:.1f}s)")
    sys.stdout.flush()

print("\nPart 2 done.")

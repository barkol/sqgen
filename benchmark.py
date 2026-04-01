"""
Benchmark: fidelity at fixed evaluation budget across seeds and qubit counts.
Results saved incrementally to benchmark_results.csv.
"""
import csv
import sys
import time
import os

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


# ── Main ──

# Remove old results
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

with open(CSV_FILE, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['method', 'ansatz', 'layers', 'n', 'seed', 'params',
                'evals', 'epochs', 'fidelity', 'time_s'])

t0 = time.time()
total_runs = len(ANSATZE) * len(N_QUBITS) * len(SEEDS) * 2
done = 0

for method_name, runner in [("QGAN", run_qgan), ("SQGEN", run_sqgen)]:
    for ansatz in ANSATZE:
        n_layers_eff = 1 if ansatz == 'mcmt' else N_LAYERS
        for nq in N_QUBITS:
            n_g = generator_num_params(nq, ansatz, n_layers_eff)
            for seed in SEEDS:
                t1 = time.time()
                f, evals, epochs = runner(nq, seed, ansatz, n_layers_eff)
                dt = time.time() - t1
                done += 1
                with open(CSV_FILE, 'a', newline='') as fh:
                    csv.writer(fh).writerow([
                        method_name, ansatz, n_layers_eff, nq, seed,
                        2 * n_g, evals, epochs, f'{f:.6f}', f'{dt:.1f}'])
                print(f"[{done:3d}/{total_runs}] {method_name:5s} "
                      f"{ansatz:25s} n={nq} seed={seed:3d} "
                      f"F={f:.4f} ({dt:.1f}s)")
                sys.stdout.flush()

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s. Results in {CSV_FILE}")

# ── Summary table ──
import pandas as pd
df = pd.read_csv(CSV_FILE)
print(f"\n{'='*80}")
print(f" Summary: mean fidelity ± std  (budget={MAX_EVALS} evals)")
print(f"{'='*80}")
for method in ['QGAN', 'SQGEN']:
    print(f"\n {method}:")
    sub = df[df['method'] == method]
    tbl = sub.groupby(['ansatz', 'layers'])['fidelity'].agg(['mean', 'std'])
    # pivot by n
    rows = []
    for ansatz in ANSATZE:
        nl = 1 if ansatz == 'mcmt' else N_LAYERS
        vals = []
        for nq in N_QUBITS:
            s = sub[(sub['ansatz'] == ansatz) & (sub['n'] == nq)]['fidelity']
            vals.append(f"{s.mean():.3f}±{s.std():.3f}")
        p = sub[(sub['ansatz'] == ansatz) & (sub['n'] == N_QUBITS[-1])]['params'].iloc[0]
        label = f"{ansatz} (L={nl})"
        print(f"  {label:25s} {p:3d}p  {'  '.join(vals)}")
    print(f"  {'':25s} {'':>4s}  " + "  ".join(f"  n={nq}  " for nq in N_QUBITS))

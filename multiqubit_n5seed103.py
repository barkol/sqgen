"""
Train SQGEN and QGAN circuits on a statevector simulator.

Paper: "Synergic quantum generative machine learning" (arXiv:2112.13255v2)
"""

import time

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from sqgen import (
    build_circuits, generator_num_params,
    real_true_sv, fake_true_sv, fidelity_rg_sv,
    cost_new_sv, disc_cost_sv, gen_cost_sv,
)

# ── Configuration ──────────────────────────────────────────────────────────

SEED = 103
N_QUBITS = 5
ITR_NEW = 20
ITR_OLD = 20

# Ansatz: 'mcmt' (original), 'hardware_efficient', 'strongly_entangling', 'circular'
ANSATZ = 'mcmt'
N_LAYERS = 2  # number of variational layers (ignored for 'mcmt')

ALG = 'BFGS'
OPT = {'maxiter': 1, 'disp': False, 'eps': 1e-6, 'finite_diff_rel_step': 1e-8}
OPT_D = {'maxiter': 1, 'disp': False, 'eps': 1e-6, 'finite_diff_rel_step': 1e-8}
OPT_G = {'maxiter': 1, 'disp': False, 'eps': 1e-6, 'finite_diff_rel_step': 1e-8}

backend = AerSimulator(method='statevector')


# ── Helpers ────────────────────────────────────────────────────────────────

def save_time(n, t, dq, dqR, dqG, dqRG, a, b, c, seed):
    with open(f"log_times_seed{seed}_paperALT.txt", "a") as f:
        f.write(f"{n}\t{t}\t{dq}\t{dqR}\t{dqG}\t{dqRG}\t{a}\t{b}\t{c}\n")


def learn_old(n, itr, seed, circuits, verbose=True):
    """QGAN alternating training loop."""
    n_g = circuits['n_g']
    np.random.seed(seed)
    x0 = np.random.rand(2 * n_g)
    xD_val = x0[:n_g]
    xG_val = x0[n_g:2 * n_g]

    prt_iter, pft_iter, fid_iter = [], [], []
    disc_counter, gen_counter = [0], [0]

    tprr = real_true_sv(n, x0.tolist(), circuits, backend, prt_iter)
    tpfr = fake_true_sv(n, x0.tolist(), circuits, backend, pft_iter)
    tfid = fidelity_rg_sv(n, x0.tolist(), circuits, backend, fid_iter)
    if verbose:
        print("start:\t", np.array([tprr, tpfr, tfid]))

    prt_all, pft_all, fid_all = [], [], []

    for l in range(itr):
        print(f"QGAN iteration:\t{l}")
        solD = minimize(
            lambda xd: disc_cost_sv(
                n, xd.tolist() + xG_val.tolist(), circuits, backend,
                disc_counter, prt_all, pft_all),
            xD_val, method=ALG, options=OPT_D)
        xD_val = solD.x
        tprr = real_true_sv(n, xD_val.tolist() + xG_val.tolist(),
                            circuits, backend, prt_iter)
        tpfr = fake_true_sv(n, xD_val.tolist() + xG_val.tolist(),
                            circuits, backend, pft_iter)
        if verbose:
            print("D:", np.array([l, 0, tprr, tpfr, tfid]))

        solG = minimize(
            lambda xg: gen_cost_sv(
                n, xD_val.tolist() + xg.tolist(), circuits, backend,
                gen_counter, fid_all),
            xG_val, method=ALG, options=OPT_G)
        xG_val = solG.x
        tfid = fidelity_rg_sv(n, xD_val.tolist() + xG_val.tolist(),
                              circuits, backend, fid_iter)
        if verbose:
            print("G:", np.array([l, 0, tprr, tpfr, tfid]))

    print("============= COST FUNCTION EVALUATIONS ============")
    print(np.array([gen_counter[0], disc_counter[0]]))
    return prt_all, pft_all, fid_all, prt_iter, pft_iter, fid_iter


# ── Main ───────────────────────────────────────────────────────────────────

for seed in [SEED]:
    for n in [N_QUBITS]:
        print(f"GHZ:\t{n}")

        n_g = generator_num_params(n, ANSATZ, N_LAYERS)
        xG = ParameterVector('xG', n_g)
        xD = ParameterVector('xD', n_g)
        circuits = build_circuits(n, backend, xD, xG,
                                  ansatz=ANSATZ, n_layers=N_LAYERS)
        dq, dqR = circuits['dq'], circuits['dqR']
        dqG, dqRG = circuits['dqG'], circuits['dqRG']
        print(f"  ansatz={ANSATZ}, n_layers={N_LAYERS}, "
              f"params/circuit={n_g}, total={2*n_g}")
        print(f"  depths: q={dq}, qR={dqR}, qG={dqG}, qRG={dqRG}")

        # ── SQGEN ──────────────────────────────────────────────────────
        start_time = time.time()
        np.random.seed(seed)
        x0 = np.random.rand(2 * n_g)

        prt_new, pft_new, fid_new = [], [], []
        prt_iter_new, pft_iter_new, fid_iter_new = [], [], []
        cost_new_list = []
        counter_new = [0]

        for m in range(ITR_NEW):
            print(f"SQGEN iteration:\t{m}")
            fake_true_sv(n, x0, circuits, backend, pft_iter_new)
            real_true_sv(n, x0, circuits, backend, prt_iter_new)
            fidelity_rg_sv(n, x0, circuits, backend, fid_iter_new)
            cost_new_list.append(
                cost_new_sv(n, x0, circuits, backend, counter_new))
            sol = minimize(
                lambda x: cost_new_sv(
                    n, x, circuits, backend, counter_new, False,
                    prt_new, pft_new, fid_new),
                x0, method=ALG, options=OPT)
            x0 = sol.x

        fake_true_sv(n, x0, circuits, backend, pft_iter_new)
        real_true_sv(n, x0, circuits, backend, prt_iter_new)
        fidelity_rg_sv(n, x0, circuits, backend, fid_iter_new)

        t_new = time.time() - start_time
        print(f"NEW --- {t_new:.1f} seconds ---")
        save_time(n, t_new, dq, dqR, dqG, dqRG, 0, 0, counter_new[0], seed)
        print(counter_new[0])

        sfx = f"seed{seed}_n_{n}disc1ALT"
        np.save(f"prt_new_{sfx}.npy", prt_new)
        np.save(f"pft_new_{sfx}.npy", pft_new)
        np.save(f"fid_new_{sfx}.npy", fid_new)
        np.save(f"prt_iter_new_{sfx}.npy", prt_iter_new)
        np.save(f"pft_iter_new_{sfx}.npy", pft_iter_new)
        np.save(f"fid_iter_new_{sfx}.npy", fid_iter_new)

        # ── QGAN ───────────────────────────────────────────────────────
        start_time = time.time()
        (prt_old, pft_old, fid_old,
         prt_iter_old, pft_iter_old, fid_iter_old) = learn_old(
            n, ITR_OLD, seed, circuits, verbose=False)

        t_old = time.time() - start_time
        print(f"OLD --- {t_old:.1f} seconds ---")
        save_time(n, t_old, dq, dqR, dqG, dqRG,
                  len(prt_old), len(pft_old), len(fid_old), seed)

        np.save(f"prt_old_{sfx}.npy", prt_old)
        np.save(f"pft_old_{sfx}.npy", pft_old)
        np.save(f"fid_old_{sfx}.npy", fid_old)
        np.save(f"prt_iter_old_{sfx}.npy", prt_iter_old)
        np.save(f"pft_iter_old_{sfx}.npy", pft_iter_old)
        np.save(f"fid_iter_old_{sfx}.npy", fid_iter_old)

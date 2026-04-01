"""Smoke tests for sqgen module — ansatz construction and circuit building."""

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from sqgen import (
    ANSATZE, generator_num_params, generator_g, discriminator,
    build_circuits, cost_new_sv, real_true_sv, fake_true_sv, fidelity_rg_sv,
)

backend = AerSimulator(method='statevector')


def test_generator_num_params():
    """Parameter counts match expected formulas."""
    n = 4
    assert generator_num_params(n, 'mcmt') == 4 * n
    assert generator_num_params(n, 'hardware_efficient', 3) == 2 * n * 3
    assert generator_num_params(n, 'strongly_entangling', 2) == 3 * n * 2
    assert generator_num_params(n, 'circular', 2) == 2 * n * 2
    print("PASS: test_generator_num_params")


def test_all_ansatze_build():
    """Every ansatz builds a generator instruction without error."""
    n = 3
    for ansatz in ANSATZE:
        n_layers = 1 if ansatz == 'mcmt' else 2
        n_g = generator_num_params(n, ansatz, n_layers)
        x_g = ParameterVector('xG', n_g)
        inst = generator_g(n, x_g, ansatz=ansatz, n_layers=n_layers)
        assert inst is not None, f"generator_g returned None for {ansatz}"
    print("PASS: test_all_ansatze_build")


def test_discriminator_builds():
    """Discriminator builds with each ansatz."""
    n = 2
    for ansatz in ANSATZE:
        n_layers = 1 if ansatz == 'mcmt' else 2
        n_g = generator_num_params(n, ansatz, n_layers)
        x_d = ParameterVector('xD', n_g)
        inst = discriminator(n, x_d, ansatz=ansatz, n_layers=n_layers)
        assert inst is not None, f"discriminator returned None for {ansatz}"
    print("PASS: test_discriminator_builds")


def test_build_circuits_all_ansatze():
    """build_circuits succeeds and returns correct n_g for all ansatze."""
    n = 2
    for ansatz in ANSATZE:
        n_layers = 1 if ansatz == 'mcmt' else 2
        n_g = generator_num_params(n, ansatz, n_layers)
        x_g = ParameterVector('xG', n_g)
        x_d = ParameterVector('xD', n_g)
        circ = build_circuits(n, backend, x_d, x_g,
                              ansatz=ansatz, n_layers=n_layers)
        assert circ['n_g'] == n_g, f"n_g mismatch for {ansatz}"
        assert circ['dq'] > 0, f"zero depth for {ansatz}"
    print("PASS: test_build_circuits_all_ansatze")


def test_cost_runs():
    """Cost function produces a finite scalar for each ansatz."""
    n = 2
    for ansatz in ANSATZE:
        n_layers = 1 if ansatz == 'mcmt' else 2
        n_g = generator_num_params(n, ansatz, n_layers)
        x_g = ParameterVector('xG', n_g)
        x_d = ParameterVector('xD', n_g)
        circ = build_circuits(n, backend, x_d, x_g,
                              ansatz=ansatz, n_layers=n_layers)
        np.random.seed(42)
        x0 = np.random.rand(2 * n_g)
        counter = [0]
        c = cost_new_sv(n, x0, circ, backend, counter)
        assert np.isfinite(c), f"non-finite cost for {ansatz}: {c}"
        assert counter[0] == 1
    print("PASS: test_cost_runs")


def test_fidelity_and_probs():
    """Probability and fidelity helpers return values in [0, 1]."""
    n = 2
    ansatz = 'hardware_efficient'
    n_layers = 2
    n_g = generator_num_params(n, ansatz, n_layers)
    x_g = ParameterVector('xG', n_g)
    x_d = ParameterVector('xD', n_g)
    circ = build_circuits(n, backend, x_d, x_g,
                          ansatz=ansatz, n_layers=n_layers)
    np.random.seed(0)
    x0 = np.random.rand(2 * n_g)

    prt, pft, fid = [], [], []
    rt = real_true_sv(n, x0, circ, backend, prt)
    ft = fake_true_sv(n, x0, circ, backend, pft)
    f = fidelity_rg_sv(n, x0, circ, backend, fid)
    for val, name in [(rt, 'real_true'), (ft, 'fake_true'), (f, 'fidelity')]:
        assert 0 <= val <= 1, f"{name} out of range: {val}"
    assert len(prt) == 1 and len(pft) == 1 and len(fid) == 1
    print("PASS: test_fidelity_and_probs")


def test_qgan_loop():
    """QGAN with fixed evaluation budget for fair comparison."""
    from sqgen import disc_cost_sv, gen_cost_sv
    from scipy.optimize import minimize

    MAX_EVALS = 1500
    n = 3
    print(f"  QGAN (budget={MAX_EVALS} evals, n={n}):")
    for ansatz in ANSATZE:
        n_layers = 1 if ansatz == 'mcmt' else 3
        n_g = generator_num_params(n, ansatz, n_layers)
        x_g = ParameterVector('xG', n_g)
        x_d = ParameterVector('xD', n_g)
        circ = build_circuits(n, backend, x_d, x_g,
                              ansatz=ansatz, n_layers=n_layers)

        np.random.seed(42)
        x0 = np.random.rand(2 * n_g)
        xD_val = x0[:n_g]
        xG_val = x0[n_g:2 * n_g]

        disc_counter, gen_counter = [0], [0]
        prt_all, pft_all, fid_all = [], [], []

        epoch = 0
        while disc_counter[0] + gen_counter[0] < MAX_EVALS:
            solD = minimize(
                lambda xd: disc_cost_sv(
                    n, xd.tolist() + xG_val.tolist(), circ, backend,
                    disc_counter, prt_all, pft_all),
                xD_val, method='BFGS',
                options={'maxiter': 1, 'disp': False})
            xD_val = solD.x

            if disc_counter[0] + gen_counter[0] >= MAX_EVALS:
                break

            solG = minimize(
                lambda xg: gen_cost_sv(
                    n, xD_val.tolist() + xg.tolist(), circ, backend,
                    gen_counter, fid_all),
                xG_val, method='BFGS',
                options={'maxiter': 1, 'disp': False})
            xG_val = solG.x
            epoch += 1

        total = disc_counter[0] + gen_counter[0]
        f = fidelity_rg_sv(n, xD_val.tolist() + xG_val.tolist(),
                           circ, backend)
        assert 0 <= f <= 1
        print(f"    {ansatz:25s}  params={2*n_g:3d}  "
              f"evals={total:5d}  epochs={epoch:3d}  F={f:.4f}")

    print("  PASS: test_qgan_loop")


def test_sqgen_loop():
    """SQGEN with fixed evaluation budget for fair comparison."""
    from scipy.optimize import minimize

    MAX_EVALS = 1500
    n = 3
    print(f"  SQGEN (budget={MAX_EVALS} evals, n={n}):")
    for ansatz in ANSATZE:
        n_layers = 1 if ansatz == 'mcmt' else 3
        n_g = generator_num_params(n, ansatz, n_layers)
        x_g = ParameterVector('xG', n_g)
        x_d = ParameterVector('xD', n_g)
        circ = build_circuits(n, backend, x_d, x_g,
                              ansatz=ansatz, n_layers=n_layers)

        np.random.seed(42)
        x0 = np.random.rand(2 * n_g)

        counter = [0]
        prt, pft, fid = [], [], []
        fid_iter = []

        epoch = 0
        while counter[0] < MAX_EVALS:
            fidelity_rg_sv(n, x0, circ, backend, fid_iter)
            cost_new_sv(n, x0, circ, backend, counter)
            if counter[0] >= MAX_EVALS:
                break
            sol = minimize(
                lambda x: cost_new_sv(
                    n, x, circ, backend, counter, False, prt, pft, fid),
                x0, method='BFGS',
                options={'maxiter': 1, 'disp': False})
            x0 = sol.x
            epoch += 1

        fidelity_rg_sv(n, x0, circ, backend, fid_iter)
        f = fid_iter[-1]
        assert 0 <= f <= 1
        print(f"    {ansatz:25s}  params={2*n_g:3d}  "
              f"evals={counter[0]:5d}  epochs={epoch:3d}  F={f:.4f}")

    print("  PASS: test_sqgen_loop")


if __name__ == '__main__':
    test_generator_num_params()
    test_all_ansatze_build()
    test_discriminator_builds()
    test_build_circuits_all_ansatze()
    test_cost_runs()
    test_fidelity_and_probs()
    test_qgan_loop()
    test_sqgen_loop()
    print("\nAll tests passed.")

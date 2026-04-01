"""
Plot learning curves from saved .npy data.

Paper: "Synergic quantum generative machine learning" (arXiv:2112.13255v2)
"""

import numpy as np
import matplotlib.pyplot as plt

SEED = 103
N_QUBITS = 5

plt.rcParams['text.usetex'] = True
serif_font = {'fontname': 'serif'}
cm = 1 / 2.54

for n in [N_QUBITS]:
    base_new = f"_iter_new_seed{SEED}_n_{n}disc1ALT.npy"
    base_old = f"_iter_old_seed{SEED}_n_{n}disc1ALT.npy"

    cases = {"SQGEN": base_new, "QGAN": base_old}

    for alg, base in cases.items():
        y1 = np.load("prt" + base)
        y2 = np.load("pft" + base)
        y3 = np.load("fid" + base)

        if alg == "QGAN":
            y1 = y1[:len(y3)]
            y2 = y2[:len(y3)]

        fig, ax = plt.subplots(figsize=(8.5 * cm, 6 * cm))

        ax.plot(y1, 'g', label=r"$1-p$", marker=">")
        ax.plot(y2, 'r', label=r"$1-q$", marker="<")
        ax.plot(y3, 'b', label=r"$F$", marker="^")

        tit = (r"$\mathrm{" + alg + r"}:\quad n=" + str(n)
               + r",\quad \mathrm{seed}=" + str(SEED) + r"$")
        ax.set_title(tit)
        ax.legend(loc='center right')

        ax.set_ylabel(r'$\mathrm{Learning\; parameters}$', fontsize=10, **serif_font)
        ax.set_xlabel(r'$\mathrm{Epoch}$', fontsize=10, **serif_font)
        ax.set_xticks(np.linspace(0, 20, 11))
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=10)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(f"fig6_{alg}GHZ{n}ALT.svg")
        fig.savefig(f"fig6_{alg}GHZ{n}ALT.pdf")
        plt.show()

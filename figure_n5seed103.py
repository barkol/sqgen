# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 12:01:05 2022

@author: bartkiewicz
"""

"""===========================================================================
A program that draws graphs for a given number of qubits from imported data,
for the paper: Synergic quantum generative machine learning (arXiv:2112.13255)
=============================================================================="""

import numpy as np
import matplotlib.pyplot as plt

Old = True
New = True
seed = 103

mono_font = {'fontname':'monospace'}
serif_font = {'fontname':'serif'}

plt.rcParams['text.usetex'] = True

# Create plot
cm = 1/2.54  # centimeters in inches

for n in [5]:

    baseNew = "_iter_new_seed" + str(seed) + "_n_"+ str(n) +"disc1ALT.npy"        
    baseOld = "_iter_old_seed" + str(seed) + "_n_"+ str(n) +"disc1ALT.npy"        
    
    cases = {"SQGEN":baseNew, "QGAN":baseOld}
    
    for alg in cases.keys():
        
        
        base = cases[alg]
        y1 = np.load("prt"+base)
        y2 = np.load("pft"+base)
        y3 = np.load("fid"+base)
        
        
        if alg == "QGAN":
            y1 = np.array([y1[m] for m in range(len(y3.tolist())) ])
            y2 = np.array([y2[m] for m in range(len(y3.tolist())) ])
        
        fig = plt.figure(figsize=(8.5*cm, 6*cm))
        ax = fig.add_subplot(1, 1, 1)


        
        plt.plot(y1,'g',label=r"$1-p$",marker=">")
        plt.plot(y2,'r',label=r"$1-q$",marker="<")
        plt.plot(y3,'b',label=r"$F$",marker="^")
        
        tit = (r"$\mathrm{"+alg+ r"}:\quad n=" + str(n) + ",\quad \mathrm{seed}=" 
                 + str(seed) + r"$")
        
        plt.title(tit)
        plt.legend()
        plt.legend(loc='center right')
        
        plt.ylabel(r'$\mathrm{Learning\; parameters}$',fontsize = 10,**serif_font)
        plt.xticks(fontsize = 10,**serif_font)
        plt.yticks(fontsize = 10,**serif_font)
        
        plt.ylim(-0.05,1.05)
        
        plt.xlabel(r'$\mathrm{Epoch}$',fontsize = 10,**serif_font)
        plt.xticks(np.linspace(0,20,11))
        plt.yticks(np.linspace(0,1,5))
        plt.grid(True)
        plt.ylabel(r'$\mathrm{Learning\; parameters}$',fontsize = 10,**serif_font)
        plt.xticks(fontsize = 10,**serif_font)
        plt.yticks(fontsize = 10,**serif_font)
        
        plt.tight_layout()
        
        plt.savefig("fig6_" + alg + "GHZ" + str(n) + "ALT.svg")
        plt.savefig("fig6_" + alg + "GHZ" + str(n) + "ALT.pdf")
        
        plt.show()
        

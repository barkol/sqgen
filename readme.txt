This repository contains Qiskit implementations of the quantum circuits
suggested in the paper "Synergic quantum generative machine learning,"
(arXiv:2112.13255v2).
As presented in the article, an example of training the network is to 
teach both recognition and generation of a n-qubit GHZ entangled state.
Program 'singlequbit_n1.ipynb' is responsible for training 
the SQGEN network for single-qubit input on a real programmable quantum computer.
Program 'multiqubit_n5seed103.py' performs calculations on quantum simulators 
for SQGEN and QGAN. Program 'figure_n5seed103.py' is used to plot the outcomes of 'multiqubit_n5seed103.py'. The particular seed of a random number generator is set in line 14 ("seed in [103]"). The number of qubits n is set in line 299 ("n in [5]"). These numbers were varied to generate Fig.6 form the article (see also lines 18 and 28 in "figure_n5seed103.py").

The programs responsible for training the network (both for the single qubit 
case and the multi-qubit case) include definitions of the real state generator 
circuits and the trained (variational) generator and discriminator.
The code includes functions responsible for determining the probabilities 
of determining the output state as real/fake, and the fidelity of the obtained state. The programs also include definitions of the cost functions of both the generator, discriminator, and the total cost function minimized for training in the SQGEN approach. 

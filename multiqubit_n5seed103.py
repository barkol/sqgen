# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:09:01 2022

@author: karol
"""

"""==============================================================================================================
A program that trains circuits with a given number of qubits on a quantum simulator for SQGEN and QGAN approaches
suggested in the paper: Synergic quantum generative machine learning (arXiv:2112.13255)
================================================================================================================="""


for seed in [103]:
    """Import of needed libraries"""
    
    from qiskit import QuantumCircuit, execute, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit import QuantumRegister, ClassicalRegister,Aer 
    from qiskit.extensions import Initialize
    from qiskit.circuit.library import MCMT, RZGate, RYGate
    from scipy.optimize import minimize
    
    import numpy as np
    from numpy import pi,sqrt
    
    
    sv_sim = Aer.get_backend('statevector_simulator')
    backend = sv_sim
    
    
    """=============================================
    Circuit corresponding to the trained generator
    ================================================"""
    
    def GeneratorG(n,xG):
         x = ParameterVector('x',4*n)
         qr = QuantumRegister(n)
         qc = QuantumCircuit(qr, name='G')
         for i in range(n):
             if i < n-1:
                 gate = MCMT(RZGate(2*pi*x[i]),n-i-1, 1)
                 qc.append(gate,[m for m in range(n-i)])
             else:
                 qc.rz(2*pi*x[i],0)
         for i in range(n):
             if i < n-1:
                 gate = MCMT(RYGate(2*pi*x[i+n]),n-i-1, 1)
                 qc.append(gate,[m for m in range(n-i)])
             else:
                 qc.ry(2*pi*x[i+n],0)
         for i in reversed(range(n)):
             if i < n-1:
                 gate = MCMT(RYGate(2*pi*x[i+3*n]),n-i-1, 1)
                 qc.append(gate,[m for m in range(n-i)])
             else:
                 qc.ry(2*pi*x[i+3*n],0)
         for i in reversed(range(n)):
             if i < n-1:
                 gate = MCMT(RZGate(2*pi*x[i+2*n]),n-i-1, 1)
                 qc.append(gate,[m for m in range(n-i)])
             else:
                 qc.rz(2*pi*x[i+2*n],0)
         result = qc.to_instruction({x:xG})
         return result

    """===========================================================================
    Circuit corresponding to the real data generator and its conjugate transposition
    =============================================================================="""
        
    def GeneratorR(n,xR):
         qr = QuantumRegister(n)
         qc = QuantumCircuit(qr, name='R')
         
         v = [1/sqrt(2)]+(2**n-2)*[0] + [1/sqrt(2)] #GHZ state
         
         qc.initialize(v,[m for m in range(n)])
         result = qc.to_instruction()
         return result
       
    def GeneratorRdg(n,xR):
        #x =Parameter('x')
        qr = QuantumRegister(n)
        qc = QuantumCircuit(qr, name='Rdg')
        
        v = [1/sqrt(2)]+(2**n-2)*[0] + [1/sqrt(2)] #GHZ state
        
        qi=Initialize(v).gates_to_uncompute()
        qi = qi.to_instruction()
        qc.append(qi,qr)
    
        result = qc.to_instruction()
        
        return result    
       
    """===============================================
    Circuit corresponding to the trained discriminator
    =================================================="""

    def Discriminator(n,xD,testing=False):
        qr = QuantumRegister(n+1)
        qc = QuantumCircuit(qr, name='D')
        
        xD1=ParameterVector('xD1',4*n)
        sub_inst= GeneratorG(n,xD1)
        qc.append(sub_inst, [qr[i] for i in range(n)])
        
        if testing:
            shift = pi
        else:
            shift = pi/2
        
        gate = MCMT(RYGate(shift),n, 1)
        qc.append(gate,[m for m in (range(n+1))])
        qc.append(sub_inst.inverse(), [qr[i] for i in range(n)])
    
        result = qc.to_instruction({xD1:xD[0:4*n]})
        return result
     
    """=====================================================================================================================================
    Loops used to calculate the probabilities of correct recognition of states by the discriminator, and the fidelity of the obtained states
    ========================================================================================================================================"""    

    def real_true(n,x,prt=[]):
        global qR,xD,xR
        b = {xD:x[:4*n]} # ,xR:0
        qRb = qR.bind_parameters(b)
        job = execute(qRb,sv_sim)
        
        r = job.result()    
        sv = r.get_statevector()
        p = np.abs(sv[0])**2
        prt.append(p)
        return 1-p    
    
    def fake_true(n,x,pft=[]):
        global qG,xD,xR
        b = {xD:x[:(4*n)],xG:x[(4*n):(4*n+4*n)]}
        qGb = qG.bind_parameters(b)
        job = execute(qGb,sv_sim)
        r = job.result()    
        sv = r.get_statevector()
        p = np.abs(sv[0])**2
        pft.append(p)
        return 1-p 
        
    def fidelityRG(n,x,fid=[]):
        global qRG,xR,xG
        b = {xG:x[(4*n):(4*n+4*n)]} # xR:0,
        qRGb = qRG.bind_parameters(b)
        job = execute(qRGb,sv_sim)
        r = job.result()    
        sv = r.get_statevector()
        result = np.abs(sv[0])**2
        fid.append(result)
        return result

    prt_new = []
    pft_new = []
    fid_new = []
    costNew_evals = 0
    
    """==================================================================================================
    Loops to calculating the total cost function, generator cost function and discriminator cost function
    ======================================================================================================"""
    
    def costNew(n,x,verbose=False):
         global q,xG,xD,xR,costNew_evals
         global prt_new, pft_new, fid_new
         costNew_evals+=1
         b = {xD:x[:(4*n)],xG:x[(4*n):(4*n+4*n)]} # ,xR:0   
         qb = q.bind_parameters(b)
         job = execute(qb,sv_sim)
         r = job.result()    
         sv = np.asarray(r.get_statevector()).tolist()
         nR = 1e-8*np.linalg.norm(np.array(x))
         result = 1-np.abs(sv[0])**2 +nR 
         if verbose: print(np.array([result,
                                     real_true(n,x,prt_new),
                                     fake_true(n,x,pft_new),
                                     fidelityRG(n,x,fid_new)]),end="\n")
         return result
    
    
    
    prt_old = []
    pft_old = []
    fid_old = []
    disc_cost_evals = 0
    gen_cost_evals = 0
     
    
    def disc_cost(n,x):
         global disc_cost_evals
         global prt_old, pft_old
         disc_cost_evals += 1
         q = fake_true(n,x,pft_old)
         p = real_true(n,x,prt_old)
         d = np.abs(p-q)
         dp = np.abs(p)
         nR = 1e-8*np.linalg.norm(np.array(x[:(4*n)]))
         result = np.abs(1-d)  - dp   +nR # - np.log(d)
         return result
     
    def gen_cost(n,x):
         global gen_cost_evals
         global fid_old
         gen_cost_evals += 1
         nG = 1e-8*np.linalg.norm(np.array(x[(4*n):(4*n+4*n)]))
         d = 1-fidelityRG(n,x,fid_old)+ nG
         result = np.log(d)  
         return result
      
    """============================================
    Learning process according to the QGAN approach
    ==============================================="""
    
    prt_iter_old = []
    pft_iter_old = []
    fid_iter_old = [] 
    gen_cost_evals = 0
    disc_cost_evals = 0
    
    def learnOld(n,itr,seed,verbose=True):
        np.random.seed(seed)  
        x0 = np.random.rand((4*n+4*n))
        xD = x0[:(4*n)]
        xG = x0[(4*n):(4*n+4*n)]
        global prt_iter_old, pft_iter_old, fid_iter_old
        tprr = real_true(n,x0.tolist(),prt_iter_old)
        tpfr = fake_true(n,x0.tolist(),pft_iter_old)
        tfid = fidelityRG(n,x0.tolist(),fid_iter_old)
        if verbose: print("start: \t",np.array([tprr,tpfr,tfid]),end="\n")
        for l in range(itr):
          print("QGAN iteration:\t" + str(l))
          for m in range(1):
            solD = minimize(lambda xD: disc_cost(n,xD.tolist() + xG.tolist()), 
                       xD, 
                       method  = alg,
                       options = optD)
            xD = solD.x
            tprr = real_true(n,xD.tolist() + xG.tolist(),prt_iter_old)
            tpfr = fake_true(n,xD.tolist() + xG.tolist(),pft_iter_old)
            if verbose: print("D: ",np.array([l, m, tprr,tpfr,tfid]),end="\n")
          for m in range(1):
            solG = minimize(lambda xG: gen_cost(n,xD.tolist() + xG.tolist()), 
                       xG, 
                       method  = alg,
                       options = optG)
            xG = solG.x
            tfid = fidelityRG(n,xD.tolist() + xG.tolist(),fid_iter_old)
            if verbose: print("G: ",np.array([l, m, tprr,tpfr,tfid]),end="\n")
        
        print("============= COST FUNCTION EVALUATIONS ============")
        print(np.array([gen_cost_evals,disc_cost_evals]))
        return
    
    """=============================================
    Learning process according to the SQGEN approach
    ================================================"""    
          
    itrNew = 20
    itrOld = 20
    
    
    opt = {'maxiter': 1,
           'disp':False,
           'eps':1e-6,
           'finite_diff_rel_step':1e-8}
     
    optD = {'maxiter': 1,
           'disp':False,
           'eps':1e-6,
           'finite_diff_rel_step':1e-8}
    
    optG = {'maxiter': 1,
           'disp':False,
           'eps':1e-6,
           'finite_diff_rel_step':1e-8}
    
    alg = 'BFGS'
    
    
    import time
    
    def save_time(n, t, dq,dqR,dqG,dqRG,a,b,c):
        # creating a file
        global seed
        fileObject = open("log_times_seed" + str(seed)+"_paperALT.txt", "a")
         
        # writing into the file
        fileObject.write(str(n) + "\t" + str(t) + "\t" + str(dq) + "\t" + str(dqR) + "\t"
                         + str(dqG) + "\t" + str(dqRG) + "\t")
        fileObject.write(str(a) + "\t" + str(b) + "\t" + str(c) + "\n")
        fileObject.flush() 
        fileObject.close()
    
    
    for n in [5]: # number of qubits
        # https://qiskit.org/documentation/stubs/qiskit.circuit.library.MCMT.html#qiskit.circuit.library.MCMT
        
        print("GHZ:\t" + str(n))        
   
        xG = ParameterVector('xG',4*n)
        xD = ParameterVector('xD',4*n)  
        xR = Parameter('xR')
        D = Discriminator(n,xD)
        Dt = Discriminator(n,xD,testing=True)
        G = GeneratorG(n,xG)
        R = GeneratorR(n,xR)
        Rdg = GeneratorRdg(n,xR)
        
        qr = QuantumRegister(n+1, 'q')
        cr = ClassicalRegister(n+1, 'c')
        qc = QuantumCircuit(qr,cr)
        qc.append(R,[qr[i] for i in range(n)])
        qc.append(D,[qr[i] for i in range(n+1)])
        qc.x(qr[n])
        qc.append(D.inverse(),[qr[i] for i in range(n+1)])
        qc.append(G.inverse(),[qr[i] for i in range(n)])
        q = transpile(qc,backend)
        dq= q.depth()
    
        qrR = QuantumRegister(n+1,'q')
        crR = ClassicalRegister(n+1, 'c')
        qcR = QuantumCircuit(qrR,crR)
        qcR.append(R,[qrR[i] for i in range(n)])
        qcR.append(Dt,[qrR[i] for i in range(n+1)])
        qcR.append(Rdg,[qrR[i] for i in range(n)])
        qR = transpile(qcR,backend)
        dqR = qR.depth()
    
        qrG = QuantumRegister(n+1,'q')
        crG = ClassicalRegister(n+1,'c')
        qcG = QuantumCircuit(qrG,crG)
        qcG.append(G,[qrG[i] for i in range(n)])
        qcG.append(Dt,[qrG[i] for i in range(n+1)])
        qcG.append(G.inverse(),[qrG[i] for i in range(n)])
        qG = transpile(qcG,backend)
        dqG = qG.depth()
    
    
        qrRG = QuantumRegister(n,'q')
        crRG = ClassicalRegister(n,'c')
        qcRG = QuantumCircuit(qrRG,crRG)
        qcRG.append(R,[qrRG[i] for i in range(n)])
        qcRG.append(G.inverse(),[qrRG[i] for i in range(n)])
        qRG = transpile(qcRG,backend)
        dqRG = qRG.depth()
        
        """==========================
        Saving SQGEN results to files
        ============================="""
        
        if True:
            start_time = time.time()
            
            np.random.seed(seed)  
            x0 = np.random.rand((4*n+4*n))
        
            prt_iter_new = []
            pft_iter_new = []
            fid_iter_new = []
            cost_new = []
            for m in range(itrNew):
              print("SQGEN iteration:\t" + str(m))
              fake_true(n,x0,pft_iter_new)  
              real_true(n,x0,prt_iter_new)  
              fidelityRG(n,x0,fid_iter_new)
              cost_new.append(costNew(n,x0))
              sol = minimize(lambda x: costNew(n,x,False),x0,method  = alg, options = opt) 
              x0 = sol.x
             
            fake_true(n,x0,pft_iter_new)  
            real_true(n,x0,prt_iter_new)  
            fidelityRG(n,x0,fid_iter_new)
            t=(time.time() - start_time)
            print("NEW --- %s seconds ---" % t)
            save_time(n, t, dq,dqR,dqG,dqRG,0,0,costNew_evals)
            
            print(costNew_evals)  
            
            np.save("prt_new_seed_" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",prt_new)
            np.save("pft_new_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",pft_new)
            np.save("fid_new_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",fid_new)   
            np.save("prt_iter_new_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",prt_iter_new)
            np.save("pft_iter_new_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",pft_iter_new)
            np.save("fid_iter_new_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",fid_iter_new)
            
            costNew_evals = 0
            prt_new = []
            pft_new = []
            fid_new = []
            prt_iter_new = []
            pft_iter_new = []
            fid_iter_new = []
            cost_new = []
            
        """==========================
        Saving QGAN results to files
        ============================="""
        start_time = time.time()
        learnOld(n,itrOld,seed,False)
        t=(time.time() - start_time)
        print("OLD --- %s seconds ---" % t)
        save_time(n, t, dq,dqR,dqG,dqRG,len(prt_old),
                  len(pft_old),len(fid_old))
        
            
        np.save("prt_old_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",prt_old)
        np.save("pft_old_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",pft_old)
        np.save("fid_old_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",fid_old)
        np.save("prt_iter_old_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",prt_iter_old)
        np.save("pft_iter_old_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",pft_iter_old)
        np.save("fid_iter_old_seed" + str(seed) + "_n_" + str(n) + "disc1ALT.npy",fid_iter_old)
    
        prt_old = []
        pft_old = []  
        fid_old = []
        prt_iter_old= [] 
        pft_iter_old = [] 
        fid_iter_old = []
        gen_cost_evals = 0
        disc_cost_evals = 0 	

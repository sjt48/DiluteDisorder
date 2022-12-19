"""
Exact Diagonalization Tests for Dilute Disorder
S. J. Thomson
Dahlem Centre for Complex Quantum Systems, FU Berlin
steven.thomson@fu-berlin.de
steventhomson.co.uk
https://orcid.org/0000-0001-9065-9842
---------------------------------------------

"""

import os,h5py,sys,itertools,copy
from psutil import cpu_count
from datetime import datetime
from multiprocessing import Pool, freeze_support
from functools import partial
# Import ED code from QuSpin
from quspin.operators import hamiltonian,exp_op
from quspin.tools.measurements import ED_state_vs_time
from quspin.basis import spin_basis_1d
from scipy.linalg import eig
from scipy import sparse as sp
# Set up threading options
os.environ['OMP_NUM_THREADS']= str(int(1)) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(1)) # set number of MKL threads to run in parallel

import numpy as np
import matplotlib.pyplot as plt

def level_stat(levels):
    """ Function to compute the level spacing statistics. """
    list1 = np.zeros(len(levels))
    lsr = 0.
    for i in range(1,len(levels)):
        list1[i-1] = levels[i] - levels[i-1]
    for j in range(len(levels)-2):
        lsr += min(list1[j],list1[j+1])/max(list1[j],list1[j+1])
    lsr *= 1/(len(levels)-2)
    
    return lsr

def ED(n,J0,times,delta,var):

        rep,d,p = [i for i in var]
        print(n,J0,rep,d,delta,p)
        np.random.seed()
        hlist = [np.random.choice([1.0,np.random.uniform(-d,d)],p=[1-p,p]) for i in range(n)]
        # hlist = [np.random.choice([1.0,d],p=[1-p,p]) for i in range(n)]

        h = [[hlist[i],i] for i in range(n)]
        J = [[J0,i,i+1] for i in range(n-1)]

        Delta = [[delta,i,i+1] for i in range(n-1)]
        static = [["z",h],["xx",J],["yy",J],["zz",Delta]]

        dynamic=[]
        no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
        basis = spin_basis_1d(n,Nup=n//2,pauli=False)

        H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128,**no_checks)
        E1,V1 = H.eigh()

        inf_corr_list = np.zeros(len(times),dtype=complex)

        pure_state = []
        temp = np.random.normal(0,np.sqrt(0.5),basis.Ns)+1j*np.random.normal(0,np.sqrt(0.5),basis.Ns)
        temp *= 1/np.sqrt(np.dot(np.conjugate(temp),temp))
        pure_state += [temp]

        for i in [n//2]:
            operator = [["z",[i],1.0]]
            psi1 = copy.deepcopy(pure_state[0])
            psi1_t = ED_state_vs_time(psi1,E1,V1,times,iterate=False)
            psi2 = copy.deepcopy(pure_state[0])
            psi2_op = basis.inplace_Op(psi2,operator,np.complex128)
            psi2_op_t = ED_state_vs_time(psi2_op,E1,V1,times,iterate=False)
            psi2_op_t2 = basis.inplace_Op(psi2_op_t,operator,np.complex128)
            inf_corr_list = 4*np.einsum("ij,ij->j",psi1_t.conj(),psi2_op_t2)/np.einsum("ij,ij->j",psi1_t.conj(),psi1_t)

        st = "".join("10" for i in range(n//2))
        iDH = basis.index(st)
        psi1 = np.zeros(basis.Ns)
        psi1[iDH] = 1.0
        psi1_t = ED_state_vs_time(psi1,E1,V1,times,iterate=False)

        for i in [n//2]:
            operator = [["z",[i],1.0]]
            psi2 = np.zeros(basis.Ns)
            psi2[iDH] = 1.0
            psi2_op = basis.inplace_Op(psi2,operator,np.complex128)
            psi2_op_t = ED_state_vs_time(psi2_op,E1,V1,times,iterate=False)
            psi2_op_t2 = basis.inplace_Op(psi2_op_t,operator,np.complex128)
            corrlist = 4*np.einsum("ij,ij->j",psi1_t.conj(),psi2_op_t2)

        imblist = np.zeros((n,len(times)))
        for site in range(n):
            # Time evolution of observables
            n_list = [hamiltonian([["z",[[1.0,site]]]],[],basis=basis,dtype=np.complex64,**no_checks)]
            n_t = np.vstack([n.expt_value(psi1_t).real for n in n_list]).T
            imblist[site] = ((-1)**site)*n_t[::,0]
        imb = np.sum(imblist,axis=0)/n

        ent = np.zeros(len(times))
        for t in range(len(times)):
            ent[t] = basis.ent_entropy(psi1_t[::,t])["Sent_A"]

        if not os.path.exists('data/dataN%s/dyn-d%.2f-Jz%.2f-p%s-rep%s.h5' %(n,d,delta,p,rep)):
            #==============================================================
            # Export data   
            with h5py.File('data/dataN%s/dyn-d%.2f-Jz%.2f-p%s-rep%s.h5' %(n,d,delta,p,rep),'w') as hf:
                hf.create_dataset('eig',data=E1)
                hf.create_dataset('imb',data=imb)
                hf.create_dataset('imblist',data=imblist)
                hf.create_dataset('n_t',data=n_t)
                hf.create_dataset('ent',data=ent)
                hf.create_dataset('corr',data=corrlist)
                hf.create_dataset('inf_temp_corr',data=inf_corr_list)
                hf.create_dataset('tlist',data=times)
                hf.create_dataset('hlist',data=hlist)


#----------------------------------------------------------------
if __name__ == '__main__': 

    startTime = datetime.now()
    print('Start time: ', startTime)

    freeze_support()

    n = int(sys.argv[1])
    J= 1.0
    delta = float(sys.argv[2])
    reps = int(sys.argv[3])
    times = np.logspace(-2,3,100,base=10.0,endpoint=True)

    if not os.path.exists('data/dataN%s' %(n)):
        os.makedirs('data/dataN%s' %(n))

    print('Number of CPUs = %s' %(cpu_count(logical=False)))
 
    pool = Pool(processes = cpu_count(logical=False))
    
    combinations = itertools.product(range(reps),[1.0,3.0,6.0,9.0],[0.1,0.2,0.5,0.9])

    pool.map(partial(ED,n,J,times,delta),combinations)

    pool.close()
    pool.join()


    print('End time: ', datetime.now() - startTime)
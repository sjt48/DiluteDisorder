"""
Exact Diagonalization for Dilute Disorder
S. J. Thomson
Dahlem Centre for Complex Quantum Systems, FU Berlin
steven.thomson@fu-berlin.de
steventhomson.co.uk
https://orcid.org/0000-0001-9065-9842
---------------------------------------------

Code intended for reproducability of results only.

"""

import os,h5py,sys
from psutil import cpu_count
from datetime import datetime
from multiprocessing import Pool, freeze_support
from functools import partial

# Import ED code from QuSpin
from quspin.operators import hamiltonian 
from quspin.tools.measurements import ED_state_vs_time
from quspin.basis import spin_basis_1d

# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(2)) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(2)) # set number of MKL threads to run in parallel
import numpy as np

def level_stat(levels):
    """ Function to compute the level spacing statistics."""
    list1 = np.zeros(len(levels))
    lsr = 0.
    for i in range(1,len(levels)):
        list1[i-1] = levels[i] - levels[i-1]
    for j in range(len(levels)-2):
        lsr += min(list1[j],list1[j+1])/max(list1[j],list1[j+1])
    lsr *= 1/(len(levels)-2)
    
    return lsr

def ED(n,J0,times,rep,d,delta,p,dis_type='random'):

        np.random.seed()
        if dis_type == 'random':
            hlist = [np.random.choice([1.0,np.random.uniform(-d,d)],p=[1-p,p]) for i in range(n)]
            folder = "data"
        else:
            hlist = [np.random.choice([1.0,d],p=[1-p,p]) for i in range(n)]
            folder = "data2"

        h = [[hlist[i],i] for i in range(n)]
        J = [[J0,i,i+1] for i in range(n-1)]

        # Build Hamiltonian and basis
        Delta = [[delta,i,i+1] for i in range(n-1)]
        static = [["z",h],["xx",J],["yy",J],["zz",Delta]]
        dynamic=[]
        no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
        basis = spin_basis_1d(n,pauli=False,Nup=n//2)
        H = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
        E1,V1 = H.eigh()

        # Pick initial state
        # NB: for odd system sizes, be sure to check the state is reliable
        st = "".join("10" for i in range(n//2))
        iDH = basis.index(st)
        psi1 = np.zeros(basis.Ns)
        psi1[iDH] = 1.0
        psi1_t = ED_state_vs_time(psi1,E1,V1,times,iterate=False)

        # Compute correlation function
        for i in [n//2]:
            operator = [["z",[i],1.0]]
            psi2 = np.zeros(basis.Ns)
            psi2[iDH] = 1.0
            psi2_op = basis.inplace_Op(psi2,operator,np.float64)
            psi2_op_t = ED_state_vs_time(psi2_op,E1,V1,times,iterate=False)
            psi2_op_t2 = basis.inplace_Op(psi2_op_t,operator,np.float64)
            corrlist = 4*np.einsum("ij,ij->j",psi1_t.conj(),psi2_op_t2)                         

        # Compute imbalance
        imblist = np.zeros((n,len(times)))
        for site in range(n):
            n_list = [hamiltonian([["z",[[1.0,site]]]],[],basis=basis,dtype=np.complex64,**no_checks)]
            n_t = np.vstack([n.expt_value(psi1_t).real for n in n_list]).T
            imblist[site] = ((-1)**site)*n_t[::,0]/n/2
        n_t = 2*np.sum(imblist,axis=0)

        # Compute entanglement entropy
        ent = np.zeros(len(times))
        for t in range(len(times)):
            ent[t] = basis.ent_entropy(psi1_t[::,t])["Sent_A"]

        #==============================================================
        # Export data   
        if not os.path.exists('%s/dataN%s' %(folder,n)):
            os.makedirs('%s/dataN%s/' %(folder,n))
        with h5py.File('%s/dataN%s/dyn-d%.2f-Jz%.2f-p%.2f-rep%s.h5' %(folder,n,d,delta,p,rep),'w') as hf:
            hf.create_dataset('eig',data=E1)
            hf.create_dataset('imb',data=n_t)
            hf.create_dataset('ent',data=ent)
            hf.create_dataset('corr',data=corrlist)
            hf.create_dataset('tlist',data=times)
            hf.create_dataset('hlist',data=hlist)

#----------------------------------------------------------------
if __name__ == '__main__': 

    startTime = datetime.now()
    print('Start time: ', startTime)
    freeze_support()

    n = int(sys.argv[1])
    reps = int(sys.argv[3])
    J= 1.0
    delta = float(sys.argv[2])
    times = np.logspace(-2,3,100,base=10.0,endpoint=True)

    pool = Pool(processes = 3)

    for rep in range(reps):

        # For phase diagram
        #for d in [0.2*k for k in range(1,51)]:
        #    plist = [0.05*i for i in range(1,21)]
        #    ed = pool.map(partial(ED,n,J,times,rep,d,delta),plist)

        # For individual plots
        for d in [0.0,3.0,6.0,9.0]:
            plist = [0.2,0.5,0.8]
            ed = pool.map(partial(ED,n,J,times,rep,d,delta),plist)

    pool.close()
    pool.join()

    print('End time: ',datetime.now()-startTime)

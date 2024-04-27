from utils.SimFunctions import SimData
from utils.EstGMM import Estimate
from utils.DataDesc import CustomPlot
import multiprocessing as mp
import numpy as np

##########################################################
# Define function to simulate data & estimate model
##########################################################

def SimOnce(n, T, J, dgpopt, nrm, ffopt, adj, seed):
    np.random.seed(seed)
    data = SimData(n, T+1, J, dgpopt)
    r = Estimate(data, nrm, ffopt, adj)
    return r

def SimMulti(n, T, J, dgpopt, nrm, ffopt, adj, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(SimOnce, 
                     [(n, T, J, dgpopt, nrm, ffopt, adj, seed) 
                      for seed in seeds])
    pool.close()
    return r

##########################################################
# Main program
##########################################################

if __name__ == '__main__':

    # Import packages
    import matplotlib.pyplot as plt
    from utils.SimFunctions import DGP, MomsFewVarsDGP

    # DGP
    seed = 1118
    np.random.seed(seed)
    T, J = 4, 2
    n = 50000
    dgpopt = 'fewvars'
    psiMtrue, mu_tr, nuP, betaL, betaPhi, xmeans, covx1to3, pL = DGP(T, J, dgpopt)
    psi_true = psiMtrue @ pL
    avg_moms_X = MomsFewVarsDGP(T)

    # Parameters for estimation
    nrm = avg_moms_X[0]
    ffopt = 'np'
    adj = 'ipw'
    iters = 100
    seeds = np.random.choice(100000, iters)

    # Results
    r = SimMulti(n, T, J, dgpopt, nrm, ffopt, adj, seeds)

    # Unpack results
    keys = [k for k in r[0].keys() if r[0][k] is not None]
    avg = {k: np.mean([r[i][k] for i in range(iters)], axis=0) for k in keys}
    notSEkeys = [k for k in keys if 'SE' not in k]
    se_iters = {k: np.std([r[i][k] for i in range(iters)], axis=0) for k in notSEkeys}
    
    # Plots 

    # Plot structural hazard
    CustomPlot([psi_true, avg['psi']], [None, avg['psiSE']],
           legendlabs=['True', 'Est'], title = 'Structural Hazard')
    
    # Comparing SEs
    CustomPlot([avg['psiSE'], se_iters['psi']], 
               legendlabs=['Anal SEs', 'Iter SEs'], ydist=0.01)
    
    # Plot Average Type Moments
    CustomPlot([avg_moms_X, avg['mu']], [None, avg['muSE']], ylims=[0, 1],
               legendlabs=['True', 'Est'], title = 'Average Type Moments')

##########################################################





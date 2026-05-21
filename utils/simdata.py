import numpy as np
import pandas as pd
import sympy as sp
from utils.simdgp import nupars
from sympy.stats import sample

##########################################################
# Function to simulate data
##########################################################

# Possible to fasten this code?


def sim_data(n, dgpqnts):

    # Unpack parameters
    Xdgp, psiM = dgpqnts["X"], dgpqnts["psiM"]
    X1, X2 = sp.symbols("X1 X2")
    phiXdgp = sp.lambdify((X1, X2), dgpqnts["phiX"], "numpy")
    pL_Xdgp = sp.lambdify((X1, X2), dgpqnts["pL_X"], "numpy")
    T, J = psiM.shape

    # Generate X & phi(X)
    X = np.zeros((n, len(Xdgp)))
    X[:, 0] = 1
    xseeds = np.random.randint(1, 100000, len(Xdgp) - 1)
    for k in range(1, len(Xdgp)):
        X[:, k] = sample(Xdgp[k], size=n, seed=xseeds[k - 1]).astype(int)
    phiX = np.array([phiXdgp(X[i, 1], X[i, 2]) for i in range(n)])
    phiX = phiX.flatten().reshape(n, 1)

    # Generate nu
    pars = np.array([nupars(X[i, 1], X[i, 2]) for i in range(n)])
    nu = np.zeros((n, 1))
    for i in range(n):
        nu[i] = np.random.beta(pars[i, 0], pars[i, 1])

    # Notice assignment mechanism & corresponding psi_l(d)
    pL_X = np.array(pL_Xdgp(X[:, 1], X[:, 2])).T
    L = np.array([0] * n)
    psi_i = np.zeros((n, T))
    for i in range(n):
        L[i] = np.random.choice(range(J), p=pL_X[i, :])
        psi_i[i, :] = psiM[:, L[i]]

    # Exit probabilities & unemployment duration
    exited = np.array([0] * n)
    exit_probs = psi_i * phiX * nu
    crit = np.random.random((n, T))
    exit = (exit_probs > crit).astype(int)
    for t in range(T):
        exited += exit[:, t]
        exit[exited != 0, t + 1 :] = 0
    unemp_dur = np.argmax(exit, axis=1).astype(float)
    unemp_dur[exited == 0] = np.nan  # we do not observe unemp_dur > T

    # (Independent) Censoring
    censtime = np.random.choice(range(1, T), n)
    cens = ((censtime < unemp_dur) | (exited == 0)).astype(int)
    obsdur = unemp_dur.copy()
    obsdur[cens == 1] = censtime[cens == 1]
    T_bar = round(T / 2)
    cens_ind = (censtime > T_bar).astype(int)

    # Create pandas dataframe (return X, L, censored, obs_dur)
    data = np.column_stack((obsdur, cens, L, cens_ind, X))
    col_labs = ["dur", "cens", "notice", "cens_ind"] + [
        "x" + str(i) for i in range(len(Xdgp))
    ]
    data = pd.DataFrame(data, columns=col_labs)

    return data, nu, pL_X, phiX


##########################################################

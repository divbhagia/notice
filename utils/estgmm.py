import numpy as np
from scipy.optimize import minimize
from utils.datadesc import pred_ps
from utils.datamoms import data_moms
from utils.inference import opt_wt_mat, indv_moms, indv_moms_ipw, std_errs
from utils.esthelpers import unstack, model_moms, numgrad, unstack_all

##########################################################
# GMM moments & objective function at given parameters
##########################################################


def avg_moms(thta, exits, surv, nrm, ffopt):
    T, J = exits.shape
    h_model = model_moms(*unstack(T, J, thta, nrm, ffopt)[:2])
    m = np.zeros([T, J])
    for t in range(T):
        for j in range(J):
            m[t, j] = exits[t, j] - h_model[t, j] * surv[t, j]
    m = m.reshape(T * J, 1).flatten()
    return m


def objfun(thta, exits, surv, nrm, ffopt, W=None):
    m = avg_moms(thta, exits, surv, nrm, ffopt)
    W = np.eye(m.size) if W is None else W
    obj = m.T @ W @ m
    return obj


##########################################################
# Two-step GMM
##########################################################


def gmm(data, nrm, ffopt="np", ps=None):

    # Data moments
    exits, surv, _, _ = data_moms(data, ps)

    # Initialize
    T, J = exits.shape
    opt = ffopt["opt"] if isinstance(ffopt, dict) else ffopt
    if opt == "np":
        num_pars = 2 * (T - 1) + J
    elif opt == "baseline":
        num_pars = 2 + (T - 1) + J
    num_moms = T * J
    thta0 = 0.25 * np.ones(num_pars)
    W = np.eye(num_moms)

    # Wrapper for the numerical gradient
    def numgrad_wrapper(x, *args):
        return numgrad(objfun, x, *args).flatten()

    def minimize_(thta0, W):
        opts = {"disp": False, "maxiter": 100000}
        results = minimize(
            objfun,
            thta0,
            method="L-BFGS-B",
            tol=1e-32,
            args=(exits, surv, nrm, ffopt, W),
            options=opts,
            jac=numgrad_wrapper,
        )
        return results.x

    # Two-step GMM
    thta_hat = minimize_(thta0, W)
    if num_moms > num_pars:
        # print('Two-step GMM')
        W = opt_wt_mat(thta_hat, data, nrm, ffopt, MomsFunc=indv_moms)
        thta_hat = minimize_(thta_hat, W)

    # Hansen-Sargan J-statistic
    n = len(data)
    W = opt_wt_mat(thta_hat, data, nrm, ffopt, ps, MomsFunc=indv_moms)
    Jstat = n * objfun(thta_hat, exits, surv, nrm, ffopt, W)

    return thta_hat, Jstat


##########################################################
# Function to estimate and calculate standard errors
##########################################################


def estimate(data, nrm, ffopt, adj="none", seadj=False):

    # Estimate
    nL = data["notice"].value_counts().sort_index().values
    T, J = len(data["dur"].unique()) - 1, len(data["notice"].unique())
    if adj == "ipw":
        ps, coefs = pred_ps(data)
        thta_hat, Jstat = gmm(data, nrm, ffopt, ps)
        thta_all = np.append(thta_hat, coefs)
        if seadj:
            se = std_errs(thta_all, data, nrm, ffopt, MomsFunc=indv_moms_ipw)
            se = se[: len(thta_hat)]
        else:
            se = std_errs(thta_hat, data, nrm, ffopt, MomsFunc=indv_moms)
    elif adj == "none":
        ps, coefs = None, None
        thta_hat, Jstat = gmm(data, nrm, ffopt)
        se = std_errs(thta_hat, data, nrm, ffopt, MomsFunc=indv_moms)
    psin, psi, par, mu, psinSE, psiSE, parSE, muSE = unstack_all(
        T, J, nL, thta_hat, se, nrm, ffopt
    )

    # Stack results
    r = {
        "psin": psin,
        "psi": psi,
        "par": par,
        "mu": mu,
        "psinSE": psinSE,
        "psiSE": psiSE,
        "parSE": parSE,
        "muSE": muSE,
        "ps": ps,
        "coefs": coefs,
        "Jstat": Jstat,
    }

    return r


##########################################################

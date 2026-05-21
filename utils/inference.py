import numpy as np
from utils.datamoms import data_moms
from utils.esthelpers import unstack, model_moms, numgrad
from utils.datadesc import pred_ps

##########################################################
# Individual moments incomplete
##########################################################


def indv_moms(thta, data, nrm, ffopt="np", ps=None):
    T, J = len(data["dur"].unique()) - 1, len(data["notice"].unique())
    h_model = model_moms(*unstack(T, J, thta, nrm, ffopt)[:2])
    _, _, exit_i, surv_i = data_moms(data, ps)
    m_i = exit_i - h_model * surv_i
    m_i = m_i.reshape(len(data), -1)
    return m_i


##########################################################
# Individual moments complete
##########################################################
# Only for IPW with logit model & two categories


def indv_moms_ipw(thta_all, data, nrm, ffopt="np"):

    # Unpack data
    notice = data["notice"]
    notcats = np.sort(notice.unique())
    T = len(data["dur"].unique()) - 1
    n, J = len(data), len(notcats)
    notX_vars = ["dur", "cens", "notice", "cens_ind"]
    X = data[[col for col in data.columns if col not in notX_vars]]
    X = np.array(X, dtype=float)

    # Unstack thta
    nvars = X.shape[1]
    if ffopt == "np":
        npars = 2 * (T - 1) + J
    elif ffopt == "baseline":
        npars = 2 + (T - 1) + J
    thta = thta_all[:npars]
    coefs = thta_all[npars:]
    if J > 2:
        coefs = coefs.reshape(nvars, J - 1)

    # All moments
    ps = pred_ps(data, coefs)[0]
    psmoms = np.zeros((n, (J - 1) * nvars))
    for j in range(1, J):
        psmoms[:, (j - 1) * nvars : j * nvars] = X * np.array(
            (notice == notcats[j]) - ps[:, j]
        ).reshape(-1, 1)
    m_i = np.column_stack([psmoms, indv_moms(thta, data, nrm, ffopt, ps)])

    return m_i


##########################################################
# Other functions for inference
##########################################################


def avg_moms_inference(*args, MomsFunc, **kwargs):
    m_i = MomsFunc(*args, **kwargs)
    m = m_i.mean(axis=0)
    return m


def opt_wt_mat(*args, MomsFunc, **kwargs):
    m_i = MomsFunc(*args, **kwargs)
    n, nmoms = m_i.shape
    omega_i = np.zeros([n, nmoms, nmoms])
    m_i = m_i.reshape(n, nmoms, 1)
    for i in range(n):
        omega_i[i, :, :] = m_i[i, :, :] @ m_i[i, :, :].T
    omega = np.mean(omega_i, axis=0)
    W = np.linalg.pinv(omega)
    return W


def std_errs(*args, MomsFunc, **kwargs):
    n = len(args[1])
    W = opt_wt_mat(*args, MomsFunc=MomsFunc, **kwargs)
    M = numgrad(avg_moms_inference, *args, MomsFunc=MomsFunc, **kwargs)
    V = np.linalg.pinv(M.T @ W @ M)
    se = np.sqrt(np.diag(V) / n)
    return se


##########################################################

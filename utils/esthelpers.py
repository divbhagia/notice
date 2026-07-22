import numpy as np

##########################################################
# Calculates the numerical gradient or Jacobian
##########################################################


def numgrad(f, x, *args, **kwargs):

    # Initialize
    eps = 1e-6
    f_x = f(x, *args, **kwargs)
    multiOut = isinstance(f_x, tuple)
    f_x = f_x[0] if multiOut else f_x
    nEqs = len(f_x) if isinstance(f_x, np.ndarray) else 1
    nVars = len(x)

    # Calculate gradient or Jacobian
    df_dx = np.zeros([nEqs, nVars])
    for k in range(nVars):
        dx = np.zeros(nVars)
        dx[k] = eps
        f_x_plus = f(x + dx, *args, **kwargs)
        f_x_minus = f(x - dx, *args, **kwargs)
        if multiOut:
            f_x_plus, f_x_minus = f_x_plus[0], f_x_minus[0]
        if nEqs > 1:
            for j in range(nEqs):
                df_dx[j, k] = (f_x_plus[j] - f_x_minus[j]) / (2 * eps)
        else:
            df_dx[0, k] = (f_x_plus - f_x_minus) / (2 * eps)
    df_dx = df_dx if nEqs > 1 else df_dx.T

    return df_dx


##########################################################
# Function outputs model implied hazard
##########################################################


def model_moms(psiM, mu, out="h"):

    # Initialize
    T, J = psiM.shape
    if mu.shape != (T, J):
        mu = np.repeat(mu.reshape(-1, 1), J, axis=1)

    # Density
    g = np.zeros((T, J))
    c = np.zeros((T, T, J))
    c[:, 0, :] = np.ones((T, J))
    for t in range(1, T):
        for k in range(1, T):
            for j in range(J):
                c[t, k, j] = c[t - 1, k, j] - psiM[t - 1, j] * c[t - 1, k - 1, j]
    for j in range(J):
        g[:, j] = psiM[:, j] * (c[:, :, j] @ mu[:, j])

    # Hazard rate
    h, S = np.zeros((T, J)), np.zeros((T + 1, J))
    h[0, :], S[0, :] = g[0, :], 1
    for t in range(1, T):
        S[t, :] = S[t - 1, :] * (1 - h[t - 1, :])
        h[t, :] = g[t, :] / S[t, :]
    S[T, :] = S[T - 1, :] * (1 - h[T - 1, :])

    # Return
    if out == "all":
        return h, g, S
    else:
        return h


##########################################################
# Other helper functions
##########################################################


def psi_baseline(par, T):
    a1, a2 = par[0], par[1]
    psi = np.zeros(T - 1)
    with np.errstate(invalid="ignore"):
        for d in range(1, T):
            psi[d - 1] = (a2 / a1) * (d / a1) ** (a2 - 1) / (1 + (d / a1) ** a2)
    return psi


# Note: ignoring error as it occurs due to initial values during optimization


def meanshift_add(k0, mu):  # SPECIFIC TO D = 4 & J = 2
    k = np.zeros((4, 2))
    k[0, 1] = k0
    k[1, 1] = k0 * (k0**1 + 2 * mu[0])
    k[2, 1] = k0 * (k0**2 + 3 * mu[0] * k0 + 3 * mu[1])
    k[3, 1] = k0 * (k0**3 + 4 * mu[0] * k0**2 + 6 * mu[1] * k0 + 4 * mu[2])
    return k


def meanshift_mult(k0):  # SPECIFIC TO D = 4 & J = 2
    k = np.ones((4, 2))
    k[0, 1] = k0
    k[1, 1] = k0**2
    k[2, 1] = k0**3
    k[3, 1] = k0**4
    return k


##########################################################
# Function unpacks model parameters from a stacked vector
##########################################################


def unstack(T, J, x, nrm, ffopt="np"):

    # Initialize
    muM = np.zeros((T, J))
    psiM = np.zeros((T, J))
    gamma = np.ones((T - 1, J))
    kappa = np.zeros((T, J))
    opt = ffopt if isinstance(ffopt, str) else ffopt["opt"]

    # Unpack parameters
    if isinstance(ffopt, dict):
        gamma = ffopt.get("gamma", gamma)
        kappa = ffopt.get("kappa", kappa)

    # If opt_eta is set, override gamma/kappa based on the requested parameterization:
    #   opt_eta = "kappa" -> kappa = eta (scale-shift on types), gamma = 1
    #   opt_eta = "gamma" -> gamma = eta on long group, kappa = 1
    if isinstance(ffopt, dict) and "opt_eta" in ffopt:
        eta_val = ffopt.get("eta", 1.0)
        if ffopt["opt_eta"] == "kappa":
            gamma = np.ones((T - 1, J))
            ffopt = dict(ffopt)
            ffopt["kappa0_mult"] = eta_val
        elif ffopt["opt_eta"] == "gamma":
            gamma = np.ones((T - 1, J))
            gamma[:, 1] = eta_val
            ffopt = dict(ffopt)
            ffopt.pop("kappa0_mult", None)
            ffopt.pop("kappa0_add", None)
        else:
            raise ValueError(f"Unknown opt_eta: {ffopt['opt_eta']}")

    # Extract common parameters
    psin = x[:J]
    mu = np.zeros(T)
    mu[0] = nrm
    mu[1:T] = x[J : J + T - 1]

    # Compute psi based on option
    if opt == "baseline":
        par = x[J + T - 1 :]
        psi = psi_baseline(par, T)
    else:  # "np"
        psi = x[J + T - 1 :]
        par = None

    # Apply kappa shifts and build muM
    if isinstance(ffopt, dict) and "kappa0_add" in ffopt:
        kappa = meanshift_add(ffopt["kappa0_add"], mu)
        for j in range(J):
            muM[:, j] = mu + kappa[:, j]
    elif isinstance(ffopt, dict) and "kappa0_mult" in ffopt:
        kappa = meanshift_mult(ffopt["kappa0_mult"])
        for j in range(J):
            muM[:, j] = mu * kappa[:, j]
    else:
        for j in range(J):
            muM[:, j] = mu

    # Build psiM
    for j in range(J):
        psiM[:, j] = np.concatenate(([psin[j]], psi * gamma[:, j]))

    # Return
    return (psiM, muM, par)


##########################################################
# Unstack PsiM further (including SEs) (Remove if not used)
##########################################################

# def unstack_psiM(nL, psiM, psiSE=None):
#     n, pi = nL.sum(), nL/nL.sum()
#     psin = psiM[0, :]
#     psi = np.sum(pi * psiM, axis=1)
#     if psiSE is not None:
#         piSE = np.sqrt(pi * (1-pi)/n)
#         psinSE = psiSE[0, :]
#         psiSE = psiSE[:, 0]
#         psiSE[0] = np.sqrt(((piSE * psin)**2 + (pi * psinSE)**2).sum())
#         return psi, psin, psiSE, psinSE
#     else:
#         return psi, psin

##########################################################
# Unstack standard errors
##########################################################


def unstack_all(T, J, nL, thta, se, nrm, ffopt="np"):

    # Initialize
    n, pi = nL.sum(), nL / nL.sum()
    piSE = np.sqrt(pi * (1 - pi) / n)
    muSE = np.zeros(T)
    psiSE = np.zeros(T)

    # Unstack parameters
    psiM, muM, par = unstack(T, J, thta, nrm, ffopt)
    mu = muM[:, 0]

    # Unstack parameters further
    psin = psiM[0, :]
    psi = np.sum(pi * psiM, axis=1)

    # Standard errors for psin & mu
    psinSE = se[:J]
    muSE[1:T] = se[J : J + T - 1]
    psiSE[0] = np.sqrt(((piSE * psin) ** 2 + (pi * psinSE) ** 2).sum())

    # Standard errors for psi
    if ffopt == "np":
        psiSE[1:] = se[J + T - 1 :]
    elif ffopt == "baseline":
        parSE = se[J + T - 1 :]
        parVar = parSE**2
        dpsi_dpar = numgrad(psi_baseline, par, T)
        for t in range(1, T):
            psiSE[t] = np.sqrt(dpsi_dpar[t - 1] ** 2 @ parVar)

    # Return
    if ffopt == "np":
        par, parSE = None, None

    return psin, psi, par, mu, psinSE, psiSE, parSE, muSE


##########################################################
# Moment constraints for optimization
##########################################################


def mom_ineq(mu):
    return np.array(
        [
            mu[1] - mu[0] ** 2,
            mu[2] - mu[1] ** 2,
            mu[3] * (mu[1] - mu[0] ** 2) - mu[2] ** 2 + 2 * mu[1] * mu[2] - mu[1] ** 3,
        ]
    )


def mom_const(T, J, nrm, ffopt):

    def c_fun(x):
        _, muM, _ = unstack(T, J, x, nrm, ffopt)
        mu = np.min(muM, axis=1)
        return mom_ineq(mu)

    return [{"type": "ineq", "fun": c_fun}]


## MODIFY mom_ineq SO IT IS APPROPRIATE FOR D>4
##########################################################

import numpy as np

##########################################################
# Helper function to calculate se of product of 2 RVs
##########################################################

# Note: Var(XY) = X^2Var(Y) + Y^2Var(X) + Var(X)Var(Y)


def se_product(X, Y, se_X, se_Y):
    var_X, var_Y = se_X**2, se_Y**2
    var_XY = (X**2) * var_Y + (Y**2) * var_X + var_X * var_Y
    return np.sqrt(var_XY)


##########################################################
# Outputs Data Moments
##########################################################


def data_moms(data, ps=None, purpose="est"):

    # Unpack data
    L, D, C = data["notice"], data["dur"], data["cens"]
    dvals = np.sort(D.unique())[:-1]
    lvals = np.sort(L.unique())
    T, J, n = len(dvals), len(lvals), len(data)

    # Initialize arrays
    h = np.zeros((T, J))
    S, S_se = np.zeros((T + 1, J)), np.zeros((T + 1, J))
    exits, surv = np.zeros((T, J)), np.zeros((T, J))
    exit_i, surv_i = np.zeros((n, T, J)), np.zeros((n, T, J))

    # Raw unadjusted moments
    for j, l in enumerate(lvals):
        for t, d in enumerate(dvals):
            exit_i[:, t, j] = (D == d) & (C == 0) & (L == l)
            surv_i[:, t, j] = (D >= d) & (L == l)

    # Inverse probability weighted moments
    if ps is not None:
        ps = ps.reshape(-1, J, 1)
        for j in range(J):
            exit_i[:, :, j] = exit_i[:, :, j] / ps[:, j]
            surv_i[:, :, j] = surv_i[:, :, j] / ps[:, j]

    # Exits, Survivors, and Hazard rate
    exits = np.mean(exit_i, axis=0)
    surv = np.mean(surv_i, axis=0)
    h = exits / surv
    h_se = np.sqrt(h * (1 - h) / np.sum(surv_i, axis=0))  # move den by 1 t

    # Survival rate
    S[0, :] = 1
    for d in range(T):
        S[d + 1, :] = S[d, :] * (1 - h[d, :])
        S_se[d + 1, :] = se_product(1 - h[d, :], S[d, :], h_se[d, :], S_se[d, :])

    # Return
    if purpose == "est":
        return exits, surv, exit_i, surv_i
    elif purpose == "output":
        return h, h_se, S, S_se
    else:
        raise ValueError("Invalid purpose argument")


##########################################################

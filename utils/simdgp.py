import numpy as np
import sympy as sp
from sympy.stats import Binomial, Beta, E
from utils.esthelpers import model_moms

##########################################################

# Default DGP:
#   x0 is constant
#   x1 is Bernoulli with p = 0.5
#   x2 is Bernoulli with p = 0.2
#   X = (1, x1, x2) (3 dimensional)
#   Notice (L) prob: exp(betaL * X) / sum(exp(betaL * X))
#   phi(X) = betaP * X
#   nu independent of L | X,
#       nu ~ Beta(a1, b1) if X1 = 1
#       nu ~ Beta(a2, b2) if X1 = 0 & X2 = 1
#       nu ~ Beta(a3, b3) if X1 = 0 & X2 = 0
#   psin defines first period probs by notice length
#  Note: with beta_L = (c, 0, 0), exogenous L

##########################################################
# Helper functions
##########################################################


# Log-Logistic hazard
def loglogistic(x, a1, a2):
    return (a2 / a1) * (x / a1) ** (a2 - 1) / (1 + (x / a1) ** a2)


# Parameter a1 for turning point and a2
def llpars(tpval, a2val):
    a1, a2, x, tp = sp.symbols("a1 a2 x tp")
    dhdx = sp.diff(loglogistic(x, a1, a2), x).simplify()
    turning_point = sp.logcombine(sp.solve(dhdx, x)[0], force=True)
    pars = sp.solve(turning_point - tp, (a1))
    parfunc = sp.lambdify((tp, a2), pars, "numpy")
    return parfunc(tpval, a2val)[0]


# Log-Logistic hazard as a function of tp and a2
def loglogistictp(x, tp, a2):
    if a2 <= 1:
        raise ValueError("No turning point for a2<=1.")
    a1 = llpars(tp, a2)
    return loglogistic(x, a1, a2)


# Weibull Hazard
def weibull(x, opt="cons"):
    if opt == "cons":
        b, k = 0.15, 1
    if opt == "inc":
        b, k = 0.2, 1.2
    if opt == "dec":
        b, k = 0.2, 0.75
    return b * k * (x ** (k - 1))


# nu parameters
def nupars(x1, x2=None):
    if x1 == 1:
        return 0.1, 0.1
    if x1 == 0 and x2 == 1:
        return 0.3, 0.5
    if x1 == 0 and x2 == 0:
        return 0.25, 0.5


##########################################################
# Define DGP function
##########################################################


def dgp(T, psin, psiopt="nm", betaL=None, betaP=None, interval=1):
    """
    psiopt options:
        'nm': non-monotonic
        'inc': increasing
        'dec': decreasing
        'cons': constant
    """

    # Initialize and errors and warnings
    J = len(psin)
    if betaP is not None:
        if sum(betaP) > 0 or sum(betaP) < -1:
            print("sum(betaP) not in (-1, 0), may lead to phi(X)>1 or <0")
    if betaL is not None:
        if len(betaL.shape) != J - 1:
            raise ValueError("betaL must have J-1 dimensions")

    # BetaL and BetaP coefficients
    betaP = np.array([1, 0, 0]) if betaP is None else np.append(1, betaP)
    betaL = np.array([1, 1, 1]) if betaL is None else np.append(1, betaL)
    betaL = np.column_stack((np.array([1, 1, 1]), betaL))

    # Implied moments of nu: E[phi(X)^k nu^k]
    X1 = Binomial("X1", 1, 0.5)
    X2 = Binomial("X2", 1, 0.2)
    X = np.array([1, X1, X2])
    phiX = betaP @ X
    pars = [nupars(x1=1), nupars(x1=0, x2=1), nupars(x1=0, x2=0)]
    nu = [Beta(f"nu{i}", pars[i][0], pars[i][1]) for i in range(3)]
    nu = nu[0] * X1 + nu[1] * (1 - X1) * X2 + nu[2] * (1 - X1) * (1 - X2)
    mu = np.zeros(T)
    for t in range(1, T + 1):
        mu[t - 1] = (
            E((phiX**t * nu**t).subs({X1: 1})) * E(X1)
            + E((phiX**t * nu**t).subs({X1: 0, X2: 1})) * E((1 - X1) * X2)
            + E((phiX**t * nu**t).subs({X1: 0, X2: 0})) * E((1 - X1) * (1 - X2))
        )

    # Probability of notice
    pL_Xnum = sp.Array([sp.exp(X @ betaL[:, j]) for j in range(J)])
    pL_X = pL_Xnum / np.sum(pL_Xnum)
    pL = np.array([E(pL_X[j]) for j in range(J)], dtype=float)

    # Specify psi and stack psin & psi into psiM
    if psiopt == "nm":
        # hazard increases till T/2 and then decreases
        psi = np.zeros(T - 1)
        psi[: T // 2] = weibull(np.arange(1, T // 2 + 1), "inc")
        psi[T // 2 :] = 1.75 * weibull(np.arange(1, T // 2), "dec")
        # tvals = np.linspace(10, 20, T-1)
        # psi = loglogistictp(tvals, tp=tvals[T//2-2], a2=8)
    else:
        psi = weibull(np.arange(1, T), psiopt)
    psiM = np.zeros((T, J))
    psiM[0, :] = psin[:J]
    psiM[1:, :] = np.repeat(psi.reshape(-1, 1), J, axis=1)
    psi = psiM @ pL

    # Collect all results in a dictionary
    quants = {
        "mu": mu,
        "pL": pL,
        "psi": psi,
        "psiM": psiM,
        "X": X,
        "phiX": phiX,
        "pL_X": pL_X,
        "T": T,
        "J": J,
        "betaL": betaL,
        "betaP": betaP,
    }

    return quants


##########################################################

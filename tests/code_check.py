##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np

# Import custom functions
from utils.simdgp import dgp
from utils.simdata import sim_data
from utils.customplot import custom_plot
from utils.datamoms import data_moms
from utils.datadesc import pred_ps
from utils.esthelpers import model_moms, unstack
from utils.estgmm import gmm, estimate
from utils.inference import std_errs, indv_moms, indv_moms_ipw

# Seed
np.random.seed(1118)

##########################################################
# DGP and data simulation for the exogenous model
##########################################################

# Note: X always impacts nu

# DGP parameters
n = 500000
T, J = 6, 2
psin = np.array([0.15, 0.4, 0.6, 0.8])[:J]
dgpopt = "endog"
if dgpopt == "exog":
    betaL = np.array([1, 1])  # X doesn't impact notice
    betaP = np.array([-0.5, 0.25])  # X impacts phi(X) and nu
elif dgpopt == "endog":
    betaL = np.array([2, 5])  # X impacts notice assignment
    betaP = np.array([-0.5, 0.25])  # X impacts phi(X) and nu
psiopt = "nm"
ffopt = "np"
ipwadj = False

# Get DGP quantities
print("Getting DGP quantities...")
dgpqnts = dgp(T + 1, psin, psiopt, betaL, betaP)
nrm = dgpqnts["mu"][0]

# Simulate data
print(f"Simulating data for n={n} with T={T}, J={J} and psiopt={psiopt}.")
data, nu, pL_X, phiX = sim_data(n, dgpqnts)
nL = data["notice"].value_counts().sort_index().values

# If using IPW, get the propensity scores
ps = None
if ipwadj:
    ps, coefs = pred_ps(data)
adj = "ipw" if ipwadj else "none"
momsf = indv_moms_ipw if ipwadj else indv_moms

# Verify model moments match data moments
print("Comparing model and data moments...")
h_data = data_moms(data, ps, purpose="output")[0]
h_model = model_moms(dgpqnts["psiM"], dgpqnts["mu"])
custom_plot([h_data[:, 0], h_model[:, 0]], legendlabs=["Data", "Model"])

# Estimate using GMM and unstack and do inference
print("Estimating the model on simulated data...")
thta_hat = gmm(data, nrm, ffopt="np", ps=ps)[0]
psiM, mu = unstack(T, J, thta_hat, nrm, ffopt)
psi_hat = psiM @ nL / n
thta_hat = np.append(thta_hat, coefs) if ipwadj else thta_hat
se = std_errs(thta_hat, data, nrm, ffopt, MomsFunc=momsf)
custom_plot([psi_hat, dgpqnts["psi"]], legendlabs=["Estimates", "True"])

# Use the direct function
r = estimate(data, nrm, ffopt, adj)
custom_plot(
    [r["psi"], dgpqnts["psi"]], [r["psiSE"], None], legendlabs=["Estimates", "True"]
)

##########################################################

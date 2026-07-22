# ============================================================================
# Estimate the model at the eta bounds from 07a_bounds_value.py, using the pure
# scale-shift parameterization (kappa0_mult).
#
# What kappa0_mult implements
# ---------------------------
# Setting `ffopt["kappa0_mult"] = k0` invokes `meanshift_mult` in unstack, which
# imposes the scale-shift model:
#
#     nu_{l'} = k0 * nu_{l}    =>    mu_{l',d} = k0**d * mu_{l,d}
#
# Combined with `gamma = ones` (no structural-hazard wedge
# for d >= 2), this is the pure-kappa interpretation of the composite wedge eta
# from 07a: gamma=1, kappa=eta. Under fixed eta this convention pins down
# psi_{l',1} = g(1|l') / (eta * mu_1); the d >= 2 fit (psi(d), mu_d, J-stat) is
# identical to any other split of eta between gamma and kappa, since the
# moment equations for d >= 2 depend only on the composite gamma*kappa = eta.
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.datamoms import data_moms
from utils.estgmm import gmm
from utils.esthelpers import model_moms, mom_const, unstack
from utils.customplot import set_plot_aes
from utils.config import DATA_DIR, QUANTS_DIR, OUTPUT_DIR, Colors

set_plot_aes()

######################################################
# Initialize
######################################################

# Load data and estimated ps
sample = pd.read_csv(f"{DATA_DIR}/sample.csv")
X = pd.read_csv(f"{DATA_DIR}/control_vars.csv")
ps = np.load(f"{QUANTS_DIR}/ps.npy")
data_for_est = pd.concat([sample[["notice", "dur", "cens"]], X], axis=1)

# Dimensions
nL = sample["notice"].value_counts().sort_index().values
pi = nL / nL.sum()
T = len(sample["dur"].unique()) - 1
J = len(sample["notice"].unique())

# Colors
red = Colors().RED
black = Colors().BLACK
grey = Colors().GREY

# Load ub and lb on eta
eta_lb, eta_ub = np.load(f"{QUANTS_DIR}/eta_bounds.npy")

######################################################
# Estimate model at gamma = 1, kappa = 0, Baseline psi
######################################################

opt = "baseline"
nrm = 1
gamma = np.ones((T - 1, J))
use_const = False
ps = None

# Current estimate
ffopt = {"opt": opt, "gamma": gamma, "kappa0_mult": 1}
const = mom_const(T, J, nrm, ffopt) if use_const else None
thta_hat, Jstat = gmm(data_for_est, nrm, ffopt, const=const, ps=ps)
psiM, muM, _ = unstack(T, J, thta_hat, nrm, ffopt)
psi_base = np.sum(pi * psiM, axis=1)
print(Jstat)

# Upper bound
ffopt = {"opt": opt, "gamma": gamma, "kappa0_mult": eta_lb}
const = mom_const(T, J, nrm, ffopt) if use_const else None
thta_hat, Jstat = gmm(data_for_est, nrm, ffopt, const=const, ps=ps)
psiM, muM, _ = unstack(T, J, thta_hat, nrm, ffopt)
psi_ub = np.sum(pi * psiM, axis=1)
print(Jstat)

# Fitted h
h = model_moms(psiM, muM, ffopt)
h_data, _, S, _ = data_moms(sample, ps=ps, purpose="output")
g_data = h_data * S[:-1, :]
1 / (g_data[0, 1] + g_data[1, 1])

# Lower bound
gamma = 1.2 * gamma
ffopt = {"opt": opt, "gamma": gamma, "kappa0_mult": eta_ub}
const = mom_const(T, J, nrm, ffopt) if use_const else None
thta_hat, Jstat = gmm(data_for_est, nrm, ffopt, const=const, ps=ps)
psiM, muM, _ = unstack(T, J, thta_hat, nrm, ffopt)
psi_lb = np.sum(pi * psiM, axis=1)
print(Jstat)


######################################################
# Plot psi vs baseline and observed hazard
######################################################

# r = np.load(f"{QUANTS_DIR}/baseline_ests.npy", allow_pickle=True).item()
h_avg = np.load(f"{QUANTS_DIR}/h_avg_ipw.npy")
xticks = ["0-12", "12-24", "24-36", "36-48"]
xlabel = "Weeks since unemployed"
psi_base = psi_base / psi_base[0] * h_avg[0]
psi_ub = psi_ub / psi_ub[0] * h_avg[0]
psi_lb = psi_lb / psi_lb[0] * h_avg[0]


plt.figure(figsize=[3.5, 3])
plt.plot(psi_base, color=black, linestyle="-", label="Baseline")
plt.plot(psi_ub, color=grey, linestyle="-", linewidth=1, alpha=0.7)
plt.plot(psi_lb, color=grey, linestyle="-", linewidth=1, alpha=0.7)
plt.plot(h_avg, color=black, linestyle="--", label="Empirical")
plt.xticks(range(T), xticks)
plt.xlabel(xlabel)
plt.fill_between(range(T), psi_lb, psi_ub, color=grey, alpha=0.3, label="Bounds")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_sensitivity.pdf", dpi=300, format="pdf")

######################################################

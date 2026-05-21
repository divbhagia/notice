# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom functions
from utils.estgmm import gmm
from utils.esthelpers import unstack
from utils.customplot import set_plot_aes
from utils.config import DATA_DIR, QUANTS_DIR, Colors, OUTPUT_DIR
from utils.config import RUN_EXT_AGAIN

set_plot_aes()

######################################################
# Initialize
######################################################

# Load data and estimated ps for estimation
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
blue = Colors().BLUE
grey = Colors().GREY

# Parameters
nrm = 1
opt = "baseline"
numpars = 2 * (T - 1) + J if opt == "np" else 2 + (T - 1) + J
gdist = 0.01
kdist = 0.01
glims = [0.90, 1.1]
klims = [-0.1, 0.1]
K1 = int((glims[1] - glims[0]) / gdist)
K2 = int((klims[1] - klims[0]) / kdist)
gpars = np.linspace(glims[0], glims[1], K1)
kpars = np.linspace(klims[0], klims[1], K2)
print(f"K1 = {K1}, K2 = {K2}")

# Add 1 to gpar and 0 to kpar if not included
if 1 not in gpars:
    gpars = np.insert(gpars, np.where(gpars > 1)[0][0], 1)
    K1 += 1
if 0 not in kpars:
    kpars = np.insert(kpars, np.where(kpars > 0)[0][0], 0)
    K2 += 1

######################################################
# Estimate model with different values of parameters
######################################################

# Initialize arrays
gamma = np.ones((T - 1, J))
kappa = np.zeros((T, J))
psis = np.zeros((K1, K2, T))
thtas = np.zeros((K1, K2, numpars))
Jstats = np.zeros((K1, K2))

# Estimate model for different values of gamma and kappa1
if RUN_EXT_AGAIN:
    ffopt = {"opt": opt, "gamma": gamma, "kappa0": kpars[0]}
    for k in range(K1):
        for l in range(K2):
            ffopt["gamma"][:, 1] = gpars[k]
            ffopt["kappa0"] = kpars[l]
            thtas[k, l, :], Jstats[k, l] = gmm(data_for_est, nrm, ffopt, ps)
            psiM, *_ = unstack(T, J, thtas[k, l], nrm, ffopt)
            psis[k, l, :] = np.sum(pi * psiM, axis=1)

    np.save(f"{QUANTS_DIR}/ext_estimates.npy", psis)
    np.save(f"{QUANTS_DIR}/ext_jstats.npy", Jstats)
else:
    psis = np.load(f"{QUANTS_DIR}/ext_estimates.npy")
    Jstats = np.load(f"{QUANTS_DIR}/ext_jstats.npy")

# Return indexes when gamma = 1 and kappa = 0
gidx1 = np.where(gpars == 1)[0][0]
kidx0 = np.where(kpars == 0)[0][0]

# Find index of the best model
gbest = np.argmin(Jstats[:, kidx0])  # kappa = 0
kbest = np.argmin(Jstats[gidx1, :])  # gamma = 1
best = np.unravel_index(np.argmin(Jstats), Jstats.shape)

######################################################
# Plots
######################################################

# Distance for different values of gamma when kappa = 0
plt.figure(figsize=[3, 2.5])
plt.plot(gpars, Jstats[:, kidx0], color=red, linestyle="-")
plt.axvline(gpars[gbest], color=black, linestyle="--", linewidth=0.75)
plt.annotate(f"$\\gamma^*={gpars[gbest]:.2f}$", (gpars[gbest] + 0.5 * gdist, 0.6))
plt.xlabel("$\\gamma$")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_extgen_strhazA.pdf", dpi=300, format="pdf")

# Distance for different values of kappa when gamma = 1
plt.figure(figsize=[3, 2.5])
plt.plot(kpars, Jstats[gidx1, :], color=red, linestyle="-")
plt.axvline(kpars[kbest], color=black, linestyle="--", linewidth=0.75)
plt.annotate(f"$\\kappa_1^*={kpars[kbest]:.2f}$", (kpars[kbest] + 0.5 * kdist, 6))
plt.xlabel("$\\kappa_1$")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_extgen_hetA.pdf", dpi=300, format="pdf")

# Plot for both varying
text1 = f"$\\gamma^*={gpars[best[0]]:.2f}$\n$"
text2 = f"\\kappa_1^*={kpars[best[1]]:.2f}$"
plt.figure(figsize=[3, 2.5])
plt.contourf(gpars, kpars, Jstats.T, 50, cmap="ocean_r", alpha=1)
plt.plot(gpars[best[0]], kpars[best[1]], "ko", markersize=3)
plt.annotate(text1 + text2, (1.015, -0.04), fontsize=7)
plt.axhline(0, color=black, linestyle="--", linewidth=0.5)
plt.axvline(1, color=black, linestyle="--", linewidth=0.5)
plt.colorbar()
plt.xlabel("$\\gamma$")
plt.ylabel("$\\kappa_1$")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_extgen_bothA.pdf", dpi=300, format="pdf")
# Note: for plt.counterf, z[i, j] = f(x[j], y[i]), hence Jstats.T

######################################################
# See how structural hazard changes with parameters
######################################################

# Import baseline estimates and h_avg
r = np.load(f"{QUANTS_DIR}/baseline_ests.npy", allow_pickle=True).item()
h_avg = np.load(f"{QUANTS_DIR}/h_avg_ipw.npy")

# xtick-labels
xticks = ["0-12", "12-24", "24-36", "36-48"]
xlabel = "Weeks since unemployed"

# Plot X best models
ord_gidx = np.argsort(Jstats[:, kidx0])
ord_kidx = np.argsort(Jstats[gidx1, :])
ord_idx = np.unravel_index(np.argsort(Jstats, axis=None), Jstats.shape)
ord_idx = np.column_stack((ord_idx[0], ord_idx[1]))
plot_range = 10
plot_range_both = 25

# Plot for different values of gamma
plt.figure(figsize=[3, 2.5])
print(f"Plotting {plot_range}/{K1} best models for gen_strhaz")
for k in range(plot_range):
    plt.plot(
        psis[ord_gidx[k], kidx0, :],
        # label=f'$\\gamma={gpars[ord_gidx[k]]:.2f}$',
        alpha=0.35,
        color=grey,
    )
plt.plot(r["psi"], color=black, linestyle="-", label="Baseline")
plt.plot(h_avg, color=black, linestyle="--", label="Observed")
plt.xticks(range(T), xticks)
plt.xlabel(xlabel)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_extgen_strhazB.pdf", dpi=300, format="pdf")

# Plot for different values of kappa
plt.figure(figsize=[3, 2.5])
print(f"Plotting {plot_range} best models for gen_het")
for k in range(plot_range):
    plt.plot(
        psis[gidx1, ord_kidx[k], :],
        # label=f'$\\kappa_1={kpars[ord_gidx[k]]:.2f}$',
        alpha=0.35,
        color=grey,
    )
plt.plot(h_avg, color=black, linestyle="--", label="Observed")
plt.plot(r["psi"], color=black, linestyle="-", label="Baseline")
plt.xticks(range(T), xticks)
plt.xlabel(xlabel)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_extgen_hetB.pdf", dpi=300, format="pdf")

# Vary both gamma and kappa
plt.figure(figsize=[3, 2.5])
print(f"Plotting {plot_range_both} best models for both")
for k in range(plot_range_both):
    plt.plot(psis[ord_idx[k, 0], ord_idx[k, 1], :], alpha=0.35, color="grey")
plt.plot(r["psi"], color=black, linestyle="-", label="Baseline")
plt.plot(h_avg, color=black, linestyle="--", label="Observed")
plt.xticks(range(T), xticks)
plt.xlabel(xlabel)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_extgen_bothB.pdf", dpi=300, format="pdf")

######################################################

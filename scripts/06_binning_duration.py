import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils.simdgp import dgp
from utils.esthelpers import model_moms, unstack, numgrad
from utils.customplot import set_plot_aes
from utils.config import Colors, TEST_QUANTS_DIR, SIMBIN_AGAIN, OUTPUT_DIR

np.random.seed(1118)
set_plot_aes()

######################################################
# Simplified functions to estimate the model using h
######################################################


def objfun(thta, h_data, nrm, ffopt="np"):
    T, J = h_data.shape
    h_model = model_moms(*unstack(T, J, thta, nrm, ffopt)[:2])
    m = (h_data - h_model).reshape(T * J, 1).flatten()
    obj = m.T @ m
    return obj


def numgrad_wrapper(x, *args):
    return numgrad(objfun, x, *args).flatten()


def gmm(h_data, nrm, ffopt="np"):
    T, J = h_data.shape
    thta0 = 0.25 * np.ones(2 * (T - 1) + J)
    opts = {"disp": False, "maxiter": 100000}
    results = minimize(
        objfun,
        thta0,
        method="BFGS",
        tol=1e-32,
        args=(h_data, nrm, ffopt),
        options=opts,
        jac=numgrad_wrapper,
    )
    psiM, _ = unstack(T, J, results.x, nrm, ffopt)
    return psiM


######################################################
# Functions to get binned quantities
######################################################


def bin_quants(q, interval):
    S = np.insert(np.cumprod((1 - q), axis=0), 0, 1, axis=0)
    if q.ndim == 1:
        S_bin = S[::interval]
        h_bin = (S_bin[:-1] - S_bin[1:]) / S_bin[:-1]
    else:
        S_bin = S[::interval, :]
        h_bin = (S_bin[:-1, :] - S_bin[1:, :]) / S_bin[:-1]
    return h_bin


######################################################
# Initialize parameters
######################################################

# DGP parameters
T, J = 12, 2
psin = np.array([0.1, 0.2, 0.3, 0.5])[:J]

# Estimation parameters
psiopts = ["inc", "dec", "cons", "nm"]
intervals = [1, 2, 3, 4]
keys = [f"{psiopt}_{interval}" for psiopt in psiopts for interval in intervals]

# Initialize vectors
psi_hat = {key: np.zeros((T)) for key in keys}
psi_bin = {key: np.zeros((T)) for key in keys}
h_avg = {key: np.zeros((T)) for key in keys}
h_avg_bin = {key: np.zeros((T)) for key in keys}

######################################################
# Function to estimate for a given key ######################################################


def estimate(key):

    # Get parameters
    psiopt, interval = key.split("_")
    interval = int(interval)

    # Get DGP quantities
    dgpqnts = dgp(T, psin, psiopt)
    psiM, mu, psi = dgpqnts["psiM"], dgpqnts["mu"], dgpqnts["psi"]
    nrm = mu[0]
    h = model_moms(psiM, mu)
    h_avg = h @ dgpqnts["pL"]
    h_avg_bin = bin_quants(h_avg, interval)
    h_avg_bin = np.repeat(h_avg_bin, interval)

    # Calculate true binned psi
    psi = psi * mu[0]
    psi_bin = bin_quants(psi, interval)
    psi_bin = np.repeat(psi_bin, interval)

    # Estimate using binned hazard
    psiM_hat = gmm(bin_quants(h, interval), nrm, ffopt="np")
    psi_hat = psiM_hat @ dgpqnts["pL"]
    psi_hat = np.repeat(psi_hat, interval)

    return psi_hat, psi_bin, h_avg_bin, h_avg


######################################################
# Estimate model for different psiopts and intervals
######################################################

if SIMBIN_AGAIN:
    for k in keys:
        psi_hat[k], psi_bin[k], h_avg_bin[k], h_avg[k] = estimate(k)
    r = {"psi_hat": psi_hat, "psi_bin": psi_bin, "h_avg_bin": h_avg_bin, "h_avg": h_avg}
    np.save(f"{TEST_QUANTS_DIR}/sim_binning.npy", r)
else:
    r = np.load(f"{TEST_QUANTS_DIR}/sim_binning.npy", allow_pickle=True).item()
    psi_hat, psi_bin, h_avg_bin, h_avg = (
        r["psi_hat"],
        r["psi_bin"],
        r["h_avg_bin"],
        r["h_avg"],
    )

######################################################
# Plot the results
######################################################

psi_hat_nrm = {key: psi_hat[key] / psi_hat[key][0] for key in keys}
psi_bin_nrm = {key: psi_bin[key] / psi_bin[key][0] for key in keys}
h_avg_bin_nrm = {key: h_avg_bin[key] / h_avg_bin[key][0] for key in keys}

# Plot Aesthetics
ylims = {
    "nm": (-0.5, 2.75),
    "inc": (-0.5, 3.25),
    "dec": (0, 1.25),
    "cons": (0.25, 1.25),
}
ydist = {"nm": 1, "inc": 1, "dec": 0.5, "cons": 0.5}
labels = ["Estimate", "True (Cumulative)", "Observed"]
xticklabs = np.arange(2, T + 1, 2)
psioptlabs = {
    "inc": "Increasing Hazard",
    "dec": "Decreasing Hazard",
    "cons": "Constant Hazard",
    "nm": "Non-Monotonic Hazard",
}
alph = ["A", "B", "C", "D"]

# Plot hazards
fig = plt.figure(constrained_layout=True, figsize=(6, 6.5))
subfigs = fig.subfigures(nrows=4, ncols=1)
for row, subfig in enumerate(subfigs):
    subfig.suptitle(f"\nPanel {alph[row]}: {psioptlabs[psiopts[row]]}", fontsize=10)
    axs = subfig.subplots(nrows=1, ncols=4)
    for col, ax in enumerate(axs):
        key = f"{psiopts[row]}_{intervals[col]}"
        ax.set_title(f"Bin Size: {intervals[col]}", fontsize=9, pad=4)
        ax.plot(psi_hat_nrm[key], color=Colors.DGREY, linestyle="-", linewidth=1.5)
        ax.plot(psi_bin_nrm[key], color=Colors.RED, linestyle=":", linewidth=2.5)
        ax.plot(h_avg_bin_nrm[key], color=Colors.BLACK, linestyle="--")
        ax.set_ylim(ylims[psiopts[row]])
        ax.set_xticks(np.arange(1, T + 1, 2))
        ax.set_xticklabels(xticklabs)
        ax.set_yticks(np.arange(0, ylims[psiopts[row]][1], ydist[psiopts[row]]))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
# Add legend
fig.legend(
    labels,
    loc="lower center",
    ncol=3,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.075),
    handlelength=3,
    handletextpad=0.5,
    columnspacing=1.5,
)
plt.savefig(
    f"{OUTPUT_DIR}/fig_sim_binningA.pdf", bbox_inches="tight", dpi=300, format="pdf"
)

# Plot Average type
ylims = {"nm": (0, 1.25), "inc": (0, 1.25), "dec": (0.25, 1.25), "cons": (0.25, 1.25)}
fig = plt.figure(constrained_layout=True, figsize=(6, 6.5))
subfigs = fig.subfigures(nrows=4, ncols=1)
alph = ["A", "B", "C", "D"]
for row, subfig in enumerate(subfigs):
    subfig.suptitle(f"\nPanel {alph[row]}: {psioptlabs[psiopts[row]]}", fontsize=10)
    axs = subfig.subplots(nrows=1, ncols=4)
    for col, ax in enumerate(axs):
        key = f"{psiopts[row]}_{intervals[col]}"
        ax.set_title(f"Bin Size: {intervals[col]}", fontsize=9, pad=4)
        ax.plot(h_avg_bin_nrm[key] / psi_hat_nrm[key], color=Colors.BLACK)
        ax.set_ylim(ylims[psiopts[row]])
        ax.set_yticks(np.arange(0, 1.25, 0.5))
        ax.set_xticks(np.arange(1, T + 1, 2))
        ax.set_xticklabels(xticklabs)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.savefig(
    f"{OUTPUT_DIR}/fig_sim_binningB.pdf", bbox_inches="tight", dpi=300, format="pdf"
)

######################################################

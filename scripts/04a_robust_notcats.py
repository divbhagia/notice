##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns
from tabulate import tabulate

# Import custom functions and parameters
from utils.datadesc import pred_ps
from utils.config import DATA_DIR, Colors, OUTPUT_DIR, QUANTS_DIR
from utils.config import print_bold
from utils.datadesc import sum_tab, find_sig_rows

# Import custom functions
from utils.customplot import custom_plot, set_plot_aes
from utils.datamoms import data_moms
from utils.estgmm import gmm
from utils.inference import indv_moms, std_errs
from utils.esthelpers import unstack_all

##########################################################
# Initialize
##########################################################

# Load data
sample = pd.read_csv(f"{DATA_DIR}/sample_all.csv")
X = pd.read_csv(f"{DATA_DIR}/controls_all.csv")
sample["notice"] = sample["dwnotice"]

# Plot summary statistics for full sample
varlist = [
    "age",
    "female",
    "married",
    "black",
    "col",
    "pc",
    "union",
    "in_metro",
    "dwyears",
    "lnearnl",
]
table = sum_tab(sample, varlist, "notice", labels=varlist, se=False, diff=False)
print(
    tabulate(
        table,
        headers=["Variable", "1", "2", "3", "4"],
    )
)

# Table convert to numbers
tab = np.array(table[:, 1:], dtype=float)
diff = tab[:, -1].reshape(-1, 1) - tab
diff_str = np.array([[""] * (tab.shape[1])] * tab.shape[0], dtype="<U8")
# turn diff to string with 2 digits (keep structure)
for row in range(diff.shape[0]):
    for col in range(diff.shape[1]):
        diff_str[row, col] = f"{diff[row, col]:.2f}"
# Add varlist as row names for diff_str
diff_str = np.insert(diff_str[:-1, :], 0, varlist, axis=1)
print(
    tabulate(
        diff_str,
        headers=["Variable", "1", "2", "3", "4"],
    )
)

# Prepare data for estimation
data_all = pd.concat([sample[["notice", "dur", "cens"]], X], axis=1)
data_134 = data_all[data_all["notice"] != 2].copy()
data_234 = data_all[data_all["notice"] != 1].copy()
data_23c4 = data_234.copy()
data_23c4["notice"] = np.where(data_23c4["notice"] == 2, 3, data_23c4["notice"])
data_34 = data_234[data_234["notice"] != 2].copy()
datasets = {
    "data_134": data_134,
    "data_234": data_234,
    "data_23c4": data_23c4,
    "data_34": data_34,
}

# Prepare arrays to store results
Jstats, psi_hats, se_hats, h_avg, h = {}, {}, {}, {}, {}
estdata, pvals = {}, {}

# Options
nrm = 1
ffopt = "baseline"
adj = "ipw"
colors = [Colors.BLUE, Colors.RED, Colors.BLACK]
print_examine = True

# Load baseline estimates for comparison
baseline_ests = np.load(f"{QUANTS_DIR}/baseline_ests.npy", allow_pickle=True).item()
base_psi = baseline_ests["psi"]

##########################################################
# Estimate models
##########################################################

# Loop over datasets
for name, data in datasets.items():

    # Initialize
    T = len(data["dur"].unique()) - 1
    notvals = np.sort(data["notice"].unique())
    J = len(notvals)
    dof = T * J - (1 + T + J)

    # Generate propensity scores
    ps = pred_ps(data)[0]

    # Check overlap (optional)
    if print_examine:
        plt.figure(figsize=(5.5, 1.75))
        for k in range(len(notvals)):
            plt.subplot(1, 3, k + 1)
            for j in range(len(notvals)):
                sns.kdeplot(
                    ps[(data["notice"] == notvals[j])][:, k],
                    color=colors[j],
                    fill=True,
                    alpha=0.1,
                    edgecolor="black",
                    label=f"{int(notvals[j])}",
                )
            plt.title(f"$Pr(L={int(notvals[k])})$")
        plt.tight_layout()
        plt.legend(ncol=3, bbox_to_anchor=(0, -0.25))

    # Check extreme propensity scores
    ll, ul = 0.1, 0.9
    keep = 0
    for j in range(1, J):
        keep += (ps[:, j] > ll) & (ps[:, j] < ul)
    keep = keep == max(keep)
    print("")
    print_bold(f"Keep observations with {ll}<PS<{ul}: {sum(keep)}/{len(keep)}")
    if print_examine:
        data["extreme"] = 1 - keep
        exts = data.groupby("notice")["extreme"].agg(["sum", "count", "mean"])
        print(tabulate(exts, headers=["Notice", "Extreme", "Total", "Share"]))
        print("")

    # Add weights to data for summary statistics
    for j in range(J):
        datasets[name].loc[datasets[name]["notice"] == notvals[j], "wt"] = (
            1 / ps[datasets[name]["notice"] == notvals[j], j]
        )

    # Trim data
    ps = ps[keep]
    data = data.iloc[keep]
    datasets[name] = datasets[name].iloc[keep].copy()

    # Some data quantities
    nL = data["notice"].value_counts().sort_index().values
    h[name] = data_moms(data, ps, purpose="output")[0]
    h_avg[name] = h[name] @ nL / nL.sum()

    # Estimate model
    thta_hat, Jstats[name] = gmm(data, nrm, ffopt, ps)
    se = std_errs(thta_hat, data, nrm, ffopt, MomsFunc=indv_moms)
    psin, psi, par, mu, psinSE, psiSE, parSE, muSE = unstack_all(
        T, J, nL, thta_hat, se, nrm, ffopt
    )
    psi_hats[name] = psi
    se_hats[name] = psiSE

    # p-values for J-statistics
    pvals[name] = 1 - chi2.cdf(Jstats[name], dof)

##########################################################
# Plots and output
##########################################################

# Plot aesthetics
labs = ["Structural", "Baseline", "Observed"]
text = {key: f"J-Stat: {Jstats[key]:.2f} (p={pvals[key]:.2f})" for key in Jstats.keys()}
filenames = [f"{OUTPUT_DIR}/fig_notcats{x}.pdf" for x in ["A", "B", "C"]]
xticklabs = ["0-12", "12-24", "24-36", "36-48"]
ylab = "Hazard rate"
xlab = "Weeks since unemployed"

# Plots for the paper
psi_hats.pop("data_34")
set_plot_aes()
for i, key in enumerate(psi_hats.keys()):
    series = [psi_hats[key], base_psi, h_avg[key]]
    se = [se_hats[key], None, None]
    custom_plot(
        series,
        se,
        xlab=xlab,
        ylab=ylab,
        legendlabs=labs,
        xticklabs=xticklabs,
        colors=colors,
        ydist=0.2,
        legendpos="lower left",
        ylims=[-0.075, 0.9],
    )
    plt.axvline(x=2, color="black", linestyle=":", linewidth=1.25, alpha=0.75)
    plt.text(0, 0.725, text[key], color=Colors.BLACK)
    plt.savefig(filenames[i], dpi=300, format="pdf")

##########################################################
# Additional examination (optional)
##########################################################

if print_examine:

    # Data for examination
    linestyles = ["-.", "-", "--"]
    colors
    plt.figure(figsize=(10, 3))
    for i, name in enumerate(datasets.keys()):
        plt.subplot(1, 4, i + 1)
        notvals = np.sort(datasets[name]["notice"].unique())
        clrs = colors[-len(notvals) :]
        lstls = linestyles[-len(notvals) :]
        series = [h[name][:, j] for j in range(len(notvals))]
        custom_plot(
            series,
            subplot=True,
            colors=clrs,
            linestyles=lstls,
            legendlabs=[f"{int(x)}" for x in notvals],
        )

    # Summary statistics for examination
    for name in datasets.keys():
        notvars = [
            "notice",
            "dwnotice",
            "wt",
            "cens",
            "dur",
            "extreme",
            "age2",
            "dwyears2",
        ]
        varlist = datasets[name].columns.difference(notvars)
        table1 = sum_tab(datasets[name], varlist, "notice", varlist, se=False)
        table2 = sum_tab(datasets[name], varlist, "notice", varlist, se=False, wts="wt")
        print("\n" + name.upper() + " Unweighted")
        print(tabulate(table1[find_sig_rows(table1)], tablefmt="orgtbl"))
        print("\n" + name.upper() + " Weighted")
        print(tabulate(table2[find_sig_rows(table2)], tablefmt="orgtbl"))


##########################################################

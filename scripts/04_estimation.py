##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Import custom functions
from utils.customplot import custom_plot
from utils.estgmm import estimate
from utils.datamoms import data_moms

# Import parameters
from utils.config import DATA_DIR, QUANTS_DIR, OUTPUT_DIR
from utils.config import Colors, CRIT

# Set colors
red = Colors.RED
blue = Colors.BLUE
black = Colors.BLACK

##########################################################
# Initialize
##########################################################

# Load data and estimated quantities
sample = pd.read_csv(f"{DATA_DIR}/sample.csv")
X = pd.read_csv(f"{DATA_DIR}/control_vars.csv")
ps = np.load(f"{QUANTS_DIR}/ps.npy")
coefs = np.load(f"{QUANTS_DIR}/coefs.npy")

# Data for estimation
data_for_est = pd.concat([sample[["notice", "dur", "cens"]], X], axis=1)
nL = sample["notice"].value_counts().sort_index().values
T, J = len(sample["dur"].unique()) - 1, len(sample["notice"].unique())

# Data moments for comparison
h, *_ = data_moms(data_for_est, ps, purpose="output")
h_avg = h @ nL / nL.sum()
np.save(f"{QUANTS_DIR}/h_avg_ipw.npy", h_avg)

# Baseline estimates
nrm = 1
r = estimate(data_for_est, nrm, ffopt="baseline", adj="ipw")
np.save(f"{QUANTS_DIR}/baseline_ests.npy", r)

# Verify ps & coefs same as saved
if r["ps"] is not None and r["coefs"] is not None:
    print("PS match:", np.allclose(r["ps"], ps))
    print("Coefs match:", np.allclose(r["coefs"], coefs))
    if not np.allclose(r["ps"], ps) or not np.allclose(r["coefs"], coefs):
        raise ValueError("Estimates do not match saved quantities")

# Print quantities
h_avg_dec_1to2 = (h_avg[1] - h_avg[0]) / h_avg[0]
print(f"Obs hazard change 1 to 2: {100*h_avg_dec_1to2:.1f}\n")
psi_dec_1to2 = (r["psi"][1] - r["psi"][0]) / r["psi"][0]
print(f"Structural hazard change 1 to 2: {100*psi_dec_1to2:.1f}\n")
psi_dec_2to3 = (r["psi"][2] - r["psi"][1]) / r["psi"][1]
print(f"Structural hazard change 2 to 3: {100*psi_dec_2to3:.1f}\n")

##########################################################
# Baseline estimates table
##########################################################

############### Panel A

# Specify parameters
parmts = ["$\\psi_S(1)$", "$\\psi_L(1)$", "$\\alpha_1$", "$\\alpha_2$"]
expln = [
    "Structural hazard 0-12 weeks: Short notice",
    "Structural hazard 0-12 weeks: Long notice",
    "Scale parameter for $\\psi(d)$",
    "Shape parameter for $\\psi(d)$",
]
xH = np.concatenate([r["psin"], r["par"]])
SE = np.concatenate([r["psinSE"], r["parSE"]])

# Open the file and write
f = open(f"{OUTPUT_DIR}/tab_baseline_estsA.tex", "w")
for i in range(len(xH)):
    f.write(f"{parmts[i]} & {expln[i]} & {xH[i]:.2f} & {SE[i]:.2f} \\\\ \n")
f.close()

############### Panel B

# Specify parameters
parmts = ["$\\bar{\\psi}(1)$", "$\\psi(2)$", "$\\psi(3)$", "$\\psi(4)$", "$c$"]
expln = [
    "Structural hazard: 0-12 weeks",
    "Structural hazard: 12-24 weeks",
    "Structural hazard: 24-36 weeks",
    "Structural hazard: 36-48 weeks",
    "Last one",
]
xH, SE = r["psi"], r["psiSE"]

# Open the file and write
f = open(f"{OUTPUT_DIR}/tab_baseline_estsB.tex", "w")
for i in range(len(xH)):
    f.write(f"{parmts[i]} & {expln[i]} & {xH[i]:.2f} & {SE[i]:.2f} \\\\ \n")
f.close()

############### Panel C
alpha, df = 0.05, 1
critJ = chi2.ppf(1 - alpha, df)
alpha = "{" + str(alpha) + "}"
line = f" & Test statistic: {r['Jstat']:.2f} & Critical value, $df={df}, \\chi_{alpha}^2$: {critJ:.2f} \\\\\n"
f = open(f"{OUTPUT_DIR}/tab_baseline_estsC.tex", "w")
f.write(line)
f.close()

##########################################################
# Plot baseline estimates
##########################################################

# Parameters for plot
xticklabs = ["0-12", "12-24", "24-36", "36-48"]

# Plot structural hazard
# h_avg = psi[0] * h_avg / h_avg[0]
series = [r["psi"], h_avg]
se = [r["psiSE"], None]
custom_plot(
    series,
    se,
    xlab="Weeks since unemployed",
    ylab="Hazard",
    legendlabs=["Structural", "Observed"],
    xticklabs=xticklabs,
    colors=[blue, black],
    linestyles=["-", ":"],
    ylims=[-0.05, 0.95],
    legendpos="lower left",
    ydist=0.2,
)
plt.axvline(x=2, color="black", linestyle=":", linewidth=1.25, alpha=0.75)
plt.text(0.69, 0.85, "UI Exhaustion")
plt.savefig(f"{OUTPUT_DIR}/fig_baseline_estsA.pdf", dpi=300, format="pdf")

# Plot average type
psiM = np.array([np.append(r["psin"][j], r["psi"][1:]) for j in range(J)]).T
avg_type = h / psiM
series = [avg_type[:, j] for j in range(J)]
custom_plot(
    series,
    xticklabs=xticklabs,
    ylab="Average Type",
    ydist=0.2,
    xlab="Weeks since unemployed",
    legendlabs=["Short notice", "Long notice"],
    legendpos="lower left",
)
plt.savefig(f"{OUTPUT_DIR}/fig_baseline_estsB.pdf", dpi=300, format="pdf")

##########################################################
# Appendix figure: unweighted estimates
##########################################################

h, *_ = data_moms(data_for_est, purpose="output")
h_avg = h @ nL / nL.sum()
runwtd = estimate(data_for_est, nrm, ffopt="baseline", adj="none")
series = [runwtd["psi"], r["psi"], h_avg]
se = [runwtd["psiSE"], None, None]
custom_plot(
    series,
    se,
    xlab="Weeks since unemployed",
    ylab="Hazard",
    legendlabs=["Unweighted", "Weighted", "Observed"],
    xticklabs=xticklabs,
    colors=[blue, red, black],
    ylims=[-0.05, 0.95],
    legendpos="lower left",
    ydist=0.2,
)
plt.axvline(x=2, color="black", linestyle=":", linewidth=1.25, alpha=0.75)
plt.text(0.69, 0.85, "UI Exhaustion")
plt.savefig(f"{OUTPUT_DIR}/fig_robust_unwtd.pdf", dpi=300, format="pdf")

##########################################################
# Appendix figure: different functional forms
##########################################################

# Non-parametric estimates
rnp = estimate(data_for_est, nrm, ffopt="np", adj="ipw")

# Parameters for plot
xticklabs = ["0-12", "12-24", "24-36", "36-48"]

# Plot structural hazard
xax = np.arange(len(h_avg))
plt.figure(figsize=(4.5, 3))
plt.errorbar(
    xax, r["psi"], yerr=CRIT * r["psiSE"], color=blue, capsize=3, fmt="o", markersize=4
)
plt.errorbar(
    xax + 0.075,
    rnp["psi"],
    yerr=CRIT * rnp["psiSE"],
    color=black,
    capsize=3,
    fmt="x",
    markersize=5,
)
plt.xticks(xax, xticklabs)
plt.tight_layout(pad=2)
plt.xlabel("Weeks since unemployed")
plt.ylabel("Hazard")
plt.legend(["Log-Logistic (Baseline)", "Non-parametric"], loc="lower left")
plt.savefig(f"{OUTPUT_DIR}/fig_robust_np.pdf", dpi=300, format="pdf")

##########################################################
# Appendix figure: estimates with 9 week intervals
##########################################################

# Data for estimation
data_for_est = pd.concat([sample[["notice", "dur", "cens"]], X], axis=1)
data_for_est["dur"] = sample["dur_9week"]
T = len(sample["dur_9week"].unique()) - 1

# Data moments for comparison
h, *_ = data_moms(data_for_est, ps, purpose="output")
h_avg = h @ nL / nL.sum()

# Baseline estimates
nrm = 1
r9week = estimate(data_for_est, nrm, ffopt="baseline", adj="ipw")
r9weeknp = estimate(data_for_est, nrm, ffopt="np", adj="ipw")

# Plot estimates
series = [r9week["psi"], r9weeknp["psi"], h_avg]
se = [r9week["psiSE"], None, None]
xticklabs = ["0-9", "9-18", "18-27", "27-36", "36-45"]
custom_plot(
    series,
    se,
    xlab="Weeks since unemployed",
    ylab="Hazard",
    legendlabs=["Log-Logistic", "Non-parametric", "Observed"],
    xticklabs=xticklabs,
    figsize=(4.35, 2.85),
    colors=[blue, red, black],
    linestyles=["-", "--", ":"],
    ylims=[-0.075, 0.8],
    legendpos="lower left",
    ydist=0.2,
)
plt.axvline(x=2, color="black", linestyle=":", linewidth=1.25, alpha=0.75)
plt.savefig(f"{OUTPUT_DIR}/fig_robust_9week.pdf", dpi=300, format="pdf")

##########################################################

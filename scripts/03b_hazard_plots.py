##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom functions
from utils.customplot import custom_plot
from utils.datamoms import data_moms

# Import parameters
from utils.config import DATA_DIR, QUANTS_DIR, OUTPUT_DIR

##########################################################
# Initialize
##########################################################

# Load data
sample = pd.read_csv(f"{DATA_DIR}/sample.csv")
data_for_est = sample[["notice", "dur", "cens"]].copy()
J = len(sample["notice"].unique())

# Load propensity scores and coefficients
ps = np.load(f"{QUANTS_DIR}/ps.npy")

# Adjusted data moments
h, se_h, S, se_S = data_moms(data_for_est, ps, purpose="output")

# Legend labels
legendlabs = ["Short notice", "Long notice"]
xticklabs = ["0-12", "12-24", "24-36", "36-48", ">48"]
xlab = "Weeks since unemployed"

##########################################################
# Plot hazard and survival rates for main analysis
##########################################################

# Plot adjusted hazard rates
series = [h[:, j] for j in range(J)]
se = [se_h[:, j] for j in range(J)]
custom_plot(
    series,
    se,
    xlab=xlab,
    ylab="Exit rate",
    legendlabs=legendlabs,
    xticklabs=xticklabs[:-1],
)
plt.savefig(f"{OUTPUT_DIR}/fig_hazard_ipw.pdf", dpi=300, format="pdf")

# Plot adjusted survival rates
series = [S[:, j] for j in range(J)]
se = [se_S[:, j] for j in range(J)]
custom_plot(
    series,
    se,
    xlab=xlab,
    ylab="Proportion surviving",
    legendlabs=legendlabs,
    xticklabs=xticklabs,
    ydist=0.2,
)
plt.savefig(f"{OUTPUT_DIR}/fig_survival_ipw.pdf", dpi=300, format="pdf")

##########################################################
# Appendix figure: unadjusted hazard rates
##########################################################

# Unadjusted data moments
h, se_h, S, se_S = data_moms(data_for_est, purpose="output")
series = [h[:, j] for j in range(J)]
se = [se_h[:, j] for j in range(J)]
custom_plot(
    series,
    se,
    xlab=xlab,
    ylab="Exit rate",
    legendlabs=legendlabs,
    xticklabs=xticklabs[:-1],
)
plt.savefig(f"{OUTPUT_DIR}/fig_hazard_unwtd.pdf", dpi=300, format="pdf")

##########################################################
# Appendix figure: Alternative bins
##########################################################

altbins = ["4week", "9week"]
xticklabs = [np.unique(sample["dur_4week"]).round(0).astype(int)]
xticklabs.append(["0-9", "9-18", "18-27", "27-36", "36-45", "45-54"])

for i, bins in enumerate(altbins):

    # Adjusted data moments
    data_for_est["dur"] = sample[f"dur_{bins}"]
    h, se_h, S, se_S = data_moms(data_for_est, ps, purpose="output")

    # Legend labels
    legendlabs = ["Short notice", "Long notice"]
    xlab = "Weeks since unemployed"

    # Plot adjusted hazard rates
    series = [h[:, j] for j in range(J)]
    se = [se_h[:, j] for j in range(J)]
    custom_plot(
        series,
        se,
        xlab=xlab,
        ylab="Exit rate",
        legendlabs=legendlabs,
        xticklabs=xticklabs[i][:-1],
    )
    plt.savefig(f"{OUTPUT_DIR}/fig_{bins}_hazard.pdf", dpi=300, format="pdf")

    # Plot adjusted survival rates
    series = [S[:, j] for j in range(J)]
    se = [se_S[:, j] for j in range(J)]
    custom_plot(
        series,
        se,
        xlab=xlab,
        ylab="Proportion surviving",
        legendlabs=legendlabs,
        xticklabs=xticklabs[i],
        ydist=0.2,
    )
    plt.savefig(f"{OUTPUT_DIR}/fig_{bins}_survival.pdf", dpi=300, format="pdf")

##########################################################

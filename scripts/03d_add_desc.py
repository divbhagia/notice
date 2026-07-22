##########################################################
# Housekeeping
##########################################################

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.duration.hazard_regression import PHReg

# Import custom functions and parameters
from utils.customplot import custom_plot, add_lines_labs, set_plot_aes
from utils.datadesc import latex, props
from utils.config import DATA_DIR, OUTPUT_DIR, Colors, CRIT
from utils.datamoms import data_moms

# Load data
sample = pd.read_csv(f"{DATA_DIR}/sample.csv")
notice = sample["notice"]

##########################################################
# Appendix figure: Length of notice over time
##########################################################

# Generate year dummies
X = sample["dyear"].copy().astype(int).astype(str)
X = pd.get_dummies(X, columns=["dyear"], drop_first=True)
X["intercept"] = 1
X = X.astype(float)

# Run linear regression without weights
model = OLS(notice, X).fit()
coefs = model.params
unbal = coefs.iloc[:-1] + coefs.iloc[-1]

# Run weighted linear regression
model = WLS(notice, X, weights=sample["wt"]).fit()
coefs = model.params
bal = coefs.iloc[:-1] + coefs.iloc[-1]

# Smooth data with a rolling mean, window of 3
unbal = unbal.rolling(window=2, center=True).mean()
bal = bal.rolling(window=2, center=True).mean()

# Plotting
custom_plot(
    [bal, unbal],
    legendlabs=["Balanced", "Unbalanced"],
    figsize=(5, 2.75),
    ylims=(0.35, 0.675),
    ylab="Proportion with >2 months notice",
)
plt.xticks(np.arange(0, len(bal), 3), np.arange(1996, 2021, 3))
plt.axhline(y=0.5, color="black", linestyle="--", linewidth=0.5)
plt.savefig(f"{OUTPUT_DIR}/fig_long_notice_ts.pdf")

##########################################################
# Appendix figure: Industry and occupation
##########################################################

# Divide the sample into two groups
df_s = sample[notice == 0]
df_l = sample[notice == 1]
p = {"s": {}, "l": {}}
se = {"s": {}, "l": {}}

# Unbalanced data
p["s"]["ind_ub"], se["s"]["ind_ub"] = props(df_s["ind_cat"])
p["s"]["occ_ub"], se["s"]["occ_ub"] = props(df_s["occ_cat"])
p["l"]["ind_ub"], se["l"]["ind_ub"] = props(df_l["ind_cat"])
p["l"]["occ_ub"], se["l"]["occ_ub"] = props(df_l["occ_cat"])

# Balanced data
p["s"]["ind_bal"], se["s"]["ind_bal"] = props(df_s["ind_cat"], df_s["wt"])
p["s"]["occ_bal"], se["s"]["occ_bal"] = props(df_s["occ_cat"], df_s["wt"])
p["l"]["ind_bal"], se["l"]["ind_bal"] = props(df_l["ind_cat"], df_l["wt"])
p["l"]["occ_bal"], se["l"]["occ_bal"] = props(df_l["occ_cat"], df_l["wt"])

# Labels
indlabs = add_lines_labs(p["s"]["ind_ub"].index.tolist())
occlabs = add_lines_labs(p["s"]["occ_ub"].index.tolist())

# Specify parameters
bw = 0.4  # bar width
fcolors = [(*Colors().RED, 0.1), (*Colors().WHITE, 1)]
ecolors = [(*Colors().RED, 1), (*Colors().BLACK, 0.75)]
patterns = ["", "/////"]

# Font size and set parameters for the plot
set_plot_aes()
plt.rcParams.update({"font.size": 8})
opts = []
for j in range(2):
    opts.append(
        {
            "facecolor": fcolors[j],
            "edgecolor": ecolors[j],
            "linewidth": 1,
            "capsize": 1.5,
            "ecolor": ecolors[j],
            "hatch": patterns[j],
            "color": "black",
        }
    )

# Plots
ix, ox = np.arange(len(indlabs)), np.arange(len(occlabs))
vars = ["ind_ub", "ind_bal", "occ_ub", "occ_bal"]
xax = [ix, ix, ox, ox]
labslist = [indlabs, indlabs, occlabs, occlabs]
for i, v in enumerate(vars):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.barh(xax[i] + bw / 2, p["s"][v], bw, xerr=CRIT * se["s"][v], **opts[0])
    ax.barh(xax[i] - bw / 2, p["l"][v], bw, xerr=CRIT * se["l"][v], **opts[1])
    ax.set_yticks(xax[i], labslist[i])
    ax.legend(labels=["Short", "Long"], loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_{v}.pdf", dpi=300, format="pdf", backend="pdf")

##########################################################
# Appendix table: Unemployment insurance take-up
##########################################################

# Initialize
sample_ = sample[sample["dwben"] <= 2]
recd_ui = (sample_["dwben"] == 2).astype(int)
intervals = [(-1, 0), (0, 4), (4, 8), (8, 12), (12, 1000)]
labels = ["0 weeks", "0-4 weeks", "4-8 weeks", "8-12 weeks", "> 12 weeks"]
bins = pd.IntervalIndex.from_tuples(intervals)

# Create table & write table to latex
durcat = pd.cut(sample_["obsdur"], bins=bins)
counts = recd_ui.groupby(durcat, observed=True).count().values
prop = recd_ui.groupby(durcat, observed=True).mean().values
prop = [f"{p:.2f}" for p in prop]
tab = np.column_stack((labels, counts, prop))
columns = ["Duration", "Observations", "Received UI Benefits"]
tab = pd.DataFrame(tab, columns=columns)
latex(tab, f"{OUTPUT_DIR}/tab_uiben_recd.tex", showindex=False)

##########################################################
# Appendix figure: Timing of Benefit Exhausation
##########################################################

# UI reciepients sample
sampleUI = sample[
    (sample["dwben"] == 2) & (sample["dwexben"] <= 2) & (sample["obsdur"] > 0)
].copy()
uiben_ex = (sampleUI["dwexben"] == 2).astype(int)

# Create table for the plot
uiben_ex.name = "prop"
tab = uiben_ex.groupby(sampleUI["dur"]).mean().reset_index()
tab["obs"] = uiben_ex.groupby(sampleUI["dur"]).count().values
tab["se"] = np.sqrt(tab["prop"] * (1 - tab["prop"]) / tab["obs"])

# Plot
xticklabs = ["0-12", "12-24", "24-36", "36-48", ">48"]
plt.figure(figsize=(5, 3))
plt.errorbar(
    tab["dur"],
    tab["prop"],
    yerr=CRIT * tab["se"],
    marker="o",
    linestyle="",
    color="black",
    markersize=3.5,
    capsize=3.5,
    capthick=1,
    elinewidth=1,
)
plt.xticks(tab["dur"], xticklabs)
plt.axvline(x=25.5, linestyle="--", color="black", linewidth=0.75)
plt.xlabel("Weeks since unemployed")
plt.ylabel("Exhausted UI Benefits")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.savefig(f"{OUTPUT_DIR}/fig_uiex_break.pdf")

##########################################################
# Appendix figure: Cox PH X-adjusted exit rates by notice
##########################################################

X = pd.read_csv(f"{DATA_DIR}/control_vars.csv").values.astype(float)
D = sample["obsdur"].values.astype(float)
event = 1 - sample["cens"].values.astype(int)
L = sample["notice"].values

# Unadjusted hazards (paper's fig_hazard_unwtd estimator)
h_unadj, _, _, _ = data_moms(
    sample[["notice", "dur", "cens"]], ps=None, purpose="output"
)

# X-adjusted hazards: Cox at raw duration, aggregated to 12-week bin edges
bin_edges = np.array([0.0, 12.0, 24.0, 36.0, 48.0])
h_adj = np.zeros((len(bin_edges) - 1, 2))
for l in [0, 1]:
    mask = L == l
    m = PHReg(D[mask], X[mask], status=event[mask], ties="breslow").fit()
    times, cum_haz = np.asarray(m.baseline_cumulative_hazard[0][0]), np.asarray(
        m.baseline_cumulative_hazard[0][1]
    )
    H0 = np.interp(bin_edges, times, cum_haz, left=0.0)
    S = np.exp(-np.outer(np.exp(X[mask] @ m.params), H0)).mean(axis=0)
    h_adj[:, l] = 1 - S[1:] / S[:-1]

    # Normalize so adjusted h(1) matches unadjusted h(1)
    h_adj[:, l] = h_adj[:, l] * (h_unadj[0, l] / h_adj[0, l])

custom_plot(
    [h_unadj[:, 0], h_adj[:, 0], h_unadj[:, 1], h_adj[:, 1]],
    xlab="Weeks since unemployed",
    ylab="Exit rate",
    legendlabs=[
        "Short, unadjusted",
        "Short, adjusted",
        "Long, unadjusted",
        "Long, adjusted",
    ],
    xticklabs=["0-12", "12-24", "24-36", "36-48"],
    colors=[Colors.BLACK, Colors.BLACK, Colors.RED, Colors.RED],
    linestyles=["-", "--", "-", "--"],
    figsize=(5, 3),
)
plt.savefig(f"{OUTPUT_DIR}/fig_cox_haz.pdf")

##########################################################

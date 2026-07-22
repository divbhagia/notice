##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Import custom functions and parameters
from utils.customplot import set_plot_aes
from utils.datadesc import pred_ps, rmlinestex
from utils.config import DATA_DIR, QUANTS_DIR, Colors, OUTPUT_DIR
from utils.config import print_bold

##########################################################
# Sample selection
##########################################################

# Load data
dws = pd.read_csv(f"{DATA_DIR}/dws.csv")

# Variables with missing values
missingval_vars = [
    "dwlastwrk",
    "dwnotice",
    "dwunion",
    "dwhi",
    "dwyears",
    "dwweekl",
    "dwjobsince",
    "ind",
    "occ",
    "obsdur",
]
# Note obsdur is missing when jf=1 & dwwksun is missing

# Sample selection
sample = dws.copy()
sample = sample[(sample["year"] >= 1996) & (sample["year"] <= 2020)]
sample = sample[(sample["age"] >= 21) & (sample["age"] <= 64)]

# Conditions for sample selection
cond1 = sample["dwrecall"] != 2
cond2 = sample["dwclass"] <= 3
cond3 = sample[missingval_vars].isnull().sum(axis=1) == 0
cond4 = sample["dwfulltime"] == 2
cond5 = sample["dwyears"] >= 0.5
cond6 = sample["dwhi"] == 2
cond7 = sample["dwjobsince"] <= 2
cond8 = (sample["dwnotice"] >= 2) & (sample["dwnotice"] <= 4)
cond9 = (sample["dwnotice"] >= 3) & (sample["dwnotice"] <= 4)
conds = np.array([eval(f"cond{j+1}") for j in range(9)]).T

# Also keep sample with all notcats
sample_all = sample.loc[conds[:, :7].all(axis=1)].copy()
sample_all.to_csv(f"{DATA_DIR}/sample_all.csv", index=False)

# Labels for conditions
condlabs = [
    "DWS 1996-2020, 21-64 year old respondents",
    "No recall expectation",
    "Lost job was not self-employment",
    "Non-missing values for variables used",
    "Worked full-time at lost job",
    "Employed for at least 6 months at lost job",
    "Had health insurance at lost job",
    "Held less than 3 jobs since lost job",
    "Got a notice before job loss",
    "Got a notice of >1 month",
]

# Apply conditions and record sample size reductions in a table
sample_sizes = []
for j in range(conds.shape[1]):
    sample_sizes.append(sample.shape[0])
    sample = sample.loc[conds[:, j]]
    conds = conds[conds[:, j]]
sample_sizes.append(sample.shape[0])

# Create latex table
tab = np.array([condlabs, sample_sizes]).T
print(tabulate(tab))
tabpath = f"{OUTPUT_DIR}/tab_sample_selection.tex"
with open(tabpath, "w") as f:
    f.write(tabulate(tab, tablefmt="latex"))
rmlinestex(tabpath)

# Are there individuals who haven't found a job and left lf?
sample[(sample["jf"] == 0)]["nilf"].value_counts()

##########################################################
# Function to prepare control variables
##########################################################


def prep_controls(sample):

    # Specify controls
    varlist = [
        "age",
        "female",
        "married",
        "black",
        "educ_cat",
        "dwreas",
        "union",
        "in_metro",
        "dwyears",
        "lnearnl",
        "statefip",
        "dyear",
        "ind_cat",
        "occ_cat",
    ]

    # Keep only relevant variables
    controls = sample[varlist].copy()

    # Add quadratic terms for continuous variables
    controls["age2"] = controls["age"] ** 2
    controls["dwyears2"] = controls["dwyears"] ** 2

    # Categories as numbers
    controls["dwreas"] = controls["dwreas"].astype(int)
    controls["statefip"] = controls["statefip"].astype(int)
    controls["dyear"] = controls["dyear"].astype(int)
    controls["educ_cat"] = pd.Categorical(controls["educ_cat"]).codes
    controls["ind_cat"] = pd.Categorical(controls["ind_cat"]).codes
    controls["occ_cat"] = pd.Categorical(controls["occ_cat"]).codes

    # Hot encode multiple category variables
    catvars = ["dwreas", "occ_cat", "statefip", "dyear", "ind_cat", "educ_cat"]
    controls = pd.get_dummies(controls, columns=catvars, drop_first=True)

    # Convert all boolean variables to integers
    for cols in controls.columns:
        if controls[cols].dtype == bool:
            controls[cols] = controls[cols].astype(int)

    return controls


##########################################################
# Estimate propensity scores and add balancing weights
##########################################################

# Save control variables for later use
controls = prep_controls(sample)
controls_all = prep_controls(sample_all)
controls.to_csv(f"{DATA_DIR}/control_vars.csv", index=False)
controls_all.to_csv(f"{DATA_DIR}/controls_all.csv", index=False)

# Estimate propensity scores and save results
est_data = pd.concat([sample[["notice", "dur", "cens"]], controls], axis=1)
ps, coefs = pred_ps(est_data)
np.save(f"{QUANTS_DIR}/ps.npy", ps)
np.save(f"{QUANTS_DIR}/coefs.npy", coefs)

# Check if any extreme propensity scores
notvals = np.sort(sample["notice"].unique())
keep = 0
ll, ul = 0.1, 0.9
for j in range(1, len(notvals)):
    keep += (ps[:, j] > ll) & (ps[:, j] < ul)
keep = keep == max(keep)
print_bold(f"Observations with {ll}<PS<{ul}: {sum(keep)}/{len(keep)}")

# Add balancing weights to the data
for j in range(len(notvals)):
    sample.loc[(sample["notice"] == notvals[j]), "wt"] = (
        1 / ps[(sample["notice"] == notvals[j]), j]
    )

# Save sample with weights
sample.to_csv(f"{DATA_DIR}/sample.csv", index=False)

##########################################################
# Check overlap in propensity scores
##########################################################

set_plot_aes()
colors = [Colors().BLACK, Colors().RED]
plt.figure(figsize=(5, 3))
for j in range(len(notvals)):
    sns.kdeplot(
        ps[(sample["notice"] == notvals[j]), 1],
        color=colors[j],
        fill=True,
        alpha=0.075,
        edgecolor="black",
    )
plt.annotate(
    "Short Notice",
    xy=(0.4, 2.05),
    xytext=(0.125, 2),
    arrowprops=dict(facecolor="black", arrowstyle="<-"),
)
plt.annotate(
    "Long Notice",
    xy=(0.615, 2.45),
    xytext=(0.725, 2.4),
    arrowprops=dict(facecolor="black", arrowstyle="<-"),
)
sns.despine()
plt.xticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Density")
plt.xlabel("Propensity score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig_ps_balance.pdf", dpi=300)

##########################################################

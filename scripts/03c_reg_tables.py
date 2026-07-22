##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Import custom functions and parameters
from utils.datadesc import sig_asterisk
from utils.config import DATA_DIR, OUTPUT_DIR

##########################################################
# Initialize
##########################################################

# Load data
sample = pd.read_csv(f"{DATA_DIR}/sample.csv")
X = pd.read_csv(f"{DATA_DIR}/control_vars.csv")

# Data for estimation
notice = sample[["notice"]].copy()
notice["itercept"] = 1
allX = pd.concat([notice, X], axis=1)

# Indicator for leaving in first interval
d0 = np.sort(sample["dur"].unique())[0]
h0 = np.where(sample["obsdur"] == 0, 1, 0)
h0to12 = np.where((sample["dur"] == d0) & (sample["cens"] == 0), 1, 0)
lnearn = sample["lnearnc"]

# Print mean of h0 and h0to12 by notice
m_h0 = pd.crosstab(sample["notice"], h0, normalize="index")[1]
m_h0to12 = pd.crosstab(sample["notice"], h0to12, normalize="index")[1]
print(f"Mean of h0 by notice: {m_h0}")
print(f"Mean of h0to12 by notice: {m_h0to12}")

##########################################################
# Initial hazard and wages at subsequent job regressions
##########################################################

vars = [h0, h0to12, lnearn]
files = ["tab_reg_init_hazardA", "tab_reg_init_hazardB", "tab_reg_wagesA"]

for v in range(len(vars)):

    # Initialize
    var = vars[v]
    coefs = np.array(4 * [""], dtype="<U16")
    se = np.array(4 * [""], dtype="<U16")

    # Sample for earnings regression
    if v == 2:
        cond = (sample["dwjobsince"] == 1) & (~np.isnan(lnearn))
        notice, allX = notice[cond], allX[cond]
        sample, var = sample[cond], var[cond]

    # Fit models
    model1 = sm.OLS(var, notice).fit(cov_type="HC1")
    model2 = sm.OLS(var, allX).fit(cov_type="HC1")
    model3 = sm.WLS(var, notice, weights=sample["wt"]).fit(cov_type="HC1")
    model4 = sm.WLS(var, allX, weights=sample["wt"]).fit(cov_type="HC1")
    models = [model1, model2, model3, model4]

    # Extract coefficients and standard errors
    for i, model in enumerate(models):
        sig_ast = sig_asterisk(model.pvalues["notice"])
        coefs[i] = f"{model.params['notice']:.3f}{sig_ast}"
        se[i] = f"({model.bse['notice']:.3f})"

    # Write to file
    coefs = np.insert(coefs, 0, "> 2 month notice")
    se = np.insert(se, 0, " ")
    with open(f"{OUTPUT_DIR}/{files[v]}.tex", "w") as f:
        f.write(" & ".join(coefs) + "\\\\" + "\n")
        f.write(" & ".join(se) + "\\\\")

    # Write number of observations
    nobs = [f"{model.nobs:.0f}" for model in models]
    nobs = np.array(nobs, dtype="<U16")
    nobs = np.insert(nobs, 0, "Observations")
    if v != 2:
        nobsfile = "tab_reg_init_hazardC"
    else:
        nobsfile = "tab_reg_wagesB"
    with open(f"{OUTPUT_DIR}/{nobsfile}.tex", "w") as f:
        f.write(" & ".join(nobs) + "\\\\")

##########################################################

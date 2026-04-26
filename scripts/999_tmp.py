##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
import statsmodels.api as sm

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
ps = pd.Series(ps[:, 1], index=sample.index, name="ps")

# Add ps to sample
sample = pd.concat([sample, ps], axis=1)

# Generate variables
d0 = np.sort(sample["dur"].unique())[0]
h0 = np.where(sample["obsdur"] == 0, 1, 0)
h0to12 = np.where((sample["dur"] == d0) & (sample["cens"] == 0), 1, 0)
sample["h0to12"] = h0to12
lnearn = sample["lnearnc"]

# Estimate g(1|L, X)
controls = pd.concat([sample["notice"], X], axis=1)
model = sm.Logit(h0to12, controls).fit()
print(model.summary())

# predict g(1|L, X)
sample["g1"] = model.predict(controls)

# correlation of g and notice
print(sample[["g1", "notice"]].corr())

# correlation of g and h0to12
print(sample[["g1", "h0to12"]].corr())

# Print g1 for notice = 0 and notice = 1
print(sample.groupby("notice")["g1"].mean())

# variance in

h, h_se, S, S_se = data_moms(sample, ps=None, purpose="output")
ps = np.load(f"{QUANTS_DIR}/ps.npy")
h_wt, h_se, S, S_se = data_moms(sample, ps=ps, purpose="output")

mu = h[0, :] / h_wt[0, :]
mu / mu[0]

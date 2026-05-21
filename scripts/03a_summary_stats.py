import numpy as np
import pandas as pd
from utils.datadesc import sum_tab, latex

# Data directory
from utils.config import DATA_DIR, OUTPUT_DIR

##########################################################
# Summary statistics for the sample (Table 1)
##########################################################

# Load data
sample = pd.read_csv(f"{DATA_DIR}/sample.csv")

# Variable list
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

# Variable labels
labels = [
    "Age",
    "Female",
    "Married",
    "Black",
    "College Degree",
    "Plant Closure",
    "Union Membership",
    "In Metro Area",
    "Years of Tenure",
    "Log Earnings",
]

# Unbalanced and balanced summary statistics
table1 = sum_tab(sample, varlist, "notice", labels)
table2 = sum_tab(sample, varlist, "notice", labels, wts="wt")[:, 1:]

# Merge table 1 & 2 by column
extracol = np.array([[""] * (2 * len(varlist) + 1)]).T
table = np.concatenate((table1, extracol, table2), axis=1)

# Print table to latex
latex(table, f"{OUTPUT_DIR}/tab_sum_stats.tex")

##########################################################
# Appendix table: Comparison to DWS & CPS
##########################################################

# Load more data data
cps = pd.read_csv(f"{DATA_DIR}/cps.csv")
dws = pd.read_csv(f"{DATA_DIR}/dws.csv")
sample["dataind"], dws["dataind"], cps["dataind"] = 1, 2, 3
master = pd.concat([cps, dws, sample], axis=0)

# Select sample
master = master[(master["year"] >= 1996) & (master["year"] <= 2020)]
master = master[(master["age"] >= 21) & (master["age"] <= 64)]

# Variable list and labels
varlist = [
    "age",
    "female",
    "married",
    "black",
    "hs",
    "sc",
    "col",
    "emp",
    "unemp",
    "nilf",
]
labels = [
    "Age",
    "Female",
    "Married",
    "Black",
    "High School",
    "Some College",
    "College Degree",
    "Employed",
    "Unemployed",
    "NILF",
]

# Create table and output to latex
table = sum_tab(master, varlist, "dataind", labels, diff=False, se=False)
latex(table, f"{OUTPUT_DIR}/tab_cps_comp.tex", addlines=[9])

##########################################################

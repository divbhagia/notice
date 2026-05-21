import numpy as np
from tabulate import tabulate
from scipy.stats import t
import statsmodels.api as sm

##########################################################
# Function to predict propensity scores
##########################################################


def pred_ps(data, coefs=None, outmodel=False):

    # Initialize
    notice = data["notice"]
    notX_vars = ["dur", "cens", "notice", "cens_ind", "wt", "wts"]
    X = data[[col for col in data.columns if col not in notX_vars]]

    # Fit model if no coefficients are provided
    if coefs is None:
        model = sm.MNLogit(notice, X).fit(disp=0)
        coefs = model.params.values

    # Predict propensity scores
    betaL = np.column_stack([np.zeros(X.shape[1]), coefs])
    X = np.array(X, dtype=float)
    ps = np.exp(X @ betaL) / np.exp(X @ betaL).sum(axis=1, keepdims=True)

    # Return model if requested
    if outmodel:
        return ps, coefs, model

    return ps, coefs


##########################################################
# Function to calculate proportions and standard errors
##########################################################


def props(var, wt=None):
    if wt is None:
        p = var.value_counts(normalize=True).sort_index()
        se = np.sqrt(p * (1 - p) / len(var))
        return p, se
    p = wt.groupby(var).sum() / np.sum(wt)
    adj = np.sum(wt**2) / np.sum(wt) ** 2
    se = np.sqrt(p * (1 - p) * adj)
    return p, se


##########################################################
# Functions to write tables to latex
##########################################################


# Remove enclosing lines from a latex table
def rmlinestex(file):
    with open(file, "r") as f:
        lines = f.readlines()
    with open(file, "w") as f:
        for line in lines:
            if (
                not line.startswith("\\hline")
                and not line.startswith("\\begin{tabular}")
                and not line.startswith("\\end{tabular}")
            ):
                f.write(line)


# Write table to latex
def latex(table, file, rmlines=True, addlines=None, **kwargs):
    with open(file, "w") as f:
        f.write(tabulate(table, tablefmt="latex", disable_numparse=True, **kwargs))
    if rmlines:
        rmlinestex(file)
    if addlines is not None:
        lines = open(file, "r").readlines()
        with open(file, "w") as f:
            for i, line in enumerate(lines):
                f.write(line)
                if i in addlines:
                    f.write(f"\\midrule\n")


# Find significant rows
def find_sig_rows(table):
    sig_rows = np.array([False] * len(table))
    for i, x in enumerate(table):
        for j in range(len(x)):
            if "***" in x[j]:
                sig_rows[i] = True
    return sig_rows


##########################################################
# Function to give asterisks for significance
##########################################################


def sig_asterisk(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return " "


##########################################################
# Mean and Standard Error for weighted & unweighted data
##########################################################


def var_stats(var, wts):
    wts = 1 if wts is None else wts
    n = len(var)
    mean = np.average(var, weights=wts)
    variance = np.sum(wts * (var - mean) ** 2) / np.sum(wts)
    se = np.sqrt(variance) / np.sqrt(n)
    return mean, se, variance, n


##########################################################
# Summary statistics for a variable by category
##########################################################


def stats_by_cat(df, var, byvar, label=None, wts=None, stars=True):

    # Initialize
    label = var if label is None else label
    levels = df[byvar].value_counts().sort_index().index
    K = len(levels)
    df[wts] = 1 if wts is None else df[wts]
    row1 = np.array((2 * K - 1) * [""], dtype="<U32")
    row2 = np.array((2 * K - 1) * [""], dtype="<U32")

    # Subset data by category
    for k in range(K - 1):

        # Subset data
        level = levels[k]
        next_level = levels[k + 1]
        group1 = df[df[byvar] == level]
        group2 = df[df[byvar] == next_level]
        n1, n2 = len(group1), len(group2)

        # Calculate means & standard errors
        mean1, se1, variance1, n1 = var_stats(group1[var], group1[wts])
        mean2, se2, variance2, n2 = var_stats(group2[var], group2[wts])

        # Compare means: t-test
        S = np.sqrt((variance1 / n1) + (variance2 / n2))
        t_stat = (mean1 - mean2) / S
        p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=(n1 + n2 - 2)))
        sig_ast = sig_asterisk(p_val)

        # Fill rows
        row1[k] = f"{mean1:.2f}"
        row1[k + 1] = f"{mean2:.2f}"
        row1[K + k] = f"{mean2-mean1:.2f}"
        if stars:
            row1[K + k] = f"{row1[K+k]}{sig_ast}"
        row2[k] = f"({se1:.2f})"
        row2[k + 1] = f"({se2:.2f})"
        row2[K + k] = f"({S:.2f})"

    # Add label
    row1 = np.insert(row1, 0, label)
    row2 = np.insert(row2, 0, "")

    return row1, row2


##########################################################
# Summary statistics table for multiple variables by category
##########################################################


def sum_tab(df, varlist, byvar, labels=None, wts=None, se=True, stars=True, diff=True):
    labels = varlist if labels is None else labels
    table = []
    for i in range(len(varlist)):
        row1, row2 = stats_by_cat(df, varlist[i], byvar, labels[i], wts, stars)
        table.append(row1)
        if se:
            table.append(row2)
    nL = df[byvar].value_counts().sort_index().values
    last_row = np.array((2 * len(nL)) * [""], dtype="<U32")
    last_row[0] = "Observations"
    last_row[1 : len(nL) + 1] = nL
    table.append(last_row)
    table = np.array(table)
    if diff is False:
        table = table[:, : len(nL) + 1]
    return table


##########################################################

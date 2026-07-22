import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.datamoms import data_moms
from utils.config import DATA_DIR, QUANTS_DIR, OUTPUT_DIR, Colors

# Use a real LaTeX backend so \underline renders
plt.rcParams["text.usetex"] = True

black = Colors().BLACK
grey = Colors().GREY

########################################################################
# Load data and joint exit probabilities
########################################################################

sample = pd.read_csv(f"{DATA_DIR}/sample.csv")
ps = np.load(f"{QUANTS_DIR}/ps.npy")

h_data, _, S, _ = data_moms(sample, ps=None, purpose="output")
g = h_data * S[:-1, :]

# Unpack g: short-notice ("l") in col 0, long-notice ("l'") in col 1
g1, g1p = g[0]
g2, g2p = g[1]
g3, g3p = g[2]
g4, g4p = g[3]


########################################################################
# Closed-form solvers for (lam_d, psi_d) given the composite wedge eta
########################################################################

# Each pair of moment equations has the form
#     psi_d * (A_d + B_d * lam_d)         = g_d
#     eta * psi_d * (Ap_d + Bp_d * lam_d) = g_dp
# with A, B, Ap, Bp known from earlier steps. Dividing eliminates psi_d and
# leaves a linear equation in lam_d with the closed-form solution
#     lam_d = (Ap_d - r_d * A_d) / (r_d * B_d - Bp_d),   r_d = g_dp / (eta * g_d)


def x2(eta):
    A, B = 1.0, -g1
    Ap, Bp = 1.0, -g1p
    r = g2p / (eta * g2)
    lam2 = (Ap - r * A) / (r * B - Bp)
    psi2 = g2 / (A + B * lam2)
    return lam2, psi2


def x3(eta):
    lam2, psi2 = x2(eta)
    e1, e2 = g1 + psi2, g1 * psi2
    e1p, e2p = g1p + eta * psi2, g1p * eta * psi2
    A, B = 1 - e1 * lam2, e2
    Ap, Bp = 1 - e1p * lam2, e2p
    r = g3p / (eta * g3)
    lam3 = (Ap - r * A) / (r * B - Bp)
    psi3 = g3 / (A + B * lam3)
    return lam3, psi3


def x4(eta):
    lam2, psi2 = x2(eta)
    lam3, psi3 = x3(eta)
    e1 = g1 + psi2 + psi3
    e2 = g1 * psi2 + g1 * psi3 + psi2 * psi3
    e3 = g1 * psi2 * psi3
    e1p = g1p + eta * (psi2 + psi3)
    e2p = g1p * eta * (psi2 + psi3) + (eta**2) * psi2 * psi3
    e3p = g1p * (eta**2) * psi2 * psi3
    A, B = 1 - e1 * lam2 + e2 * lam3, -e3
    Ap, Bp = 1 - e1p * lam2 + e2p * lam3, -e3p
    r = g4p / (eta * g4)
    lam4 = (Ap - r * A) / (r * B - Bp)
    psi4 = g4 / (A + B * lam4)
    return lam4, psi4


def eta_from_lam2(lam2):
    """Inverse of x2: returns the eta that produces the given lam2."""
    return (g2p / g2) * (1 - g1 * lam2) / (1 - g1p * lam2)


########################################################################
# Trace lam_d as a function of eta and locate the cumulative lower bound
########################################################################

# Upper bound on eta from the support argument: lambda2 <= 1 / (g1p + g2p)
lam2_ub = 1 / (g1p + g2p)
eta_ub = eta_from_lam2(lam2_ub)
print(f"Upper bound on eta: {eta_ub:.3f}")
eta = np.linspace(0.8, eta_ub, 800)


def plot_lam(
    eta_grid,
    lam,
    floor,
    eta_lb,
    eta_ub,
    ylabel,
    floor_label,
    step=0.05,
    savepath=None,
):
    """One panel: lam vs eta_grid with floor and dashed bounds at eta_lb, eta_ub."""
    plt.figure(figsize=[2, 2])
    plt.plot(eta_grid, lam, color=black, linestyle="-", label=ylabel)
    if np.isscalar(floor):
        plt.axhline(floor, color=grey, linewidth=1, label=floor_label)
    else:
        plt.plot(eta_grid, floor, color=grey, linewidth=1, label=floor_label)
    plt.axvline(eta_lb, color=grey, linestyle="--", linewidth=1)
    plt.axvline(eta_ub, color=grey, linestyle="--", linewidth=1)
    lo = np.ceil(eta_grid[0] / step) * step
    hi = np.floor(eta_grid[-1] / step) * step
    xticks = np.round(np.arange(lo, hi + step / 2, step), 2)
    plt.xticks(xticks, [f"{x:.2f}" for x in xticks])
    # Headroom at the top so the eta annotation has clear space
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax + 0.20 * (ymax - ymin))
    plt.annotate(
        rf"$\underline{{\eta}}={eta_lb:.2f}$",
        xy=(eta_lb, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
    )
    plt.xlabel(r"$\eta$")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, format="pdf")


# Lambda2: range = full initial grid; floor is the constant 1.
lam2, _ = np.array([x2(e) for e in eta]).T
feas_eta = eta[lam2 >= 1]
eta_lb_lam2 = feas_eta[0]
print(f"Feasible lower bound (lam2): {eta_lb_lam2:.3f}")
plot_lam(
    eta,
    lam2,
    1.0,
    eta_lb_lam2,
    eta_ub,
    r"$\lambda_2$",
    r"$\lambda_2 = 1$",
    step=0.15,
    savepath=f"{OUTPUT_DIR}/fig_sen_ineq1.pdf",
)


# Lambda3: range = feasible region after the lam2 cut; floor is lam2^2.
eta_grid = feas_eta
lam2, _ = np.array([x2(e) for e in eta_grid]).T
lam3, _ = np.array([x3(e) for e in eta_grid]).T
floor3 = lam2**2
feas_eta = eta_grid[lam3 >= floor3]
eta_lb_lam3 = feas_eta[0]
print(f"Feasible lower bound (lam3): {eta_lb_lam3:.3f}")
plot_lam(
    eta_grid,
    lam3,
    floor3,
    eta_lb_lam3,
    eta_ub,
    r"$\lambda_3$",
    r"$\lambda_2^2$",
    step=0.1,
    savepath=f"{OUTPUT_DIR}/fig_sen_ineq2.pdf",
)


# Lambda4: range = feasible region after the lam3 cut; floor is the 3x3 Hankel floor.
eta_grid = feas_eta
lam2, _ = np.array([x2(e) for e in eta_grid]).T
lam3, _ = np.array([x3(e) for e in eta_grid]).T
lam4, _ = np.array([x4(e) for e in eta_grid]).T
floor4 = (lam3**2 - 2 * lam2 * lam3 + lam2**3) / (lam2 - 1)
feas_eta = eta_grid[lam4 >= floor4]
eta_lb_lam4 = feas_eta[0]
print(f"Feasible lower bound (lam4): {eta_lb_lam4:.3f}")
plot_lam(
    eta_grid,
    lam4,
    floor4,
    eta_lb_lam4,
    eta_ub,
    r"$\lambda_4$",
    r"$\underline{\lambda}_4$",
    savepath=f"{OUTPUT_DIR}/fig_sen_ineq3.pdf",
)

# Save eta bounds for use in other scripts. The lam4 step combined all three
# inequalities cumulatively, so eta_lb_lam4 is the overall lower bound.
print(f"Overall feasible eta range: [{eta_lb_lam4:.3f}, {eta_ub:.3f}]")
np.save(f"{QUANTS_DIR}/eta_bounds.npy", np.array([eta_lb_lam4, eta_ub]))

########################################################################

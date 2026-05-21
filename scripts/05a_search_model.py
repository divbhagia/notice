import numpy as np
from scipy.optimize import minimize
from utils.customplot import custom_plot
from utils.searchmodel import parameters, avg_opt
from utils.config import QUANTS_DIR, OUTPUT_DIR, Colors

######################################################
# Initialization
######################################################

# Parameters
T = 4
sgma, beta, rho, thta, w, b, a = parameters(T, Tb=3)
dlta0 = 1

# Load observed h and estimated psiH
h_data = np.load(f"{QUANTS_DIR}/h_avg_ipw.npy")
est = np.load(f"{QUANTS_DIR}/baseline_ests.npy", allow_pickle=True)
est = est.item()
psiH = est["psi"]


# Distance to find optimal dlta & nu
def distance(x, h_opt, h_data, types):
    dlta = np.append(dlta0, x[: T - 1])
    if types == 1:
        if len(x) != T:
            raise ValueError("len(x) must be equal to T")
        nu = np.array([x[-1]])
        p = np.array([1])
    elif types == 2:
        if len(x) != T + 2:
            raise ValueError("len(x) must be equal to T+2")
        nu = x[T - 1 : T + 1]
        p = np.array([x[-1], 1 - x[-1]])
    r = avg_opt(T, sgma, beta, rho, thta, w, b, a, dlta, nu=nu, p=p)
    vec_dist = np.append(r["h_str"] - h_opt, r["h_obs"] - h_data)
    dist = np.linalg.norm(vec_dist)
    if dlta.min() < 0 or nu.min() < 0:
        dist = 1e10
    return dist


######################################################
# Calibrations
######################################################

# Optimization options
opt = {"disp": False, "maxiter": 100000, "ftol": 1e-12, "xtol": 1e-2}
mthd = "Powell"

# Calibration 1: h_str = h_obs, no heterogeneity
x0 = 0.5 * np.ones(T)
x = minimize(distance, x0, args=(h_data, h_data, 1), method=mthd, options=opt).x
dlta1 = np.concatenate([np.array([dlta0]), x[: T - 1]])
nu1 = np.array([x[-1]])
r1 = avg_opt(T, sgma, beta, rho, thta, w, b, a, dlta1, nu=nu1)

# Calibration 2: h_str = h_obs, no heterogeneity
x0 = 0.5 * np.ones(T + 2)
x = minimize(distance, x0, args=(psiH, h_data, 2), method=mthd, options=opt).x
dlta2 = np.concatenate([np.array([dlta0]), x[: T - 1]])
nu2 = x[T - 1 : T + 1]
p2 = np.array([x[-1], 1 - x[-1]])
r2 = avg_opt(T, sgma, beta, rho, thta, w, b, a, dlta2, nu=nu2, p=p2)

######################################################
# Assess model fit
######################################################

# Panel A
xticklabs = ["0-12", "12-24", "24-36", "36-48"]
xlab = "Weeks since unemployed"
labs = ["Data", "No heterogeneity", "2-types of workers"]
colors = [Colors.BLACK, Colors.RED, Colors.BLUE]
linestyles = ["-", ":", "--"]
linewidths = [1.5, 3, 1.5]
savepath = f"{OUTPUT_DIR}/fig_search_model_fitA.pdf"
custom_plot(
    [h_data, r1["h_obs"], r2["h_obs"]],
    colors=colors,
    xticklabs=xticklabs,
    xlab=xlab,
    linestyles=linestyles,
    linewidths=linewidths,
    legendlabs=labs,
    savepath=savepath,
)

# Panel B
colors = [Colors.BLACK, Colors.BLUE]
linestyles = ["-", "--"]
labs = ["MH Estimate", "2-types of workers"]
savepath = f"{OUTPUT_DIR}/fig_search_model_fitB.pdf"
custom_plot(
    [psiH, r2["h_str"]],
    colors=colors,
    xticklabs=xticklabs,
    xlab=xlab,
    linestyles=linestyles,
    legendlabs=labs,
    ylims=[0.19, 0.81],
    ydist=0.2,
    savepath=savepath,
)

######################################################
# Main figure
######################################################

labs = ["No heterogeneity", "2-types of workers"]
linestyles = ["--", "-"]
colors = [Colors.BLACK, Colors.BLUE]

# Panel A
custom_plot(
    [dlta1, dlta2],
    legendlabs=labs,
    xticklabs=xticklabs,
    xlab=xlab,
    ylab="Offer Arrival Rate",
    colors=colors,
    linestyles=linestyles,
    legendpos="lower left",
    ylims=[0.4, 1.01],
    savepath=f"{OUTPUT_DIR}/fig_calib_offer.pdf",
)

# Panel B
custom_plot(
    [r1["s_str"] / r1["s_str"][0], r2["s_str"] / r2["s_str"][0]],
    legendlabs=labs,
    xticklabs=xticklabs,
    ylims=[0.59, 1.41],
    ydist=0.2,
    colors=colors,
    ylab="Search Effort",
    linestyles=linestyles,
    legendpos="lower left",
    xlab=xlab,
    savepath=f"{OUTPUT_DIR}/fig_calib_search.pdf",
)

######################################################

import numpy as np
from utils.searchmodel import parameters, uiben, avg_opt
from utils.customplot import custom_plot

######################################################
# Baseline parameters
######################################################


# Log-logistic duration dependence (k>1 inc, k<1 dec, k=1 constant)
def durdep(d, k=0.5, alph=0.5):
    dlta = (k * alph) * ((alph * d + 1)) ** (k - 1)
    return dlta


# Parameters
T = 12
Tb = 6
sgma, beta, rho, thta, w, b, a = parameters(T, Tb)
dlta = [durdep(d, k=1, alph=0.4) for d in range(T)]
nu = 1
pars = (T, sgma, beta, rho, thta, w, b, a, dlta)

######################################################
# Test 1: Impact of UI Extension
######################################################

b1 = uiben(T, Tb=4, r=0.5)
b2 = uiben(T, Tb=8, r=0.5)
dlta2 = np.append(2 * dlta[0], dlta[1:])
pars1 = (T, sgma, beta, rho, thta, w, b1, a, dlta)
pars2 = (T, sgma, beta, rho, thta, w, b2, a, dlta)
r1 = avg_opt(*pars1, nu=[nu, nu])
r2 = avg_opt(*pars2, nu=[nu, nu])

custom_plot(
    [r1["s_str"], r2["s_str"], dlta],
    legendlabs=["4 Weeks", "8 Weeks", "Arrival Rate"],
    ylab="Search Effort",
    xlab="Weeks",
    title="Impact of UI Extension on Search Effort",
)

######################################################
# Test 2: Observed hazard vs structural hazard
######################################################

# Two types of workers
nu = [0.5, 1]
p = [0.2, 0.8]
r = avg_opt(*pars, nu=nu, p=p)
custom_plot(
    [r["h_str"], r["h_obs"]],
    legendlabs=["Structural", "Observed"],
    ylab="Exit Rate",
    xlab="Weeks",
    title="With Heterogenity and Constant Arrival Rate",
)

######################################################

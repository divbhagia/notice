import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.searchmodel import parameters, avg_opt, sim_search_model
import multiprocessing as mp
from utils.customplot import custom_plot
from utils.config import QUANTS_DIR, OUTPUT_DIR, Colors, SIM_SM_AGAIN

# np print options
np.set_printoptions(precision=4, suppress=True)

######################################################
# Function to simulate the search model using mp
######################################################


def sim_multi(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, nrm, seeds):
    num_cores = round(mp.cpu_count() / 1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(
        sim_search_model,
        [(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, nrm, seed) for seed in seeds],
    )
    pool.close()
    np.save(f"{QUANTS_DIR}/sim_search_results.npy", r)
    return r


######################################################
# Main program
######################################################

if __name__ == "__main__":

    # Parameters
    np.random.seed(1117)
    T, Tb, n = 4, 3, 3000
    othpars = parameters(T, Tb)
    nu = [1, 0.5]
    p = [0.5, 0.5]

    # Two notice types
    pi = np.array([0.5, 0.5])
    nL = n * pi
    dlta0 = np.append(1.00, 0.95 * np.ones(T - 1))  # short notice
    dlta1 = np.append(1.25, 0.95 * np.ones(T - 1))  # long notice

    # Average structural (E[h(d|nu)]) and observed hazard
    L0 = avg_opt(T, *othpars, dlta0, nu=nu, p=p)
    L1 = avg_opt(T, *othpars, dlta1, nu=nu, p=p)
    h_str = L0["h_str"] * pi[0] + L1["h_str"] * pi[1]
    piD = np.array([L0["S"] / (L0["S"] + L1["S"]), L1["S"] / (L0["S"] + L1["S"])]).T
    h_obs = L0["h_obs"] * piD[:-1:, 0] + L1["h_obs"] * piD[:-1:, 1]

    # Simulate the search model
    iters = 1000
    nrm = 1
    ffopt = "baseline"
    seeds = np.random.choice(999999999, iters)
    if SIM_SM_AGAIN:
        results = sim_multi(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, nrm, seeds)
    else:
        results = np.load(f"{QUANTS_DIR}/sim_search_results.npy", allow_pickle=True)
    psi_all = np.array([results[i]["psi"] for i in range(iters)])
    psi = np.mean(psi_all, axis=0)
    psiSE = np.array([results[i]["psiSE"] for i in range(iters)])
    psiSE = np.mean(psiSE, axis=0)

    # Plot average estimate
    labs = ["Estimate", "$E[h(d|\\nu)]$", "Empirical"]
    xticklabs = ["1", "2", "3", "4"]
    colors = [Colors.BLUE, Colors.RED, Colors.BLACK]
    custom_plot(
        [h_obs[0] * psi / psi[0], h_obs[0] * h_str / h_str[0], h_obs],
        legendpos="lower left",
        legendlabs=labs,
        ylims=[0.2, 0.385],
        ydist=0.05,
        xlab="Time since unemployed",
        ylab="Hazard",
        figsize=[4.5, 3],
        xticklabs=xticklabs,
        colors=colors,
        savepath=f"{OUTPUT_DIR}/fig_sm_sim_avgpred.pdf",
    )

    # x and y to superimpose standard normal distribution
    x = np.linspace(-4, 4, 1000)
    y = np.exp(-0.5 * x**2) / (2 * np.pi) ** 0.5

    # Plot distribution of estimates
    figopts = {
        "bins": 25,
        "color": Colors.BLACK,
        "alpha": 0.1,
        "edgecolor": Colors.BLACK,
        "linewidth": 0.75,
        "stat": "density",
    }
    psi_nrm = (psi_all - psi_all.mean(axis=0)) / psi_all.std(axis=0)
    for t in range(T):
        plt.figure(figsize=(2.55, 2.25))
        plt.axvline(psi_nrm[:, t].mean(), color=Colors.RED, linewidth=1)
        sns.histplot(psi_nrm[:, t], **figopts)
        plt.plot(x, y, color=Colors.BLACK, linewidth=1)
        sns.despine()
        plt.xlim(-4, 4)
        plt.tight_layout(pad=0.75)
        plt.ylim(0, 0.52)
        plt.ylabel("")
        plt.savefig(f"{OUTPUT_DIR}/fig_sm_sim_dist{t+1}.pdf", dpi=300, format="pdf")

######################################################

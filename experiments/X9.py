"""
I want to know, whether the imitation process leads to equal return rates in both sectors.
Parameters that this could depend on are

1) the rate of exploration (random changes in opinion and rewiring),
2) also, the rate of rewiring could have an effect.

This should only work in the equilibrium condition where the environment stays constant.

"""

import getpass
import itertools as it
import os
import pickle as cp
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from random import uniform

from pysave.visualization.data_visualization \
    import plot_trajectories, plot_tau_smean,plot_tau_ymean
from pysave.model.model import SavingsCore_thebest_ext as Model
from pymofa.experiment_handling \
    import experiment_handling, even_time_series_spacing


def RUN_FUNC(tau, phi, eps, test, filename):
    """
    Set up the model for various parameters and determine
    which parts of the output are saved where.
    Output is saved in pickled dictionaries including the
    initial values, parameters and convergence state and time
    for each run.

    Parameters:
    -----------
    tau : float > 0
        the inverse rate of the social updates, i.e. mean waiting time
    phi : float \in [0, 1]
        the std of labor distribution, corresponds to sigma_L
    eps: float > 0
        modulo of the maximum imitation error, corresponds to gamma
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Parameters:
    input_params = {'phi': phi, 'tau': tau, 'd': 0.2,
                    'eps': eps, 'test': test,
                    'e_trajectory_output': True,
                    'm_trajectory_output': False}

    # building initial conditions

    # network:
    n = 100
    k = 4
    if test:
        n = 30
        k = 3

    while True:
        net = nx.watts_strogatz_graph(n, k, 0.4)
        # net = nx.complete_graph(n)
        if len(max(nx.connected_component_subgraphs(net), key=len).nodes()) == n:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # savings rates

    savings_rate = [uniform(0, 1) for i in range(n)]

    init_conditions = (adjacency_matrix, savings_rate)

    t_1 = 2000 *tau

    # initializing the model
    m = Model(*init_conditions, **input_params)

    # storing initial conditions and parameters
    res = {
        "parameters": pd.Series({"tau": m.tau,
                                 "phi": m.phi,
                                 "n": m.n,
                                 "P": m.P,
                                 "capital depreciation rate": m.d,
                                 "pi": m.pi,
                                 "kappa": m.kappa,
                                 "epsilon": m.eps})}

    # start timer
    t_start = time.clock()

    # run model with abundant resource
    t_max = t_1 if not test else 1
    exit_status = m.run(t_max=t_max)

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run

    if exit_status in [0, 1] or test:
        # even and safe macro trajectory
        res["trajectory"]= m.get_e_trajectory()
        # save micro data
        res["adjacency"] = m.neighbors
        res["final state"] = pd.DataFrame(data=np.array([m.savings_rate,
                                                         m.capital,
                                                         m.income, m.P, m.consumption
                                                         ]).transpose(),
                                          columns=['s', 'k', 'i', 'L', 'C'
                                                   ])
        # compute national savings rate and save
        res["savings_rate"] = sum(m.income * m.savings_rate) / sum(m.income)
        res["macro"] = [m.r, m.w, m.Y,m.tau]




    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        np.load(filename)["savings_rate"]
    except IOError:
        print("writing results failed for " + filename)

    return exit_status


def run_experiment(argv):
    """
    Take arv input variables and run sub_experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test, mode],
    2)  set output folders according to switches,
    3)  generate parameter combinations,
    4)  define names and dictionaries of callables to apply to sub_experiment
        data for post processing,
    5)  run computation and/or post processing and/or plotting
        depending on execution on cluster or locally or depending on
        experimentation mode.

    Parameters
    ----------
    argv: list[N]
        List of parameters from terminal input

    Returns
    -------
    rt: int
        some return value to show whether sub_experiment succeeded
        return 1 if successful.
    """

    """
    Get switches from input line in order of
    [test, mode]
    """

    # switch testing mode

    if len(argv) > 1:
        test = bool(int(argv[1]))
    else:
        test = False
    # switch sub_experiment mode
    if len(argv) > 2:
        mode = int(argv[2])
    else:
        mode = 0

    """
    set input/output paths
    """
    respath = os.path.dirname(os.path.realpath(__file__)) + "/output_data"
    if getpass.getuser() == "yuki":
        tmppath = respath
    elif getpass.getuser() == "asano":
        tmppath = "/p/tmp/asano/Savings_Experiments"
    else:
        tmppath = "./"

    folder = 'X9'

    # make sure, testing output goes to its own folder:

    test_folder = ['', 'test_output/'][int(test)]

    # check if cluster or local and set paths accordingly
    save_path_raw = \
        "{}/{}{}/" \
        .format(tmppath, test_folder, folder)
    save_path_res = \
        "{}/{}{}/" \
        .format(respath, test_folder, folder)
    """
    create parameter combinations and index
    """

    taus = [round(x, 5) for x in [7,8,9,10,11,12,13,14,15]]
    phis = [0.01]
    epss = [0.01] # [round(0.01, 5)]
    tau, phi, eps = [1., 10., 100.], [0], [0]

    if test:
        param_combs = list(it.product(tau, phi, eps, [test]))
    else:
        param_combs = list(it.product(taus, phis, epss, [test]))

    index = {0: "tau", 1: "phi", 2: "eps"}

    """
    create names and dicts of callables for post processing
    """

    name = 'parameter_scan'
    name6 = name + '_corr30'
    def x6(f):
        """
        computes the mean 30-year correlation coefficient (corr_30y) for one simulation

        Parameters:
            f: name of the inputfile
        Returns:
            dict{tau, corr_30y}
        """
        try:
            traj = np.load(f)["trajectory"]['capital']
            print f
            if f[36:37] == "1":
                tau = int(f[36:38])
            else:
                tau = int(f[36:37])
            #K = np.zeros(shape=(len(traj.index), 100))
            #for a, t in enumerate(traj.index):
            #    K[a, :] = traj['capital'][t]
            df = pd.DataFrame(traj.apply(pd.Series), index=traj.index)
            dfk = even_time_series_spacing(df, 2000 * tau)
            corrs = []
            # start, stop = 0, 2000*30
            for agent in range(100):
                x = dfk[agent].values[:-1]  # K[start:stop,agent]
                t = 30  # 300
                corrs.append(np.corrcoef(np.array([x[0:len(x) - t], x[t:len(x)]]))[0, 1])
            return {tau: np.mean(corrs)}
        except:
            return 0
    eva6 = {"capital_corr":
               lambda fnames: [x6(f)
                               for f in fnames]
           }

    """
    run computation and/or post processing and/or plotting
    """

    # cluster mode: computation and post processing
    if mode == 0:
        sample_size = 200 if not test else 2

        handle = experiment_handling(sample_size, param_combs, index,
                                     save_path_raw, save_path_res)
        handle.compute(RUN_FUNC)
        #handle.resave(eva6, name6)
        return 1
    # local mode: plotting only
    if mode == 1:
        sample_size = 200 if not test else 2

        handle = experiment_handling(sample_size, param_combs, index,
                                     save_path_raw, save_path_res)

        handle.resave(eva6, name6)
        return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)

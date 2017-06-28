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
    import plot_trajectories, plot_amsterdam
from pysave.model.model import SavingsCore as Model
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
        the rate of the social updates
    phi : float \in [0, 1]
        the rewirking probability
    eps: float > 0
        the rate of random events in the social update process
    test: int \in [0,1]
        whether this is a test run, e.g.
        can be executed with lower runtime
    filename: string
        filename for the results of the run
    """

    # Make different types of decision makers. Cues are

    # Parameters:

    input_params = {'phi': phi, 'tau': tau,
                    'eps': eps, 'test': test}

    # building initial conditions

    # network:
    n = 30
    k = 3
    if test:
        n = 30
        k = 3

    p = float(k) / n
    while True:
        net = nx.erdos_renyi_graph(n, p)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # opinions and investment

    savings_rate = [uniform(0, 1) for i in range(n)]

    init_conditions = (adjacency_matrix, savings_rate)

    t_1 = 1000

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
    m.R_depletion = False
    exit_status = m.run(t_max=t_max)

    res["runtime"] = time.clock() - t_start

    # store data in case of successful run

    if exit_status in [0, 1] or test:
        # even and safe macro trajectory
        res["trajectory"] = \
            even_time_series_spacing(m.get_e_trajectory(), 401, 0., t_max)
        # save micro data
        res["adjacency"] = m.neighbors
        res["final state"] = pd.DataFrame(data=np.array([m.savings_rate,
                                                m.capital,
                                                m.income]).transpose(),
                                          columns=['s', 'k', 'i'])
        # find connected components and their size
        g = nx.from_numpy_matrix(m.neighbors)
        cc = sorted(nx.connected_components(g), key=len)
        cluster_sizes = []
        for l in cc:
            cs = 0
            for n in l:
                cs += 1
            cluster_sizes.append(cs)
        res["cluster sizes"] = pd.DataFrame(data=cluster_sizes, columns=['cluster sizes'])
        # compute welfare and save

        def gini(x):
            # (Warning: This is a concise implementation, but it is O(n**2)
            # in time and memory, where n = len(x).  *Don't* pass in huge
            # samples!)

            # Mean absolute difference
            mad = np.abs(np.subtract.outer(x, x)).mean()
            # Relative mean absolute difference
            rmad = mad / np.mean(x)
            # Gini coefficient
            g = 0.5 * rmad
            return g

        res["welfare"] = np.mean(m.income) * (1. - gini(m.income))

        # compute national savings rate and save
        res["savings_rate"] = sum(m.income * m.savings_rate) / sum(m.income)




    # save data
    with open(filename, 'wb') as dumpfile:
        cp.dump(res, dumpfile)
    try:
        np.load(filename)["welfare"]
    except IOError:
        print("writing results failed for " + filename)

    return exit_status


def run_experiment(argv):
    """
    Take arv input variables and run sub_experiment accordingly.
    This happens in five steps:
    1)  parse input arguments to set switches
        for [test, mode, ffh/av, equi/trans],
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
        return 1 if sucessfull.
    """

    """
    Get switches from input line in order of
    [test, mode, ffh on/of, equi/transition]
    """

    # switch testing mode
    print argv
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

    folder = 'X1'

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

    taus = [round(x, 5) for x in list(np.linspace(1., 100., 2))]
    phis = [round(x, 5) for x in list(np.linspace(0., 1., 2))]
    epss = [round(x, 5) for x in list(np.linspace(0., 0.01, 2))]
    tau, phi, eps = [1., 10., 100.], [.1, .5, .9], [0., .01]

    if test:
        param_combs = list(it.product(tau, phi, eps, [test]))
    else:
        param_combs = list(it.product(taus, phis, epss, [test]))

    index = {0: "tau", 1: "phi", 2: "eps"}

    """
    create names and dicts of callables for post processing
    """

    name = 'parameter_scan'

    name1 = name + '_trajectory'
    eva1 = {"mean_trajectory":
            lambda fnames: pd.concat([np.load(f)["trajectory"]
                                      for f in fnames]).groupby(
                    level=0).mean(),
            "sem_trajectory":
            lambda fnames: pd.concat([np.load(f)["trajectory"]
                                      for f in fnames]).groupby(
                    level=0).std()
            }

    name2 = name + '_convergence'
    eva2 = {'welfare_mean':
            lambda fnames: np.nanmean([np.load(f)["welfare"]
                                       for f in fnames]),
            'savings_rate_mean':
            lambda fnames: np.nanmean([np.load(f)["savings_rate"]
                                       for f in fnames]),
            'welfare_std':
            lambda fnames: np.std([np.load(f)["welfare"]
                                   for f in fnames]),
            'savings_rate_std':
            lambda fnames: np.std([np.load(f)["savings_rate"]
                                   for f in fnames])
            }

    name3 = name + '_cluster_sizes'
    cf3 = {'cluster sizes':
           lambda fnames: pd.concat([np.load(f)["cluster sizes"]
                                     for f in fnames]).sortlevel(level=0).reset_index()
           }

    """
    run computation and/or post processing and/or plotting
    """

    # cluster mode: computation and post processing
    if mode == 0:
        sample_size = 2 if not test else 2

        handle = experiment_handling(sample_size, param_combs, index,
                                     save_path_raw, save_path_res)
        handle.compute(RUN_FUNC)
        handle.resave(eva1, name1)
        handle.resave(eva2, name2)plot_trajectories(save_path_res, name1, None, None)

        return 1

    handle.resave(cf3, name3)
    print 'now do trajectories'
    #

    # local mode: plotting only
    if mode == 1:

        plot_trajectories(save_path_res, name1, None, None)

        return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv#, 0,0]
    run_experiment(cmdline_arguments)

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
                    'eps': eps, 'test': test,
                    'e_trajectory_output': False,
                    'm_trajectory_output': False,
                    'pi': 0.5}

    # building initial conditions

    # network:
    n = 100
    k = 3
    if test:
        n = 30
        k = 3

    while True:
        net = nx.barabasi_albert_graph(n, k)
        #net = nx.complete_graph(n)
        if len(list(net)) > 1:
            break
    adjacency_matrix = nx.adj_matrix(net).toarray()

    # opinions and investment

    savings_rate = [uniform(0, 1) for i in range(n)]

    init_conditions = (adjacency_matrix, savings_rate)

    t_1 = 5000 * tau

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
        #res["trajectory"] = \
        #    even_time_series_spacing(m.get_e_trajectory(), 401, 0., t_max)
        # save micro data
        res["adjacency"] = m.neighbors
        res["final state"] = pd.DataFrame(data=np.array([m.savings_rate,
                                                         m.capital,
                                                         m.income, m.P, m.consumption,
                                                         m.w, m.r, m.Y]).transpose(),
                                          columns=['s', 'k', 'i', 'L', 'C',
                                                   'w', 'r', 'Y'])

        # compute national savings rate and save
        res["savings_rate"] = sum(m.income * m.savings_rate) / sum(m.income)




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

    folder = 'X6_Ldistphi10_bara_eps01_q_longer'

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

    taus = [round(x, 5) for x in list(np.logspace(0, 4, 100))]
    phis = [0.10]
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

    name4 = name + '_all_si'
    eva4 = {"all_si":
                lambda fnames: [np.load(f)["final state"]
                                          for f in fnames]
            }
    name5 = name + 'nat_sav'
    eva5 = {"nat_sav":
                lambda fnames: [np.load(f)["savings_rate"]
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
        handle.resave(eva4, name4)
        handle.resave(eva5, name5)
        return 1
    # local mode: plotting only
    if mode == 1:
        sample_size = 100 if not test else 2

        handle = experiment_handling(sample_size, param_combs, index,
                                     save_path_raw, save_path_res)

        handle.resave(eva4, name4)
        #handle.resave(cf3, name3)
        #plot_trajectories(save_path_res, name1, None, None)
        #print save_path_res, name1
        #plot_tau_smean(save_path_res, name1, None, None)
        #plot_tau_ymean(save_path_res, name1, None, None)

        return 1


if __name__ == "__main__":
    cmdline_arguments = sys.argv
    run_experiment(cmdline_arguments)


import datetime
import numpy as np
import pandas as pd
from itertools import chain
from scipy.integrate import odeint
from scipy.sparse.csgraph import connected_components
from random import shuffle, uniform


class SavingsCore_exploit:

    def __init__(self, adjacency=None, savings_rate=None,
                 capital=None,
                 tau=0.8, phi=0.0, eps=0.05,
                 P=1., b=1., d=0.06, pi=1./2., r_b=0,
                 test=False):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        self.mode = 2

        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = True
        self.m_trajectory_output = True
        # toggle whether to run full time or only until consensus
        self.run_full_time = True
        self.epsilon = np.finfo(dtype='float')

        # General Variables

        # System Time
        self.t = 0.
        # Step counter for output
        self.steps = 0
        # eps == 0: 0 for no consensus, 1 consensus
        # eps>0 0 for no convergence, 1 for convergence at t_max
        self.consensus = False
        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float('NaN')
        # eps==0: -1 for no consensus, 1 for clean consensus,
        # 0 for dirty consensus, in between for fragmentation
        # eps>0: if converged: opinion state at time of convergence
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # rewiring probability for adaptive voter model
        self.phi = phi
        # percentage of rewiring and imitation events that are noise
        self.eps = eps

        # number of households
        self.n = adjacency.shape[0]

        # Household variables

        # Individual

        # waiting times between rewiring events for each household
        self.waiting_times = \
            np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = adjacency
        # investment_decisions as indices of possible_opinions
        self.savings_rate = np.array(savings_rate)

        # members of ALL household = population   
        self.P = P

        # household capital in clean capital
        if capital is None:
            self.capital = np.ones(self.n)
        else:
            self.capital = capital

        # household income (for social update)
        self.income = np.zeros(self.n)

        # for Cobb Douglas economics:
        # Solow residual
        self.b = b
        # labor elasticity
        self.pi = pi
        # capital elasticity
        self.kappa = 1. - self.pi
        # capital depreciation rate
        self.d = d
        # population growth rate
        self.r_b = r_b

        # total capital (supply)
        self.K = sum(self.capital)
        # total labor (supply)
        self.P = P
        # Production
        self.Y = 0.
        # wage
        self.w = 0.
        # capital rent
        self.r = 0.

        if self.e_trajectory_output:
            self.init_e_trajectory()

        self.s_trajectory = pd.DataFrame(columns=range(self.n))

    def run(self, t_max=200.):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus/convergence reached
            if exit_status == 0: no consensus/convergence reached at t=t_max
            if exit_status ==-1: no consensus, no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """
        candidate = 0
        while self.t < t_max:
            # 1 find update candidate and respective update time
            (candidate, neighbor,
             neighbors, update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            self.update_economy(update_time)

            # 3 update opinion formation in case,
            # update candidate was found:
            if candidate >= 0:
                self.update_opinion_formation(candidate,
                                              neighbor, neighbors)

            if not self.run_full_time and self.converged:
                break

        # save final state to dictionary
        self.final_state = {
                'adjacency': self.neighbors,
                'savings_rate': self.savings_rate,
                'capital': self.capital,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'b': self.b, 'd': self.d,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached 
        elif not self.converged:
            self.convergence_state = float('nan')
            self.convergence_time = self.t
            return 0        # no consensus found during run time
        elif candidate == -2:
            return -1       # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2       # bad run - economy broken
        else:
            return -3       # very bad run. Investigations needed

    def economy_dot(self, x0, t):
        """
        economic model assuming Cobb-Douglas production:

            Y = b P^pi K^kappa

        and no profits:

            Y - w P - r K = 0,

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            N + 1. First N entries are
            household capital [0:n],
            the last entry is total population.
        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            N + 1. First N entries are changes
            household capital [n:2N],
            the last entry is the change in total population
        """

        capital = np.where(x0[0:self.n] > 0,
                           x0[0:self.n],
                           np.full(self.n, self.epsilon.eps))

        P = x0[-1]
        K = sum(capital)

        assert K >= 0, 'negative clean capital'

        self.w = self.b * self.pi * P ** (self.pi - 1) * K ** self.kappa
        self.r = self.b * self.kappa * P ** self.pi * K ** (self.kappa - 1)

        self.K = K
        self.P = P

        self.income = (self.r * self.capital + self.w * P / self.n)

        assert all(self.income > 0), \
            'income is negative, K: {} \n income: \n {}'.format(K, self.income)

        P_dot = self.r_b * P

        capital_dot = \
            self.savings_rate * self.income - self.capital * self.d

        return list(capital_dot) + [P_dot]

    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = list(self.capital) + [self.P]

        # integrate the system
        x1 = odeint(self.economy_dot, x0, dt, mxhnil=1)[1]

        self.capital = np.where(x1[0:self.n] > 0,
                                x1[0:self.n], np.zeros(self.n))
        self.P = x1[-1]

        self.t = update_time
        self.steps += 1

        # calculate economic output:
        self.Y = self.b * self.K ** self.kappa * self.P ** self.pi
        self.consumption = self.income * (1 - self.savings_rate)

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()

    def find_update_candidates(self):

        j = 0
        i_max = 1000 * self.n
        candidate = 0
        neighbor = self.n
        neighbors = []
        update_time = self.t

        while j < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]

            # NOISE in imitation (exploration of strategies)
            # people trying new stuff at random
            rdn = np.random.uniform()
            if rdn < self.eps*(1-self.phi):
                self.savings_rate[candidate] = uniform(0, 1)
                candidate = -1
                break
            # noise in rewiring (sometimes they make new friends
            # at random..)
            elif rdn > 1.-self.eps * self.phi and len(neighbors) > 0:
                unconnected = np.zeros(self.n, dtype=int)
                for j in range(self.n):
                    if j not in neighbors:
                        unconnected[j] = 1
                unconnected = unconnected.nonzero()[0]
                if len(unconnected) > 0:
                    old_neighbor = np.random.choice(neighbors)
                    new_neighbor = np.random.choice(unconnected)
                    self.neighbors[candidate, old_neighbor] = \
                        self.neighbors[old_neighbor, candidate] = 0
                    self.neighbors[candidate, new_neighbor] = \
                        self.neighbors[new_neighbor, candidate] = 1
                candidate = -1
                break

            # if candidate has neighbors, chose one at random.
            if len(neighbors) > 0:
                neighbor = np.random.choice(neighbors)

                # check if preferences of candidate and random
                # neighbor differ
                if self.savings_rate[candidate] == self.savings_rate[neighbor]:

                    # if candidate and neighbor have same preferences, they
                    # not suitable for update. (RETRY)
                    neighbor = self.n

            if neighbor < self.n:
                # update candidate found (GOD)
                break
            else:
                j += 1
                if j % self.n == 0:
                    if self.detect_consensus_state(self.savings_rate):
                        # no update candidate found because of
                        # consensus state (GOD)
                        candidate = -1
                        break
            if j >= i_max:
                # no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor,
                                 neighbors):

        same_unconnected = np.zeros(self.n, dtype=int)
        p = []
        sc = sum(self.capital)
        opinion = self.savings_rate

        # adapt or rewire?
        if (self.phi == 1 or (self.phi != 1
                              and np.random.uniform() < self.phi)):
            # if rewire
            for j in range(self.n):

                # households rewire to households with same
                # savings rate. Improvement: preferential attachment
                # probability to connect decays exponentially with
                # difference in opinion.
                if j not in neighbors and j != candidate:
                    same_unconnected[j] = 1
                    p.append(np.exp(- abs(self.capital[j]
                                          - self.capital[candidate])
                                    ))
            same_unconnected = same_unconnected.nonzero()[0]
            sp = sum(p)
            p = [val/sp for val in p]
            if len(same_unconnected) > 0:
                new_neighbor = np.random.choice(same_unconnected, p=p)
                self.neighbors[candidate, neighbor] = \
                    self.neighbors[neighbor, candidate] = 0
                self.neighbors[candidate, new_neighbor] = \
                    self.neighbors[new_neighbor, candidate] = 1
        else:
            # if adapt
            # compare fitness
            df = self.fitness(neighbor) - self.fitness(candidate)
            # and imitate depending on fitness difference
            if np.random.uniform() < .5*(np.tanh(df) + 1):
                self.savings_rate[candidate] = abs(self.savings_rate[neighbor]
                                                   + self.savings_rate[candidate]) / 2.
                # self.savings_rate[candidate] = self.savings_rate[neighbor]
        return 0

    def detect_consensus_state(self, d_opinions):
        # check if network is split in components with
        # same investment_decisions/preferences
        # returns 1 if consensus state is detected,
        # returns 0 if NO consensus state is detected.

        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(d_opinions[c])) == 1
                             for c in ((cc == u).nonzero()[0]
                             for u in np.unique(cc)))
        if self.eps == 0:
            if self.consensus and self.convergence_state == -1:
                self.convergence_state = np.mean(d_opinions)
                self.convergence_time = self.t
                self.converged = True

        return self.converged

    def fitness(self, agent):
        return self.income[agent] * (1 - self.savings_rate[agent])

    def init_e_trajectory(self):
        element = ['time',
                   'wage',
                   'r',
                   's',
                   'capital share of gdp',
                   'labor share of gdp',
                   'C',
                   'P',
                   'Y',
                   'consensus'
                   ]
        self.e_trajectory.append(element)

        self.w = self.b * self.pi * self.P ** (self.pi - 1) * self.K ** self.kappa
        self.r = self.b * self.kappa * self.P ** self.pi * self.K ** (self.kappa - 1)

        self.income = (self.r * self.capital + self.w * self.P / self.n)

        self.update_e_trajectory()

    def update_e_trajectory(self):
        element = [self.t,
                   self.w,
                   self.r,
                   sum(self.savings_rate * self.income) / sum(self.income),
                   sum(self.capital) * self.r,
                   self.P * self.w,
                   sum(self.income * (1 - self.savings_rate)),
                   self.P,
                   self.Y,
                   self.converged,
                   ]
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory.pop(0)
        trj = pd.DataFrame(self.e_trajectory, columns=columns)
        trj = trj.set_index('time')

        return trj


class SavingsCore_voter:

    def __init__(self, adjacency=None, savings_rate=None,
                 capital=None,
                 tau=0.8, phi=0.0, eps=0.00,
                 P=1., b=1., d=0.06, pi=1./2., r_b=0,
                 test=False):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        self.mode = 2

        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = True
        self.m_trajectory_output = True
        # toggle whether to run full time or only until consensus
        self.run_full_time = True
        self.epsilon = np.finfo(dtype='float')

        # General Variables

        # System Time
        self.t = 0.
        # Step counter for output
        self.steps = 0
        # eps == 0: 0 for no consensus, 1 consensus
        # eps>0 0 for no convergence, 1 for convergence at t_max
        self.consensus = False
        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float('NaN')
        # eps==0: -1 for no consensus, 1 for clean consensus,
        # 0 for dirty consensus, in between for fragmentation
        # eps>0: if converged: opinion state at time of convergence
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # rewiring probability for adaptive voter model
        self.phi = phi
        # percentage of rewiring and imitation events that are noise
        self.eps = eps

        # number of households
        self.n = adjacency.shape[0]

        # Household variables

        # Individual

        # waiting times between rewiring events for each household
        self.waiting_times = \
            np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = adjacency
        # investment_decisions as indices of possible_opinions
        self.savings_rate = np.array(savings_rate)

        # members of ALL household = population
        self.P = P

        # household capital in clean capital
        if capital is None:
            self.capital = np.ones(self.n)
        else:
            self.capital = capital

        # household income (for social update)
        self.income = np.zeros(self.n)

        # for Cobb Douglas economics:
        # Solow residual
        self.b = b
        # labor elasticity
        self.pi = pi
        # capital elasticity
        self.kappa = 1. - self.pi
        # capital depreciation rate
        self.d = d
        # population growth rate
        self.r_b = r_b

        # total capital (supply)
        self.K = sum(self.capital)
        # total labor (supply)
        self.P = P
        # Production
        self.Y = 0.
        # wage
        self.w = 0.
        # capital rent
        self.r = 0.

        if self.e_trajectory_output:
            self.init_e_trajectory()

        self.s_trajectory = pd.DataFrame(columns=range(self.n))

    def run(self, t_max=200.):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus/convergence reached
            if exit_status == 0: no consensus/convergence reached at t=t_max
            if exit_status ==-1: no consensus, no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """
        candidate = 0
        while self.t < t_max:
            # 1 find update candidate and respective update time
            (candidate, neighbor,
             neighbors, update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            self.update_economy(update_time)

            # 3 update opinion formation in case,
            # update candidate was found:
            if candidate >= 0:
                self.update_opinion_formation(candidate,
                                              neighbor, neighbors)

            if not self.run_full_time and self.converged:
                break

        # save final state to dictionary
        self.final_state = {
                'adjacency': self.neighbors,
                'savings_rate': self.savings_rate,
                'capital': self.capital,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'b': self.b, 'd': self.d,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached
        elif not self.converged:
            self.convergence_state = float('nan')
            self.convergence_time = self.t
            return 0        # no consensus found during run time
        elif candidate == -2:
            return -1       # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2       # bad run - economy broken
        else:
            return -3       # very bad run. Investigations needed

    def economy_dot(self, x0, t):
        """
        economic model assuming Cobb-Douglas production:

            Y = b P^pi K^kappa

        and no profits:

            Y - w P - r K = 0,

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            N + 1. First N entries are
            household capital [0:n],
            the last entry is total population.
        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            N + 1. First N entries are changes
            household capital [n:2N],
            the last entry is the change in total population
        """

        capital = np.where(x0[0:self.n] > 0,
                           x0[0:self.n],
                           np.full(self.n, self.epsilon.eps))

        P = x0[-1]
        K = sum(capital)

        assert K >= 0, 'negative clean capital'

        self.w = self.b * self.pi * P ** (self.pi - 1) * K ** self.kappa
        self.r = self.b * self.kappa * P ** self.pi * K ** (self.kappa - 1)

        self.K = K
        self.P = P

        self.income = (self.r * self.capital + self.w * P / self.n)

        assert all(self.income > 0), \
            'income is negative, K: {} \n income: \n {}'.format(K, self.income)

        P_dot = self.r_b * P

        capital_dot = \
            self.savings_rate * self.income - self.capital * self.d

        return list(capital_dot) + [P_dot]

    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = list(self.capital) + [self.P]

        # integrate the system
        x1 = odeint(self.economy_dot, x0, dt, mxhnil=1)[1]

        self.capital = np.where(x1[0:self.n] > 0,
                                x1[0:self.n], np.zeros(self.n))
        self.P = x1[-1]

        self.t = update_time
        self.steps += 1

        # calculate economic output:
        self.Y = self.b * self.K ** self.kappa * self.P ** self.pi
        self.consumption = self.income * (1 - self.savings_rate)

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()

    def find_update_candidates(self):

        j = 0
        i_max = 1000 * self.n
        candidate = 0
        neighbor = self.n
        neighbors = []
        update_time = self.t

        while j < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]


            # choose random neighbor of candidate
            if len(neighbors) > 0:
                neighbor = np.random.choice(neighbors)

            if neighbor < self.n:
                # update candidate found (GOOD)
                break
            else:
                j += 1
                if j % self.n == 0:
                    if self.detect_consensus_state(self.savings_rate):
                        # no update candidate found because of
                        # consensus state (GOD)
                        candidate = -1
                        break
            if j >= i_max:
                # no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor,
                                 neighbors):
        self.savings_rate[candidate] = self.savings_rate[neighbor]
        return 0

    def detect_consensus_state(self, d_opinions):
        # check if network is split in components with
        # same investment_decisions/preferences
        # returns 1 if consensus state is detected,
        # returns 0 if NO consensus state is detected.

        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(d_opinions[c])) == 1
                             for c in ((cc == u).nonzero()[0]
                             for u in np.unique(cc)))
        if self.eps == 0:
            if self.consensus and self.convergence_state == -1:
                self.convergence_state = np.mean(d_opinions)
                self.convergence_time = self.t
                self.converged = True

        return self.converged

    def fitness(self, agent):
        return self.income[agent] * (1 - self.savings_rate[agent])

    def init_e_trajectory(self):
        element = ['time',
                   'wage',
                   'r',
                   's',
                   'capital share of gdp',
                   'labor share of gdp',
                   'C',
                   'P',
                   'Y',
                   'consensus'
                   ]
        self.e_trajectory.append(element)

        self.w = self.b * self.pi * self.P ** (self.pi - 1) * self.K ** self.kappa
        self.r = self.b * self.kappa * self.P ** self.pi * self.K ** (self.kappa - 1)

        self.income = (self.r * self.capital + self.w * self.P / self.n)

        self.update_e_trajectory()

    def update_e_trajectory(self):
        element = [self.t,
                   self.w,
                   self.r,
                   sum(self.savings_rate * self.income) / sum(self.income), # weighted avg savingsrate
                   sum(self.capital) * self.r,
                   self.P * self.w,
                   sum(self.income * (1 - self.savings_rate)),
                   self.P,
                   self.Y,
                   self.converged,
                   ]
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory.pop(0)
        trj = pd.DataFrame(self.e_trajectory, columns=columns)
        trj = trj.set_index('time')

        return trj


class SavingsCore_thebest:

    def __init__(self, adjacency=None, savings_rate=None,
                 capital=None,
                 tau=0.8, phi=0.0, eps=0.00,
                 P=1., b=1., d=0.06, pi=1./2., r_b=0,
                 test=False):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        self.mode = 2

        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = True
        self.m_trajectory_output = True
        # toggle whether to run full time or only until consensus
        self.run_full_time = True
        self.epsilon = np.finfo(dtype='float')

        # General Variables

        # System Time
        self.t = 0.
        # Step counter for output
        self.steps = 0
        # eps == 0: 0 for no consensus, 1 consensus
        # eps>0 0 for no convergence, 1 for convergence at t_max
        self.consensus = False
        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float('NaN')
        # eps==0: -1 for no consensus, 1 for clean consensus,
        # 0 for dirty consensus, in between for fragmentation
        # eps>0: if converged: opinion state at time of convergence
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # rewiring probability for adaptive voter model
        self.phi = phi
        # percentage of rewiring and imitation events that are noise
        self.eps = eps

        # number of households
        self.n = adjacency.shape[0]

        # Household variables

        # Individual

        # waiting times between rewiring events for each household
        self.waiting_times = \
            np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = adjacency
        # investment_decisions as indices of possible_opinions
        self.savings_rate = np.array(savings_rate)

        # members of ALL household = population
        self.P = P

        # household capital in clean capital
        if capital is None:
            self.capital = np.ones(self.n)
        else:
            self.capital = capital

        # household income (for social update)
        self.income = np.zeros(self.n)

        # for Cobb Douglas economics:
        # Solow residual
        self.b = b
        # labor elasticity
        self.pi = pi
        # capital elasticity
        self.kappa = 1. - self.pi
        # capital depreciation rate
        self.d = d
        # population growth rate
        self.r_b = r_b

        # total capital (supply)
        self.K = sum(self.capital)
        # total labor (supply)
        self.P = P
        # Production
        self.Y = 0.
        # wage
        self.w = 0.
        # capital rent
        self.r = 0.

        if self.e_trajectory_output:
            self.init_e_trajectory()

        self.s_trajectory = pd.DataFrame(columns=range(self.n))

    def run(self, t_max=200.):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus/convergence reached
            if exit_status == 0: no consensus/convergence reached at t=t_max
            if exit_status ==-1: no consensus, no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """
        candidate = 0
        while self.t < t_max:
            # 1 find update candidate and respective update time
            (candidate, neighbor,
             neighbors, update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            self.update_economy(update_time)

            # 3 update opinion formation in case,
            # update candidate was found:
            if candidate >= 0:
                self.update_opinion_formation(candidate,
                                              neighbor, neighbors)

            if not self.run_full_time and self.converged:
                break

        # save final state to dictionary
        self.final_state = {
                'adjacency': self.neighbors,
                'savings_rate': self.savings_rate,
                'capital': self.capital,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'b': self.b, 'd': self.d,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached
        elif not self.converged:
            self.convergence_state = float('nan')
            self.convergence_time = self.t
            return 0        # no consensus found during run time
        elif candidate == -2:
            return -1       # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2       # bad run - economy broken
        else:
            return -3       # very bad run. Investigations needed

    def economy_dot(self, x0, t):
        """
        economic model assuming Cobb-Douglas production:

            Y = b P^pi K^kappa

        and no profits:

            Y - w P - r K = 0,

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            N + 1. First N entries are
            household capital [0:n],
            the last entry is total population.
        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            N + 1. First N entries are changes
            household capital [n:2N],
            the last entry is the change in total population
        """

        capital = np.where(x0[0:self.n] > 0,
                           x0[0:self.n],
                           np.full(self.n, self.epsilon.eps))

        P = x0[-1]
        K = sum(capital)

        assert K >= 0, 'negative clean capital'

        self.w = self.b * self.pi * P ** (self.pi - 1) * K ** self.kappa
        self.r = self.b * self.kappa * P ** self.pi * K ** (self.kappa - 1)

        self.K = K
        self.P = P

        self.income = (self.r * self.capital + self.w * P / self.n)

        assert all(self.income > 0), \
            'income is negative, K: {} \n income: \n {}'.format(K, self.income)

        P_dot = self.r_b * P

        capital_dot = \
            self.savings_rate * self.income - self.capital * self.d

        return list(capital_dot) + [P_dot]

    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = list(self.capital) + [self.P]

        # integrate the system
        x1 = odeint(self.economy_dot, x0, dt, mxhnil=1)[1]

        self.capital = np.where(x1[0:self.n] > 0,
                                x1[0:self.n], np.zeros(self.n))
        self.P = x1[-1]

        self.t = update_time
        self.steps += 1

        # calculate economic output:
        self.Y = self.b * self.K ** self.kappa * self.P ** self.pi
        self.consumption = self.income * (1 - self.savings_rate)

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()

    def find_update_candidates(self):

        j = 0
        i_max = 1000 * self.n
        candidate = 0
        neighbor = self.n
        neighbors = []
        update_time = self.t

        while j < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]


            # choose random neighbor of candidate
            if len(neighbors) > 0:
                func_vals = (1. - self.savings_rate[neighbors]) *\
                                       self.income[neighbors]
                neighbor = neighbors[np.argmax(func_vals)]
            if neighbor < self.n:
                # update candidate found (GOOD)
                break
            else:
                j += 1
                if j % self.n == 0:
                    if self.detect_consensus_state(self.savings_rate):
                        # no update candidate found because of
                        # consensus state (GOD)
                        candidate = -1
                        break
            if j >= i_max:
                # no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor,
                                 neighbors):
        if self.fitness(neighbor)>self.fitness(candidate):
            self.savings_rate[candidate] = self.savings_rate[neighbor]
        return 0

    def detect_consensus_state(self, d_opinions):
        # check if network is split in components with
        # same investment_decisions/preferences
        # returns 1 if consensus state is detected,
        # returns 0 if NO consensus state is detected.

        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(d_opinions[c])) == 1
                             for c in ((cc == u).nonzero()[0]
                             for u in np.unique(cc)))
        if self.eps == 0:
            if self.consensus and self.convergence_state == -1:
                self.convergence_state = np.mean(d_opinions)
                self.convergence_time = self.t
                self.converged = True

        return self.converged

    def fitness(self, agent):
        return self.income[agent] * (1 - self.savings_rate[agent])

    def init_e_trajectory(self):
        element = ['time',
                   'wage',
                   'r',
                   's',
                   'capital share of gdp',
                   'labor share of gdp',
                   'C',
                   'P',
                   'Y',
                   'consensus'
                   ]
        self.e_trajectory.append(element)

        self.w = self.b * self.pi * self.P ** (self.pi - 1) * self.K ** self.kappa
        self.r = self.b * self.kappa * self.P ** self.pi * self.K ** (self.kappa - 1)

        self.income = (self.r * self.capital + self.w * self.P / self.n)

        self.update_e_trajectory()

    def update_e_trajectory(self):
        element = [self.t,
                   self.w,
                   self.r,
                   sum(self.savings_rate * self.income) / sum(self.income), # weighted avg savingsrate
                   sum(self.capital) * self.r,
                   self.P * self.w,
                   sum(self.income * (1 - self.savings_rate)),
                   self.P,
                   self.Y,
                   self.converged,
                   ]
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory.pop(0)
        trj = pd.DataFrame(self.e_trajectory, columns=columns)
        trj = trj.set_index('time')

        return trj



class SavingsCore_thebest_ext:

    def __init__(self, adjacency=None, savings_rate=None,
                 capital=None,
                 tau=0.8, phi=0.0, eps=0.01,
                 P=1., b=1., d=0.06, pi=1./2., r_b=0,
                 test=False,e_trajectory_output=True,
                 m_trajectory_output=True):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        self.mode = 2

        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = e_trajectory_output
        self.m_trajectory_output = m_trajectory_output
        # toggle whether to run full time or only until consensus
        self.run_full_time = True

        self.epsilon = np.finfo(dtype='float')

        # General Variables

        # System Time
        self.t = 0.
        # Step counter for output
        self.steps = 0

        self.consensus = False
        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float('NaN')
        # eps==0: -1 for no consensus, 1 for clean consensus,
        # 0 for dirty consensus, in between for fragmentation
        # eps>0: if converged: opinion state at time of convergence
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # rewiring probability for adaptive voter model
        self.phi = phi
        # eps is noise, or imitation error
        self.eps = eps

        # number of households
        self.n = adjacency.shape[0]

        # Household variables

        # Individual

        # waiting times between rewiring events for each household
        self.waiting_times = \
            np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = adjacency
        # investment_decisions as indices of possible_opinions
        self.savings_rate = np.array(savings_rate)

        # household capital in clean capital
        if capital is None:
            self.capital = np.ones(self.n)
        else:
            self.capital = capital

        # household income (for social update)
        self.income = np.zeros(self.n)

        # for Cobb Douglas economics:
        # Solow residual
        self.b = b
        # labor elasticity
        self.pi = pi
        # capital elasticity
        self.kappa = 1. - self.pi
        # capital depreciation rate
        self.d = d
        # population growth rate
        self.r_b = r_b

        # total capital (supply)
        self.K = sum(self.capital)
        # total labor (supply)
        self.P = np.random.normal(float(P)/self.n, (float(P)/self.n)*self.phi, self.n)
        while any(self.P < 0):
            self.P = np.random.normal(float(P)/ self.n, (float(P) / self.n) * self.phi, self.n)
        # Production
        self.Y = 0.
        # wage
        self.w = 0.
        # capital rent
        self.r = 0.

        if self.e_trajectory_output:
            self.init_e_trajectory()

        self.s_trajectory = pd.DataFrame(columns=range(self.n))

    def run(self, t_max=200.):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus/convergence reached
            if exit_status == 0: no consensus/convergence reached at t=t_max
            if exit_status ==-1: no consensus, no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """
        candidate = 0
        while self.t < t_max:
            # 1 find update candidate and respective update time
            (candidate, neighbor,
             neighbors, update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            self.update_economy(update_time)

            # 3 update opinion formation in case,
            # update candidate was found:
            if candidate >= 0:
                self.update_opinion_formation(candidate,
                                              neighbor, neighbors)

            if not self.run_full_time and self.converged:
                break

        # save final state to dictionary
        self.final_state = {
                'adjacency': self.neighbors,
                'savings_rate': self.savings_rate,
                'capital': self.capital,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'b': self.b, 'd': self.d,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached
        elif not self.converged:
            self.convergence_state = float('nan')
            self.convergence_time = self.t
            return 0        # no consensus found during run time
        elif candidate == -2:
            return -1       # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2       # bad run - economy broken
        else:
            return -3       # very bad run. Investigations needed

    def economy_dot(self, x0, t):
        """
        economic model assuming Cobb-Douglas production:

            Y = b P^pi K^kappa

        and no profits:

            Y - w P - r K = 0,

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            N + 1. First N entries are
            household capital [0:n],
            the last entry is total population.
        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            N + 1. First N entries are changes
            household capital [n:2N],
            the last entry is the change in total population
        """

        capital = np.where(x0[0:self.n] > 0,
                           x0[0:self.n],
                           np.full(self.n, self.epsilon.eps))

        P = sum(self.P)
        K = sum(capital)

        assert K >= 0, 'negative capital'

        self.w = self.b * self.pi * P ** (self.pi - 1) * K ** self.kappa
        self.r = self.b * self.kappa * P ** self.pi * K ** (self.kappa - 1)

        self.K = K

        self.income = (self.r * self.capital + self.w * self.P)

        assert all(self.income > 0), \
            'income is negative, K: {} \n income: \n {}'.format(K, self.income)

        P_dot = self.r_b * P

        capital_dot = \
            self.savings_rate * self.income - self.capital * self.d

        return list(capital_dot) + [P_dot]

    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = list(self.capital) + [sum(self.P)]

        # integrate the system
        x1 = odeint(self.economy_dot, x0, dt, mxhnil=1)[1]

        self.capital = np.where(x1[0:self.n] > 0,
                                x1[0:self.n], np.zeros(self.n))
        #self.P = x1[-1]

        self.t = update_time
        self.steps += 1

        # calculate economic output:
        self.Y = self.b * self.K ** self.kappa * sum(self.P) ** self.pi
        self.consumption = self.income * (1 - self.savings_rate)

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()

    def find_update_candidates(self):

        j = 0
        i_max = 1000 * self.n
        candidate = 0
        neighbor = self.n
        neighbors = []
        update_time = self.t

        while j < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]


            # choose best neighbor of candidate
            if len(neighbors) > 0:
                func_vals = (1. - self.savings_rate[neighbors]) *\
                                       self.income[neighbors]
                neighbor = neighbors[np.argmax(func_vals)]

            if neighbor < self.n:
                # update candidate found (GOOD)
                break
            else:
                j += 1
                if j % self.n == 0:
                    if self.detect_consensus_state(self.savings_rate):
                        # no update candidate found because of
                        # consensus state (GOD)
                        candidate = -1
                        break
            if j >= i_max:
                # no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor,
                                 neighbors):
        if self.fitness(neighbor)>self.fitness(candidate):
            self.savings_rate[candidate] = self.savings_rate[neighbor] + np.random.uniform(-self.eps, self.eps)
            while (self.savings_rate[candidate] > 1) or (self.savings_rate[candidate] < 0):
                self.savings_rate[candidate] = self.savings_rate[neighbor] + np.random.uniform(-self.eps, self.eps)
        return 0

    def detect_consensus_state(self, d_opinions):
        # check if network is split in components with
        # same investment_decisions/preferences
        # returns 1 if consensus state is detected,
        # returns 0 if NO consensus state is detected.

        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(d_opinions[c])) == 1
                             for c in ((cc == u).nonzero()[0]
                             for u in np.unique(cc)))
        if self.eps == 0:
            if self.consensus and self.convergence_state == -1:
                self.convergence_state = np.mean(d_opinions)
                self.convergence_time = self.t
                self.converged = True

        return self.converged

    def fitness(self, agent):
        return self.income[agent] * (1 - self.savings_rate[agent])

    def init_e_trajectory(self):
        element = ['time',
                   'wage',
                   'r',
                   's',
                   'capital',
                   'labor share of gdp',
                   'C',
                   'P',
                   'Y',
                   'consensus'
                   ]
        self.e_trajectory.append(element)

        self.w = self.b * self.pi * sum(self.P) ** (self.pi - 1) * self.K ** self.kappa
        self.r = self.b * self.kappa * sum(self.P) ** self.pi * self.K ** (self.kappa - 1)

        self.income = (self.r * self.capital + self.w * self.P)

        self.update_e_trajectory()

    def update_e_trajectory(self):
        element = [self.t,
                   self.w,
                   self.r,
                   self.savings_rate.copy(),
                   self.capital.copy(),
                   sum(self.P) * self.w,
                   self.income.copy() * (1 - self.savings_rate.copy()),
                   self.P,
                   self.Y,
                   self.converged,
                   ]
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory.pop(0)
        trj = pd.DataFrame(self.e_trajectory, columns=columns)
        trj = trj.set_index('time')

        return trj



class SavingsCore_thebest_all:

    def __init__(self, adjacency=None, savings_rate_=None,
                 capital=None,
                 tau=0.8, phi=0.0, eps=0.00,
                 P=1., b=1., d=0.06, pi=1./2., r_b=0,
                 test=False):

        # Modes:
        #  1: only economy,
        #  2: economy + opinion formation + decision making,

        self.mode = 2

        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = True
        self.m_trajectory_output = True
        # toggle whether to run full time or only until consensus
        self.run_full_time = True
        self.epsilon = np.finfo(dtype='float')

        # General Variables

        # System Time
        self.t = 0.
        # Step counter for output
        self.steps = 0
        # eps == 0: 0 for no consensus, 1 consensus
        # eps>0 0 for no convergence, 1 for convergence at t_max
        self.consensus = False
        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float('NaN')
        # eps==0: -1 for no consensus, 1 for clean consensus,
        # 0 for dirty consensus, in between for fragmentation
        # eps>0: if converged: opinion state at time of convergence
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.m_trajectory = []
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # rewiring probability for adaptive voter model
        self.phi = phi
        # percentage of rewiring and imitation events that are noise
        self.eps = eps

        # number of households
        self.n = adjacency.shape[0]

        # Household variables

        # Individual

        # waiting times between rewiring events for each household
        self.waiting_times = \
            np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = adjacency
        # investment_decisions as indices of possible_opinions
        self.savings_rate = np.array(savings_rate_)

        # members of ALL household = population
        self.P = P

        # household capital in clean capital
        if capital is None:
            self.capital = np.ones(self.n)
        else:
            self.capital = capital

        # household income (for social update)
        self.income = np.zeros(self.n)

        # for Cobb Douglas economics:
        # Solow residual
        self.b = b
        # labor elasticity
        self.pi = pi
        # capital elasticity
        self.kappa = 1. - self.pi
        # capital depreciation rate
        self.d = d
        # population growth rate
        self.r_b = r_b

        # total capital (supply)
        self.K = sum(self.capital)
        # total labor (supply)
        self.P = P
        # Production
        self.Y = 0.
        # wage
        self.w = 0.
        # capital rent
        self.r = 0.

        if self.e_trajectory_output:
            self.init_e_trajectory()

        self.s_trajectory = pd.DataFrame(columns=range(self.n))

    def run(self, t_max=200.):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        Return
        ------
        exit_status : int
            if exit_status == 1: consensus/convergence reached
            if exit_status == 0: no consensus/convergence reached at t=t_max
            if exit_status ==-1: no consensus, no update candidates found (BAD)
            if exit_status ==-2: economic model broken (BAD)
        """
        candidate = 0
        while self.t < t_max:
            # 1 find update candidate and respective update time
            (candidate, neighbor,
             neighbors, update_time) = self.find_update_candidates()

            # 2 integrate economic model until t=update_time:
            self.update_economy(update_time)

            # 3 update opinion formation in case,
            # update candidate was found:
            if candidate >= 0:
                self.update_opinion_formation(candidate,
                                              neighbor, neighbors)

            if not self.run_full_time and self.converged:
                break

        # save final state to dictionary
        self.final_state = {
                'adjacency': self.neighbors,
                'savings_rate': self.savings_rate,
                'capital': self.capital,
                'tau': self.tau, 'phi': self.phi, 'eps': self.eps,
                'P': self.P, 'b': self.b, 'd': self.d,
                'test': self.debug, 'R_depletion': False}

        if self.converged:
            return 1        # good - consensus reached
        elif not self.converged:
            self.convergence_state = float('nan')
            self.convergence_time = self.t
            return 0        # no consensus found during run time
        elif candidate == -2:
            return -1       # bad run - opinion formation broken
        elif np.isnan(self.G):
            return -2       # bad run - economy broken
        else:
            return -3       # very bad run. Investigations needed

    def economy_dot(self, x0, t):
        """
        economic model assuming Cobb-Douglas production:

            Y = b P^pi K^kappa

        and no profits:

            Y - w P - r K = 0,

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            N + 1. First N entries are
            household capital [0:n],
            the last entry is total population.
        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            N + 1. First N entries are changes
            household capital [n:2N],
            the last entry is the change in total population
        """

        capital = np.where(x0[0:self.n] > 0,
                           x0[0:self.n],
                           np.full(self.n, self.epsilon.eps))

        P = x0[-1]
        K = sum(capital)

        assert K >= 0, 'negative capital'

        self.w = self.b * self.pi * P ** (self.pi - 1) * K ** self.kappa
        self.r = self.b * self.kappa * P ** self.pi * K ** (self.kappa - 1)

        self.K = K
        self.P = P

        self.income = (self.r * self.capital + self.w * P / self.n)

        assert all(self.income > 0), \
            'income is negative, K: {} \n income: \n {}'.format(K, self.income)

        P_dot = self.r_b * P

        capital_dot = \
            self.savings_rate * self.income - self.capital * self.d

        return list(capital_dot) + [P_dot]

    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = list(self.capital) + [self.P]

        # integrate the system
        x1 = odeint(self.economy_dot, x0, dt, mxhnil=1)[1]

        self.capital = np.where(x1[0:self.n] > 0,
                                x1[0:self.n], np.zeros(self.n))
        self.P = x1[-1]

        self.t = update_time
        self.steps += 1

        # calculate economic output:
        self.Y = self.b * self.K ** self.kappa * self.P ** self.pi
        self.consumption = self.income * (1 - self.savings_rate)

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()

    def find_update_candidates(self):

        j = 0
        i_max = 1000 * self.n
        candidate = 0
        neighbor = self.n
        neighbors = []
        update_time = self.t

        while j < i_max:

            # find household with min waiting time
            candidate = self.waiting_times.argmin()

            # remember update_time and increase waiting time of household
            update_time = self.waiting_times[candidate]
            self.waiting_times[candidate] += \
                np.random.exponential(scale=self.tau)

            # load neighborhood of household i
            neighbors = self.neighbors[:, candidate].nonzero()[0]


            # choose random neighbor of candidate
            if len(neighbors) > 0:
                func_vals = (1. - self.savings_rate[neighbors]) *\
                                       self.income[neighbors]
                neighbor = neighbors[np.argmax(func_vals)]
            if neighbor < self.n:
                # update candidate found (GOOD)
                break
            else:
                j += 1
                if j % self.n == 0:
                    if self.detect_consensus_state(self.savings_rate):
                        # no update candidate found because of
                        # consensus state (GOD)
                        candidate = -1
                        break
            if j >= i_max:
                # no update candidate and no consensus found (BAD)
                candidate = -2

        return candidate, neighbor, neighbors, update_time

    def update_opinion_formation(self, candidate, neighbor,
                                 neighbors):
        if self.fitness(neighbor)>self.fitness(candidate):
            self.savings_rate[candidate] = self.savings_rate[neighbor].copy()
        return 0

    def detect_consensus_state(self, d_opinions):
        # check if network is split in components with
        # same investment_decisions/preferences
        # returns 1 if consensus state is detected,
        # returns 0 if NO consensus state is detected.

        cc = connected_components(self.neighbors, directed=False)[1]
        self.consensus = all(len(np.unique(d_opinions[c])) == 1
                             for c in ((cc == u).nonzero()[0]
                             for u in np.unique(cc)))
        if self.eps == 0:
            if self.consensus and self.convergence_state == -1:
                self.convergence_state = np.mean(d_opinions)
                self.convergence_time = self.t
                self.converged = True

        return self.converged

    def fitness(self, agent):
        return self.income[agent] * (1 - self.savings_rate[agent])

    def init_e_trajectory(self):
        element = ['time',
                   'wage',
                   'r',
                   's',
                   'capital',
                   'labor share of gdp',
                   'C',
                   'P',
                   'Y',
                   'consensus'
                   ]
        self.e_trajectory.append(element)

        self.w = self.b * self.pi * self.P ** (self.pi - 1) * self.K ** self.kappa
        self.r = self.b * self.kappa * self.P ** self.pi * self.K ** (self.kappa - 1)

        self.income = (self.r * self.capital + self.w * self.P / self.n)

        self.update_e_trajectory()

    def update_e_trajectory(self):
        element = [self.t,
                   self.w,
                   self.r,
                   self.savings_rate.copy(),
                   self.capital.copy(),
                   self.P * self.w,
                   self.income.copy() * (1 - self.savings_rate.copy()),
                   self.P,
                   self.Y,
                   self.converged,
                   ]
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory.pop(0)
        trj = pd.DataFrame(self.e_trajectory, columns=columns)
        trj = trj.set_index('time')

        return trj


if __name__ == '__main__':
    """
    Perform test run and plot some output to check
    functionality
    """
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import cPickle as cp
    output_location = 'test_output/'\
        + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") + '_output'

    n = 100
    p = 10. / float(n)

    savings_rates = [uniform(0, 1) for i in range(n)]

    # network:

    while True:
        #net = nx.erdos_renyi_graph(n, p)
        #net = nx.barabasi_albert_graph(n,3)
        net = nx.complete_graph(n)
        if len(list(net)) > 1:
            break

    adjacency_matrix = nx.adj_matrix(net).toarray()

    capital = np.ones(n)

    input_parameters = {'tau': 500, 'phi': 0.01, 'eps': 0.01, 'b': 1., 'P': 1.,
                        'd': 0.50, 'pi': 0.5}
    init_conditions = (adjacency_matrix, savings_rates, capital)

    model = SavingsCore_thebest_ext(*init_conditions,
                                **input_parameters)

    # Turn off economic trajectory
    model.e_trajectory_output = True

    # Turn on debugging
    model.debug = True

    # Run Model
    tmax = 200000.
    model.run(t_max=tmax)
    trajectory = model.get_e_trajectory()
    with open('trajectory_Ldist_tau500_phi01_eps01_d50', 'wb') as dumpfile:
        cp.dump(trajectory, dumpfile)


    g = nx.from_numpy_matrix(model.neighbors)

    df = pd.DataFrame(
        data=np.array([list(model.savings_rate), list(model.income), list(model.consumption)]).transpose(),
        columns=['s', 'i', 'c'])

    colors = [c for c in 'gk']



    #########################
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(221)
    trajectory['r'].plot(ax=ax1, style=colors)
    ax1.plot(range(int(tmax)),
             np.repeat(0.05, (int(tmax))))
    ax1.set_title('last r= %s, last w= %s' % (trajectory['r'].values[-1], trajectory['wage'].values[-1]))

    ax2 = fig.add_subplot(223)
    ax2.hist(model.savings_rate, bins=30)

    ax3 = fig.add_subplot(224)
    ax3.scatter(model.savings_rate, model.consumption, marker='x')
    ax3.set_xlabel('savings rate')
    ax3.set_ylabel('consumption')

    ax4 = fig.add_subplot(222)
    sc = ax4.scatter(model.savings_rate, model.capital, cmap=plt.get_cmap('jet'), c=model.consumption)
    ax4.set_xlabel('savings rate')
    ax4.set_title('max s = %s' % np.max(model.savings_rate))
    # ax4.plot(np.arange(0,np.max(model.savings_rate),0.01), trajectory['wage'].values[-1] / ((0.4/np.arange(0,np.max(model.savings_rate),0.01)) - trajectory['r'].values[-1]))

    ax4.set_ylabel('capital')
    plt.colorbar(sc, ax=ax4)

    fig.tight_layout()
    plt.savefig('best_Ldist_tau500_fully_d05.png')


    # Initialize Model
    if False:
        model = SavingsCore_thebest(*init_conditions,
                            **input_parameters)

        # Turn off economic trajectory
        model.e_trajectory_output = True

        # Turn on debugging
        model.debug = True

        # Run Model
        tmax = 200000.
        model.run(t_max=tmax)

        g = nx.from_numpy_matrix(model.neighbors)

        df = pd.DataFrame(data=np.array([list(model.savings_rate), list(model.income), list(model.consumption)]).transpose(),
                          columns=['s', 'i', 'c'])


        colors = [c for c in 'gk']

        trajectory = model.get_e_trajectory()

        fig = plt.figure(figsize=(15,10))
        ax1 = fig.add_subplot(221)
        trajectory['r'].plot(ax=ax1, style=colors)
        ax1.plot(range(int(tmax)),
                 np.repeat(0.40, (int(tmax))))
        ax1.set_title('last r= %s, last w= %s'%(trajectory['r'].values[-1], trajectory['wage'].values[-1]))

        ax2 = fig.add_subplot(223)
        ax2.hist(model.savings_rate, bins=30)

        ax3 = fig.add_subplot(224)
        ax3.scatter(model.savings_rate, model.consumption, marker='x')
        ax3.set_xlabel('savings rate')
        ax3.set_ylabel('consumption')

        ax4 = fig.add_subplot(222)
        sc = ax4.scatter(model.savings_rate, model.capital, cmap=plt.get_cmap('jet'), c=model.consumption)
        ax4.set_xlabel('savings rate')
        ax4.set_title('max s = %s'%np.max(model.savings_rate))
        #ax4.plot(np.arange(0,np.max(model.savings_rate),0.01), trajectory['wage'].values[-1] / ((0.4/np.arange(0,np.max(model.savings_rate),0.01)) - trajectory['r'].values[-1]))

        ax4.set_ylabel('capital')
        plt.colorbar(sc, ax=ax4)


        fig.tight_layout()
        plt.savefig('example_trajectory_best_nonoise_tau500_fully_d40_K100.png')

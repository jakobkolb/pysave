"""Test the micro model."""

import datetime
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform

import networkx as nx
import numpy as np

from pysave.model.model import SavingsCore as sc

output_location = \
    'test_output/' \
    + datetime.datetime.now().strftime("%d_%m_%H-%M-%Ss") \
    + '_output'

n = 100
p = 10. / float(n)

savings_rates = [uniform(0, 1) for i in range(n)]

# network:

while True:
    net = nx.erdos_renyi_graph(n, p)
    if len(list(net)) > 1:
        break

adjacency_matrix = nx.adj_matrix(net).toarray()

capital = np.ones(n)

input_parameters = {'tau': 50, 'phi': 0., 'eps': 0.0, 'b': 1., 'P': 1.}
init_conditions = (adjacency_matrix, savings_rates, capital)

# Initialize Model

model = sc(*init_conditions,
           **input_parameters)

# Turn off economic trajectory
model.e_trajectory_output = True

# Turn on debugging
model.debug = True

# Run Model
model.run(t_max=10000.)

g = nx.from_numpy_matrix(model.neighbors)

df = pd.DataFrame(data=np.array([list(model.savings_rate), list(model.income), list(model.consumption)]).transpose(),
                  columns=['s', 'i', 'c'])


colors = [c for c in 'gk']

trajectory = model.get_e_trajectory()
print(trajectory.columns)


def close_event():
    plt.close()

fig = plt.figure()
timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
timer.add_callback(close_event)

ax1 = fig.add_subplot(221)
trajectory[['s', 'r']].plot(ax=ax1, style=colors)

ax2 = fig.add_subplot(223)
nx.draw(g, ax=ax2, cmap=plt.get_cmap('jet'), node_color=model.consumption, node_size=40)

ax3 = fig.add_subplot(224)
trajectory[['c']].plot(ax=ax3, style=colors)

ax4 = fig.add_subplot(222)
sc = ax4.scatter(model.savings_rate, model.capital, cmap=plt.get_cmap('jet'), c=model.consumption)
plt.colorbar(sc, ax=ax4)

fig.tight_layout()
timer.start()
plt.show()

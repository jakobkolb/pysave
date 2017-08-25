# -*- coding: utf-8 -*-
"""

"""
import getpass
import os
import pickle as cp
import sys
import time
import itertools as it

import networkx as nx
import numpy as np
import pandas as pd
from random import uniform

loc = '/home/yuki/Dropbox/Unizeugs/Fernuni/BA/code/pysave/experiments/output_data/X2/'
name= 'parameter_scan_trajectory'
x = np.load(loc+name)
x = x.replace([np.inf, -np.inf], np.nan)
dataframe = x.where(x < 10**300, np.nan).xs(level=['eps'], key=[0.0])
print dataframe.head()

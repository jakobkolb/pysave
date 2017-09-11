from os import listdir
from os.path import isfile, join
import numpy as np
import sys
onlyfiles = [f for f in listdir('/p/tmp/asano/Savings_Experiments/X8/') if isfile(join('/p/tmp/asano/Savings_Experiments/X8/', f))]

corrs = np.zeros(shape=(200,50))

for f in sorted(onlyfiles):
    f2 = join( '/p/tmp/asano/Savings_Experiments/X8/', f)
    try:
        traj = np.load(f2)["macro"]
        print 'ok', f
    except:
        print 'failed', f


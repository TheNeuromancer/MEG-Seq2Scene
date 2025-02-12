import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg') # no output to screen.
import mne
import numpy as np
# from ipdb import set_trace
import argparse
import pickle
import time
import importlib
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

# local imports
from utils.decod import *

start_time = time.time()

print(f'Total elasped time since the script began: {(time.time()-start_time)/60:.2f}min')
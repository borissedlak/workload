import os

import pandas as pd
import pgmpy.base.DAG
from matplotlib import pyplot as plt
from pgmpy.estimators import K2Score, HillClimbSearch

from FGCS import util_fgcs

ROOT = os.path.dirname(__file__)
fig, ax = plt.subplots()

data_laptop = pd.read_csv(ROOT + '/Laptop_l2.csv')
data_orin = pd.read_csv(ROOT + '/Orin.csv')

data_combined = pd.concat([data_orin, data_laptop])
data_combined['factor'] = data_combined['pv'] * data_combined['ra']
data_combined['bitrate'] = data_combined['pixel'] * data_combined['fps']
del data_combined['timestamp']

scoring_method = K2Score(data=data_combined)  # BDeuScore | AICScore
estimator = HillClimbSearch(data=data_combined)

dag: pgmpy.base.DAG = estimator.estimate(
    scoring_method=scoring_method, max_indegree=4, epsilon=30,
)

regular = '#a1b2ff'  # blue
special = '#c46262'  # red

util_fgcs.export_BN_to_graph(dag, try_visualization=True, save=True, name="raw_model", color_map=regular)

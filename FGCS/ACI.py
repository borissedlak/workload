import os

import numpy as np
import pandas as pd
import pgmpy
import pgmpy.base.DAG
import scipy.stats as stats
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader
from scipy.stats import norm

from FGCS import util_fgcs
from FGCS.util_fgcs import print_execution_time

ROOT = os.path.dirname(__file__)
s_obsolete = pd.DataFrame()
m_obsolete = XMLBIFReader("model.xml").get_model()

initial_batch_size = 12
initial_next_batch = 30

# Alternatively create a pointer for each batch, this would be smarter...
splits = None  # samples.groupby('batch_size')
records = {}  #


class ACI:
    def __init__(self, trace=False):
        self.model = None
        self.distance = 0
        self.current_batch = pd.DataFrame()
        self.entire_training_data = pd.DataFrame()
        self.past_training_data = pd.DataFrame()
        self.latest_structure = None

        self.pv_matrix = np.full((6, 5), -1)  # low distance (object tracking?) & high transformed rate (privacy preservation)
        self.ra_matrix = np.full((6, 5), -1)  # slo violation rate? --> in_time, network, energy_cons
        self.ig_matrix = np.full((6, 5), -1)  # surprise rate?

    def iterate(self, c_pixel, c_fps):
        self.load_last_batch()
        self.past_training_data = self.entire_training_data.copy()

        if len(self.past_training_data) == 0:
            self.bnl(self.current_batch)
            self.entire_training_data = self.current_batch.copy()
            return c_pixel, c_fps

        self.entire_training_data = pd.concat([self.entire_training_data, self.current_batch], ignore_index=True)

        prediction = None
        actual = self.SLOs_fulfilled(self.current_batch)

        new_config = c_pixel, c_fps
        self.retrain_parameter()

        return new_config

    @print_execution_time
    def retrain_parameter(self):
        # duplicate_rows = self.entire_training_data[self.entire_training_data.duplicated()]
        # print(duplicate_rows)
        self.model = BayesianNetwork(ebunch=self.model)
        self.model.fit(self.entire_training_data)  # , n_prev_samples=len(self.past_training_data))
        # TODO: export model.xml

    def initialize(self):
        self.bnl(self.entire_training_data)

    @print_execution_time
    def bnl(self, samples):

        scoring_method = K2Score(data=samples)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=samples)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=4, epsilon=1,
        )

        util_fgcs.print_BN(dag, vis_ls=["circo"], save=True, name="raw_model")

        self.latest_structure = dag.copy()
        self.model = BayesianNetwork(ebunch=dag)
        self.model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    def load_last_batch(self):
        samples = pd.read_csv('../data/Performance.csv')
        samples = util_fgcs.prepare_samples(samples)
        self.current_batch = samples

    def SLOs_fulfilled(self, batch: pd.DataFrame):
        # TODO: Create more sophisticated SLOs
        ratio_in_time = batch[batch["in_time"]].size / batch.size
        print(ratio_in_time)
        if ratio_in_time > 0.8:
            return True
        else:
            return False

    def get_surprise(historical_data, batch):
        mean = np.mean(historical_data)
        std_dev = np.std(historical_data)
        pdf_values = stats.norm.pdf(batch, loc=mean, scale=std_dev)
        nll_values = -np.log(np.maximum(pdf_values, 1e-10))
        return np.sum(nll_values)

    # plot_histogram_with_normal_distribution(records[30]['part_delay'])

    # sr_per_batch_size = {}
    # ig_per_batch_size = {}
    # surprise_history_per_batch = {i: [] for i in range(12, 31)}
    #
    # model.fit(data=entire_training_data)

    # var_el = VariableElimination(model)
    # print(var_el.query(variables=["distance"]))

    def plot_boxplot(column_data):
        plt.boxplot(column_data)
        plt.show()

    def plot_histogram_with_normal_distribution(column_data):
        plt.hist(column_data, bins=20, density=True, alpha=0.6, color='b', label='Histogram')

        # Fit a normal distribution to the data
        mu, std = norm.fit(column_data)

        # Create a range of values for the x-axis
        xmin, xmax = plt.xlim()
        x = np.linspace(0, xmax, 100)

        # Calculate the probability density function (PDF) for the normal distribution
        p = norm.pdf(x, mu, std)

        # Plot the normal distribution curve
        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

        # Add labels and a legend
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram and Normal Distribution')
        plt.legend()

        # Show the plot
        plt.show()

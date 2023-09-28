import os

import numpy as np
import pandas as pd
import pgmpy
import pgmpy.base.DAG
import scipy.stats as stats
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader
from scipy.interpolate import griddata, interp2d
from scipy.stats import norm

from FGCS import util_fgcs
from FGCS.util_fgcs import print_execution_time

ROOT = os.path.dirname(__file__)
s_obsolete = pd.DataFrame()
m_obsolete = XMLBIFReader("model.xml").get_model()


class ACI:
    pixel_list = [120, 180, 240, 300, 360, 420]
    fps_list = [14, 18, 22, 26, 30]

    def __init__(self, trace=False):
        self.model = None
        self.distance = 0
        self.current_batch = pd.DataFrame()
        self.entire_training_data = pd.DataFrame()
        self.past_training_data = pd.DataFrame()
        self.latest_structure = None
        self.surprise_history = []

        self.pv_matrix = np.full((6, 5), -1.0)  # low distance & high transformed rate (privacy preservation)
        self.ra_matrix = np.full((6, 5), -1.0)  # slo violation rate? --> in_time, network, energy_cons
        self.ig_matrix = np.full((6, 5), 0)  # surprise rate?

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

        # TODO: Not knowning all parameters should be a factor for high surprise!
        if util_fgcs.verify_all_slo_parameters_known(self.model, self.current_batch):
            s = util_fgcs.get_surprise_for_data(self.model, self.current_batch)
            self.surprise_history.append(s)
            if s > (2 * np.mean(self.surprise_history[-10:])):
                self.retrain_parameter()
        else:
            # self.ig_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)]
            # self.ig_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)] += 0.1
            self.retrain_parameter()

        self.calculate_factors(self.model)
        new_config = self.get_best_configuration()

        return new_config

    def get_best_configuration(self):
        pv_interpolated = util_fgcs.interpolate_values(self.pv_matrix)
        ra_interpolated = util_fgcs.interpolate_values(self.ra_matrix)
        # ig_interpolated = util_fgcs.interpolate_values(self.ig_matrix)

        max_sum = -float('inf')
        best_index = 0, 0
        for i in range(len(ACI.pixel_list)):
            for j in range(len(ACI.fps_list)):
                element_sum = pv_interpolated[i, j] + ra_interpolated[i, j]
                if element_sum > max_sum:
                    max_sum = element_sum
                    best_index = i, j

        p, f = best_index
        return ACI.pixel_list[p], ACI.fps_list[f]

    def calculate_factors(self, model):
        bitrate_list = model.__getattribute__("states")["bitrate"]
        var_el = VariableElimination(model)

        # TODO: Probably too slow
        bitrate_group = self.entire_training_data.groupby('bitrate')

        for br in bitrate_list:
            fps = bitrate_group.get_group(br)['fps'].min()
            pixel = bitrate_group.get_group(br)['pixel'].min()

            # distance = var_el.query(variables=[distance_slo], evidence={'bitrate': br, 'config': mode}).values[1]
            time = util_fgcs.get_true(var_el.query(variables=["in_time"], evidence={'bitrate': br}))
            success = util_fgcs.get_true(var_el.query(variables=["success"], evidence={'bitrate': br}))

            self.pv_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = success
            self.ra_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = time

            print(f"{int(br/fps)}p_{fps} returns {time} and {success}")

    def infer_configuration(self):#param.__getattribute__("state_names")[param.variables[0]]
        self.pv_matrix = 10

    @print_execution_time
    def retrain_parameter(self, full_retrain=False):
        # self.model = BayesianNetwork(ebunch=self.model)
        # cpds_empty = len(self.model.get_cpds()) == 0

        if full_retrain:
            self.model.fit(self.entire_training_data)
        else:
            try:
                self.model.fit_update(self.current_batch, n_prev_samples=len(self.past_training_data))
                # util_fgcs.print_in_red("No error!")
            except ValueError as ve:
                # print(f"Caught a ValueError: {ve}")
                self.retrain_parameter(full_retrain=True)
        # TODO: export model.xml

    # TODO: Retrain the structure if it does not match the data anymore
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

    # def init_matrices(self):
    #     matrix = {}
    #
    #     row_keys = ACI.pixel_list
    #     column_keys = ACI.fps_list
    #
    #     # Initialize the nested dictionary
    #     for row_key in row_keys:
    #         matrix[row_key] = {}
    #         for col_key in column_keys:
    #             matrix[row_key][col_key] = -1  # You can set initial values here if needed

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

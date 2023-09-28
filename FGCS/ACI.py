import os

import numpy as np
import pandas as pd
import pgmpy
import pgmpy.base.DAG
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, AICScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter
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
        self.foster_bn_retrain = 0.5
        self.c_distance_bar = 50

        self.pv_matrix = np.full((6, 5), -1.0)  # low distance & high transformed rate (privacy preservation)
        self.ra_matrix = np.full((6, 5), -1.0)  # slo violation rate? --> in_time, network, energy_cons
        self.ig_matrix = np.full((6, 5), -1.0)  # surprise rate?
        self.ig_matrix[0][0], self.ig_matrix[5][4], self.ig_matrix[5][0], self.ig_matrix[0][4], \
            self.ig_matrix[3][2] = 0.0, 0.05, 0.05, 0.05, 0.05
        self.ig_matrix = util_fgcs.interpolate_values(self.ig_matrix)
        self.ig_matrix[0][0], self.ig_matrix[5][4], self.ig_matrix[5][0], self.ig_matrix[0][4], \
            self.ig_matrix[3][2] = 0.0, 0.3, 0.3, 0.3, 0.3

    def iterate(self, c_pixel, c_fps, c_stream_count):
        self.load_last_batch()
        self.past_training_data = self.entire_training_data.copy()

        if len(self.past_training_data) == 0:
            self.bnl(self.current_batch)
            self.entire_training_data = self.current_batch.copy()
            return c_pixel, c_fps

        self.entire_training_data = pd.concat([self.entire_training_data, self.current_batch], ignore_index=True)

        # prediction = None
        # actual = self.SLOs_fulfilled(self.current_batch)

        # TODO: Not knowing all parameters should be a factor for high surprise!
        if util_fgcs.verify_all_slo_parameters_known(self.model, self.current_batch):
            s = util_fgcs.get_surprise_for_data(self.model, self.current_batch)
            self.surprise_history.append(s)

            mean_surprise_last_10_values = np.median(self.surprise_history[-10:])
            if s > ((1.8 - self.foster_bn_retrain) * mean_surprise_last_10_values):
                if self.foster_bn_retrain == 0.5:
                    self.foster_bn_retrain = 0.2
                elif self.foster_bn_retrain == 0.2:
                    self.foster_bn_retrain = 0.0
                # So far it retrains the structure if the values are very surprising, not sure if that 100% fine
                self.initialize_bn()
            elif s > (1 * mean_surprise_last_10_values):
                self.retrain_parameter()

        else:
            self.retrain_parameter(full_retrain=True)

        self.calculate_factors(self.model, c_pixel, c_fps, c_stream_count)
        new_config = self.get_best_configuration(c_stream_count)

        return new_config

    def get_best_configuration(self, c_stream_count):
        # TODO: The interpolation does not work towards the borders of the matrix
        pv_interpolated = util_fgcs.interpolate_values(self.pv_matrix)
        ra_interpolated = util_fgcs.interpolate_values(self.ra_matrix)
        ig_interpolated = util_fgcs.interpolate_values(self.ig_matrix)

        max_sum = -float('inf')
        best_index = 0, 0
        for i in range(len(ACI.pixel_list)):
            for j in range(len(ACI.fps_list)):
                element_sum = (pv_interpolated[i, j] + ra_interpolated[i, j] + ig_interpolated[i, j]
                               - ((i+j) * (c_stream_count - 1)))
                if element_sum > max_sum:
                    max_sum = element_sum
                    best_index = i, j

        p, f = best_index
        return ACI.pixel_list[p], ACI.fps_list[f]

    # @print_execution_time # takes around 10-15ms
    def calculate_factors(self, model, c_pixel, c_fps, c_stream_count):

        # ig_current_config = self.ig_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)]
        # if ig_current_config == -1 or ig_current_config == 0.3:
        #     self.ig_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)] = 0.1
        # else:
        self.ig_matrix[ACI.pixel_list.index(c_pixel)][ACI.fps_list.index(c_fps)] = 0.0

        bitrate_list = model.__getattribute__("states")["bitrate"]
        inference = VariableElimination(
            util_fgcs.get_mbs_as_bn(model, ["success", "in_time", "bitrate"]))
        bitrate_group = self.entire_training_data.groupby('bitrate')  # TODO: Probably too slow

        for br in bitrate_list:
            fps = bitrate_group.get_group(br)['fps'].min()
            pixel = bitrate_group.get_group(br)['pixel'].min()

            # distance = var_el.query(variables=[distance_slo], evidence={'bitrate': br, 'config': mode}).values[1]
            # evidence_variables = model.get_markov_blanket("in_time")# In theory, this comes from the MB
            evidence = {'bitrate': br}  # 'pixel': pixel, 'fps': fps,# TODO: Must do regression from stream --> others
            time = util_fgcs.get_true(inference.query(variables=["in_time"], evidence=evidence))
            success = util_fgcs.get_true(inference.query(variables=["success"], evidence=evidence))

            # if np.isnan(time) or np.isnan(success):

            self.pv_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = success
            self.ra_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = time

            # print(f"{int(br / fps)}p_{fps} returns {time} and {success}")

    def infer_configuration(self):
        self.pv_matrix = 10

    # @print_execution_time
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

    # TODO: Retrain the structure if it does not match the data anymore
    def initialize_bn(self):
        self.bnl(self.entire_training_data)

    def export_model(self):
        XMLBIFWriter(self.model).write_xmlbif('model.xml')
        print("Model exported as 'model.xml'")

    @print_execution_time
    def bnl(self, samples):

        scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=samples)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=4, epsilon=1,
        )

        util_fgcs.export_BN_to_graph(dag, vis_ls=["circo"], save=True, name="raw_model")

        self.latest_structure = dag.copy()
        self.model = BayesianNetwork(ebunch=dag)
        self.model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    def load_last_batch(self):
        samples = pd.read_csv('../data/Performance.csv')
        samples = util_fgcs.prepare_samples(samples, self.c_distance_bar)
        self.current_batch = samples

    def SLOs_fulfilled(self, batch: pd.DataFrame):
        # TODO: Create more sophisticated SLOs
        ratio_in_time = batch[batch["in_time"]].size / batch.size
        if ratio_in_time > 0.8:
            return True
        else:
            return False

    # def get_surprise(historical_data, batch):
    #     mean = np.mean(historical_data)
    #     std_dev = np.std(historical_data)
    #     pdf_values = stats.norm.pdf(batch, loc=mean, scale=std_dev)
    #     nll_values = -np.log(np.maximum(pdf_values, 1e-10))
    #     return np.sum(nll_values)

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

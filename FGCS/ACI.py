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
from sklearn.linear_model import LinearRegression

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

        self.valid_stream_values_time = []
        self.valid_stream_values_success = []
        self.stream_regression_model_time = LinearRegression()
        self.stream_regression_model_success = LinearRegression()

        self.pv_matrix = np.full((6, 5), -1.0)  # low distance & high transformed rate (privacy preservation)
        self.ra_matrix = np.full((6, 5), -1.0)  # slo violation rate? --> in_time, network, energy_cons
        self.ig_matrix = np.full((6, 5), -1.0)  # surprise rate?
        self.ig_matrix[0][0], self.ig_matrix[5][4], self.ig_matrix[5][0], self.ig_matrix[0][4], \
            self.ig_matrix[3][2] = 0.0, 0.05, 0.05, 0.05, 0.05
        self.ig_matrix = util_fgcs.interpolate_values(self.ig_matrix)
        self.ig_matrix[0][0], self.ig_matrix[5][4], self.ig_matrix[5][0], self.ig_matrix[0][4], \
            self.ig_matrix[3][2] = 0.0, 0.3, 0.3, 0.3, 0.3

    def iterate(self, c_stream_count):
        self.load_last_batch()
        self.past_training_data = self.entire_training_data.copy()
        c_pixel = self.current_batch.iloc[0]['pixel']
        c_fps = self.current_batch.iloc[0]['fps']

        if len(self.past_training_data) == 0:
            self.bnl(self.current_batch)
            self.entire_training_data = self.current_batch.copy()
            return c_pixel, c_fps

        self.entire_training_data = pd.concat([self.entire_training_data, self.current_batch], ignore_index=True)

        # prediction = None
        # actual = self.SLOs_fulfilled(self.current_batch)

        if util_fgcs.verify_all_slo_parameters_known(self.model, self.current_batch):
            s = util_fgcs.get_surprise_for_data(self.model, self.current_batch)
            self.surprise_history.append(s)

            mean_surprise_last_10_values = np.median(self.surprise_history[-10:])
            if s > ((2 - self.foster_bn_retrain) * mean_surprise_last_10_values):
                if self.foster_bn_retrain == 0.5:
                    self.foster_bn_retrain = 0.2
                elif self.foster_bn_retrain == 0.2:
                    self.foster_bn_retrain = 0.0
                # So far it retrains the structure if the values are very surprising, not sure if that 100% fine
                self.initialize_bn()
            elif s > (1 * mean_surprise_last_10_values):
                self.retrain_parameter()

        else:
            self.retrain_parameter()

        self.calculate_factors(self.model, c_pixel, c_fps, c_stream_count)
        new_config = self.get_best_configuration(c_stream_count)

        return new_config

    def get_best_configuration(self, c_stream_count):
        pv_interpolated = util_fgcs.interpolate_values(self.pv_matrix)
        ra_interpolated = util_fgcs.interpolate_values(self.ra_matrix)
        ig_interpolated = util_fgcs.interpolate_values(self.ig_matrix)

        max_sum = -float('inf')
        best_index = 0, 0
        for i in range(len(ACI.pixel_list)):
            for j in range(len(ACI.fps_list)):
                element_sum = (pv_interpolated[i, j] + ra_interpolated[i, j] + ig_interpolated[i, j])
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
            util_fgcs.get_mbs_as_bn(model, ["success", "in_time", "bitrate", "stream_count"]))
        bitrate_group = self.entire_training_data.groupby('bitrate')  # TODO: Probably too slow

        # Ensure that the current one is processed first to train the regression
        current_bitrate = c_pixel * c_fps
        bitrate_list.remove(current_bitrate)
        bitrate_list.insert(0, current_bitrate)

        unknown_combinations = []

        for br in bitrate_list:
            fps = bitrate_group.get_group(br)['fps'].min()
            pixel = bitrate_group.get_group(br)['pixel'].min()

            # distance = var_el.query(variables=[distance_slo], evidence={'bitrate': br, 'config': mode}).values[1]
            # evidence_variables = model.get_markov_blanket("in_time")# In theory, this comes from the MB

            # if stream_count_known:
            try:
                evidence = {'bitrate': br}
                time = util_fgcs.get_true(inference.query(variables=["in_time"], evidence=evidence))
                success = util_fgcs.get_true(inference.query(variables=["success"], evidence=evidence))

                self.valid_stream_values_time.append((pixel, fps, c_stream_count, time))
                self.valid_stream_values_success.append((pixel, fps, c_stream_count, success))
                self.pv_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = success
                self.ra_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = time
            except Exception as ex:
                unknown_combinations.append((fps, pixel, c_stream_count))
                print(ex)

        if len(unknown_combinations) > 0:

            input_data = np.array([(x1, x2, x3) for x1, x2, x3, y in self.valid_stream_values_time])
            target_data = np.array([y for x1, x2, x3, y in self.valid_stream_values_time])
            self.stream_regression_model_time.fit(input_data, target_data)

            input_data = np.array([(x1, x2, x3) for x1, x2, x3, y in self.valid_stream_values_success])
            target_data = np.array([y for x1, x2, x3, y in self.valid_stream_values_success])
            self.stream_regression_model_success.fit(input_data, target_data)

            for uc in unknown_combinations:
                print(uc)

            # print(f"{int(br / fps)}p_{fps} returns {time} and {success}")

    def infer_configuration(self):
        self.pv_matrix = 10

    @print_execution_time
    def retrain_parameter(self, full_retrain=False):
        if full_retrain:
            self.model.fit(self.entire_training_data)
        else:
            try:
                self.model.fit_update(self.current_batch, n_prev_samples=len(self.past_training_data))
                # util_fgcs.print_in_red("No error!")
            except ValueError as ve:
                # print(f"Caught a ValueError: {ve}")
                self.retrain_parameter(full_retrain=True)

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
        samples = pd.read_csv('../data/Last_Batch.csv')
        samples = util_fgcs.prepare_samples(samples, self.c_distance_bar)
        self.current_batch = samples

    def SLOs_fulfilled(self, batch: pd.DataFrame):
        # TODO: Create more sophisticated SLOs
        ratio_in_time = batch[batch["in_time"]].size / batch.size
        if ratio_in_time > 0.8:
            return True
        else:
            return False

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

import itertools
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import pgmpy
import pgmpy.base.DAG
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, AICScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import util_fgcs
from util_fgcs import print_execution_time

ROOT = os.path.dirname(__file__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ACI:
    pixel_list = [120, 180, 240, 300, 360, 420]
    fps_list = [14, 18, 22, 26, 30]
    bitrate_dict = {}

    # TODO: 30 * 180 and 300 * 18 are identical!!
    for pair in itertools.product(pixel_list, fps_list):
        bitrate = pair[0] * pair[1]
        bitrate_dict.update({bitrate: [pair[0], pair[1]]})

    def __init__(self, load_model=None, distance_slo=40, network_slo=99999):
        self.c_distance_bar = distance_slo
        self.c_network_bar = network_slo
        if load_model:
            print("Loading pretained model")
            self.model = XMLBIFReader(load_model).get_model()
            util_fgcs.export_BN_to_graph(self.model, vis_ls=["circo"], save=True, name="raw_model")
            self.foster_bn_retrain = 0.2
            self.backup_data = util_fgcs.prepare_samples(pd.read_csv("backup_entire_data.csv"),
                                                         self.c_distance_bar, self.c_network_bar)
        else:
            self.model = None
            self.foster_bn_retrain = 0.5

        self.load_model = True if self.model is not None else False
        self.distance = 0
        self.current_batch = pd.DataFrame()
        self.entire_training_data = pd.DataFrame()
        self.past_training_data = pd.DataFrame()
        # self.latest_structure = None
        self.surprise_history = []
        self.function_time = []

        self.valid_stream_values_ra = []
        self.valid_stream_values_pv = []
        self.stream_regression_model_ra = LinearRegression()
        self.stream_regression_model_pv = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=4)

        self.pv_matrix = np.full((6, 5), -1.0)  # low distance & high transformed rate (privacy preservation)
        self.ra_matrix = np.full((6, 5), -1.0)  # slo violation rate? --> in_time, network, energy_cons
        self.ig_matrix = np.full((6, 5), -1.0)  # surprise rate?
        self.ig_matrix[0][0], self.ig_matrix[5][4], self.ig_matrix[5][0], self.ig_matrix[0][4], \
            self.ig_matrix[3][2] = 0.0, 0.05, 0.05, 0.05, 0.05
        self.ig_matrix = util_fgcs.interpolate_values(self.ig_matrix)
        self.ig_matrix[0][0], self.ig_matrix[5][4], self.ig_matrix[5][0], self.ig_matrix[0][4], \
            self.ig_matrix[2][2] = 0.0, 0.3, 0.3, 0.3, 0.3

    def iterate(self, c_stream_count):
        # start_time = time.time()
        self.load_last_batch()
        self.past_training_data = self.entire_training_data.copy()
        c_pixel = int(self.current_batch.iloc[0]['pixel'])
        c_fps = int(self.current_batch.iloc[0]['fps'])

        if len(self.past_training_data) == 0 and self.model is None:
            self.bnl(self.current_batch)
            self.entire_training_data = self.current_batch.copy()
        else:
            self.entire_training_data = pd.concat([self.entire_training_data, self.current_batch], ignore_index=True)

        # prediction = None
        # actual = self.SLOs_fulfilled(self.current_batch)

        if util_fgcs.verify_all_slo_parameters_known(self.model, self.current_batch):
            s = util_fgcs.get_surprise_for_data(self.model, self.current_batch)
            self.surprise_history.append(s)

            # TODO: Both the 2 and the foster_param can be hyperparameters
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
        p, f, pv, ra = self.get_best_configuration()

        # end_time = time.time()
        # execution_time_ms = (end_time - start_time) * 1000.0
        # self.function_time.append(execution_time_ms)
        return int(p), int(f), pv, ra

    def get_best_configuration(self):
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
        return ACI.pixel_list[p], ACI.fps_list[f], pv_interpolated[p, f], ra_interpolated[p, f]

    # @print_execution_time # takes around 10-15ms
    def calculate_factors(self, model, c_pixel, c_fps, c_stream_count):

        self.ig_matrix[ACI.pixel_list.index(int(c_pixel))][ACI.fps_list.index(int(c_fps))] = 0.0

        bitrate_list = model.__getattribute__("states")["bitrate"].copy()
        inference = VariableElimination(
            util_fgcs.get_mbs_as_bn(model, ["success", "in_time", "bitrate", "stream_count", "network", "distance"]))

        # Ensure that the current one is processed first to train the regression
        current_bitrate = str(int(c_pixel) * int(c_fps))
        bitrate_list.remove(current_bitrate)
        bitrate_list.insert(0, current_bitrate)

        unknown_combinations = []

        for br in bitrate_list:
            pixel = ACI.bitrate_dict[int(br)][0]
            fps = ACI.bitrate_dict[int(br)][1]

            try:
                evidence = {'bitrate': br, 'stream_count': c_stream_count}
                ra = util_fgcs.get_true(inference.query(variables=["in_time", "network"], evidence=evidence))
                pv = util_fgcs.get_true(inference.query(variables=["success", "distance"], evidence=evidence))

                if np.isnan(ra) or np.isnan(pv):
                    raise ValueError("_nan")

                self.valid_stream_values_pv.append((pixel, fps, int(c_stream_count), pv))
                self.pv_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = pv
                self.valid_stream_values_ra.append((pixel, fps, int(c_stream_count), ra))
                self.ra_matrix[ACI.pixel_list.index(pixel)][ACI.fps_list.index(fps)] = ra
            except Exception as ex:
                unknown_combinations.append((pixel, fps, int(c_stream_count)))
                if str(ex) != "_nan":
                    print(ex)

        if len(unknown_combinations) > 0:

            input_data = np.array([(x1, x2, x3) for x1, x2, x3, y in self.valid_stream_values_ra])
            input_data = self.poly_features.fit_transform(input_data)
            target_data = np.array([y for x1, x2, x3, y in self.valid_stream_values_ra])
            self.stream_regression_model_ra.fit(input_data, target_data)

            input_data = np.array([(x1, x2, x3) for x1, x2, x3, y in self.valid_stream_values_pv])
            input_data = self.poly_features.fit_transform(input_data)
            target_data = np.array([y for x1, x2, x3, y in self.valid_stream_values_pv])
            self.stream_regression_model_pv.fit(input_data, target_data)

            # x_range = np.linspace(1, 10, 100)  # Adjust the number of points as needed
            # y_pred_full = self.stream_regression_model_time.predict(self.poly_features.fit_transform(x_range))
            # fig, ax = plt.subplots()
            # # ax.scatter(x_utilization, y_delay_per_part, label='Observations', marker='o')
            # ax.plot(x_range, y_pred_full, label='Full Data', color='green')

            for p, f, s in unknown_combinations:
                input_vector = self.poly_features.fit_transform(np.array([[p, f, s]]))
                pv_predict = util_fgcs.cap_0_1(self.stream_regression_model_pv.predict(input_vector)[0])
                self.pv_matrix[ACI.pixel_list.index(p)][ACI.fps_list.index(f)] = pv_predict
                ra_predict = util_fgcs.cap_0_1(self.stream_regression_model_ra.predict(input_vector)[0])
                self.ra_matrix[ACI.pixel_list.index(p)][ACI.fps_list.index(f)] = ra_predict

            # print(f"{int(br / fps)}p_{fps} returns {time} and {success}")

    def infer_configuration(self):
        self.pv_matrix = 10

    @print_execution_time
    def retrain_parameter(self, full_retrain=False):
        if full_retrain:
            if self.load_model:
                util_fgcs.print_in_red("Should not happen when loading model")
                self.entire_training_data = pd.concat([self.entire_training_data, self.backup_data], ignore_index=True)
                self.load_model = False
            self.model.fit(self.entire_training_data)
        else:
            try:
                # TODO: This can by a hyperparameter when transferring knowledge
                past_data_length = len(self.past_training_data)
                if hasattr(self, 'backup_data'):
                    past_data_length += len(self.backup_data)
                self.model.fit_update(self.current_batch, n_prev_samples=past_data_length)
            except ValueError as ve:
                print(f"Caught a ValueError: {ve}")
                self.retrain_parameter(full_retrain=True)

    def initialize_bn(self):
        self.bnl(self.entire_training_data)

    def export_model(self):
        # self.entire_training_data.to_csv("backup_entire_data.csv", index=False)
        shutil.copy("../data/Performance_History.csv", "backup_entire_data.csv")
        writer = XMLBIFWriter(self.model)
        writer.write_xmlbif(filename='model.xml')
        print("Model exported as 'model.xml'")

    @print_execution_time
    def bnl(self, samples):

        scoring_method = AICScore(data=samples)  # BDeuScore | AICScore
        estimator = HillClimbSearch(data=samples)

        dag: pgmpy.base.DAG = estimator.estimate(
            scoring_method=scoring_method, max_indegree=4, epsilon=1,
        )

        util_fgcs.export_BN_to_graph(dag, vis_ls=["circo"], save=True, name="raw_model")

        # self.latest_structure = dag.copy()
        self.model = BayesianNetwork(ebunch=dag)
        self.model.fit(data=samples, estimator=MaximumLikelihoodEstimator)

    def load_last_batch(self):
        samples = pd.read_csv('../data/Last_Batch.csv')
        samples = util_fgcs.prepare_samples(samples, self.c_distance_bar, self.c_network_bar)
        self.current_batch = samples

    def SLOs_fulfilled(self, batch: pd.DataFrame):
        # TODO: Create more sophisticated SLOs
        ratio_in_time = batch[batch["in_time"]].size / batch.size
        if ratio_in_time > 0.8:
            return True
        else:
            return False

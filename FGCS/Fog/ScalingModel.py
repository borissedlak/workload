import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures


class ScalingModel:
    def __init__(self):
        self.regression_models = {}
        self.poly_features = PolynomialFeatures(degree=1)
        self.device_models = ["Xavier", "Laptop", "Orin", "Nano"]
        self.physical_devices = [("Xavier", 0), ("Xavier", 1), ("Laptop", 0), ("Orin", 1), ("Nano", 0)]
        self.load_devices = None
        for s in self.device_models:
            self.regression_models[s] = self.load_device_model(s)

    def load_device_model(self, device_name):
        file_exists = os.path.isfile("data/" + device_name + ".sav")
        if file_exists:
            return joblib.load("data/" + device_name + ".sav")
        else:
            return self.train_model(device_name)

    def train_model(self, device_name):
        df = pd.read_csv("data/" + device_name + ".csv")
        df['GPU'] = df['GPU'].astype(int)
        x_train = self.poly_features.fit_transform(df[['stream', 'GPU']])
        y_train = df[['pv', 'ra']]
        model = MultiOutputRegressor(LinearRegression())
        model.fit(x_train, y_train)
        joblib.dump(model, "data/" + device_name + ".sav")
        return model

    def predict(self, device_name, streams, GPU):
        return self.regression_models[device_name].predict(
            self.poly_features.fit_transform(np.array([[streams, GPU]])))

    def shuffle_load(self, total_streams):
        i = 0
        self.load_devices = {d: 0 for d in self.physical_devices}

        while i < total_streams:
            best_delta = (None, -999, None)

            for d, gpu in self.physical_devices:
                potential_next = self.predict(d, self.load_devices[(d, gpu)] + 1, gpu)
                fact = (potential_next[0][0] * potential_next[0][1])

                if fact > best_delta[1]:
                    best_delta = (d, fact, gpu)

            self.load_devices[(best_delta[0], best_delta[2])] += 1
            i += 1

        self.print_current_assignment()

    def get_assigned_streams(self, device_name, gpu):
        return self.load_devices[(device_name, gpu)]

    def override_assignment(self, assignment):
        self.load_devices = assignment

    def print_current_assignment(self):

        for d in self.load_devices:
            g = "GPU" if d[1] == 1 else "CPU"
            print(f"{d[0]} {g} got assigned {self.load_devices[d]} streams")


# s = ScalingModel()
# s.shuffle_load(10)

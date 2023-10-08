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
        self.poly_features = PolynomialFeatures(degree=2)
        self.device_models = ["Xavier", "Laptop"]
        self.physical_devices = [("Xavier", 0), ("Xavier", 1), ("Laptop", 0)]
        for s in self.device_models:  # , "Orin", "Nano"]:
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
        load_devices = {d: 0 for d in self.physical_devices}

        while i < total_streams:
            best_delta = (None, -999, None)

            for d, gpu in self.physical_devices:
                # current = self.predict(d, load_devices[d], gpu)
                potential_next = self.predict(d, load_devices[(d, gpu)] + 1, gpu)
                fact = (potential_next[0][0] * potential_next[0][1])

                if fact > best_delta[1]:
                    best_delta = (d, fact, gpu)

            load_devices[(best_delta[0], best_delta[2])] += 1
            i += 1

        for d in load_devices:
            g = "GPU" if d[1] == 1 else "CPU"
            print(f"{d[0]} {g} got assigned {load_devices[d]} streams")

    def get_assigned_streams(self, device_name):
        return None


a = ScalingModel()
# print(a.predict("Xavier", 10, 1))
# print(a.predict("Xavier", 10, 0))

a.shuffle_load(20)

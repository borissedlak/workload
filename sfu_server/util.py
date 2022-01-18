import sys
import time

import numpy as np


class FPS_:
    def __init__(self, output_string, calculate_avg=1):
        self.output_string = output_string
        self.prev_time = 0
        self.new_time = 0

        if calculate_avg < 1:
            raise ValueError("Average must be calculated over value 1 at least")
        self.store = Cyclical_Array(calculate_avg)

    def update_and_print(self):
        self.new_time = time.time()
        dif = self.new_time - self.prev_time

        if dif != 0:
            self.store.put(1 / dif)
            self.prev_time = self.new_time
            sys.stdout.write("\r" + self.output_string + "%d" % self.store.get_average())
            sys.stdout.flush()
            # print(self.output_string + str(self.store.get_average()))


class Cyclical_Array:
    def __init__(self, size):
        self.data = np.zeros(size, dtype=object)
        self.index = 0
        self.size = size

    def put(self, item):
        self.data[self.index % self.size] = item
        self.index = self.index + 1

    def get_average(self):
        return np.mean(self.data)

import sys
import time
from datetime import datetime

import numpy as np
from aiortc import RTCStatsReport, RTCRemoteInboundRtpStreamStats


def cropFrameToBoxArea(frame, box):
    return frame[box[1]:box[3], box[0]:box[2]]


class FPS_:
    def __init__(self, output_string, calculate_avg=1, display_total=True):
        self.output_string = output_string
        self.display_total = display_total
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
            msg = "\r" + self.output_string + "%d" % self.store.get_average()
            if self.display_total:
                msg += ", last total %.3fs" % dif
            sys.stdout.write(msg)
            sys.stdout.flush()


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


def diffAsStringInMS(a: datetime, b: datetime):
    return str(int((a - b).microseconds / 1000))


def printExecutionTime(name: str, a: datetime, b: datetime):
    if a is not None and b is not None:
        print(f' {name} took {diffAsStringInMS(a, b)}ms')
        return diffAsStringInMS(a, b)
    return 0


def write_execution_times(write_store):
    for function_name in write_store.keys():

        f = open(f'../evaluation/csv_export/function_time/{function_name}.csv', 'w+')
        f.write('execution_time,timestamp\n')

        for (t, delta) in write_store[function_name]:
            f.write(f'{t},{delta}\n')

        f.close()


def getTupleFromStats(consumer_stats: RTCStatsReport):
    stat_list = list(filter(lambda x: isinstance(x, RTCRemoteInboundRtpStreamStats), list(consumer_stats.values())))
    rtt = round(stat_list[0].roundTripTime, 4)
    timestamp = stat_list[0].timestamp
    return rtt, timestamp

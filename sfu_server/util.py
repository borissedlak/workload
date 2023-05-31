import math
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

        return dif


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
    return int((a - b).microseconds / 1000)


def printExecutionTime(name: str, a: datetime, b: datetime):
    if a is not None and b is not None:
        # print(f' {name} took {diffAsStringInMS(a, b)}ms')
        return diffAsStringInMS(a, b)
    return 0


def write_execution_times(write_store, video_name, model_name):
    for function_name in write_store.keys():

        f = open(f'../evaluation/csv_export/function_time/{video_name}/{model_name}/{function_name}.csv', 'w+')
        f.write('execution_time,timestamp,cpu_utilization,memory_usage,pixel,fps,bitrate,success,within_time,distance\n')

        for (delta, ts, cpu, memory, pixel, fps, detected, distance) in write_store[function_name]:
            f.write(f'{delta},{ts},{cpu},{memory},{pixel},{fps},{pixel * fps},{detected},{delta <= (1000 / fps)},{distance}\n')

        f.close()

def get_center_from_box(box):
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y

def get_relative_distance_between_points(p1,p2,img):

    p1_x, p1_y = p1
    p2_x, p2_y = p2

    # The intention was to get the relative distance in different resolutions. The results match quite, but are a bit off
    return math.ceil(math.sqrt(((p1_x - p2_x)/img.shape[1])**2 + ((p1_y - p2_y)/img.shape[0])**2) * 1000)

def get_cpu_temperature():
    temperature = None
    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
        temp_str = f.read().strip()
        if temp_str.isdigit():
            temperature = float(temp_str) / 1000.0
    return temperature


def getTupleFromStats(consumer_stats: RTCStatsReport):
    stat_list = list(filter(lambda x: isinstance(x, RTCRemoteInboundRtpStreamStats), list(consumer_stats.values())))
    rtt = round(stat_list[0].roundTripTime, 4)
    timestamp = stat_list[0].timestamp
    return rtt, timestamp

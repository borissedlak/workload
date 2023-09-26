import time
from datetime import datetime
from enum import Enum

import networkx as nx
import numpy as np
import pandas as pd
import pgmpy
import requests
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork


# class FPS(Enum):
#     TWELVE = 12
#     GREEN = 2
#     BLUE = 3


def diffAsStringInMS(a: datetime, b: datetime):
    return int((a - b).microseconds / 1000)


def getExecutionTime(a: datetime, b: datetime):
    if a is not None and b is not None:
        # print(f' {name} took {diffAsStringInMS(a, b)}ms')
        return diffAsStringInMS(a, b)
    return 0


def write_execution_times(write_store, number_threads=1):
    f = open(f'../data/Performance.csv', 'w+')
    f.write(
        'execution_time,timestamp,cpu_utilization,memory_usage,pixel,fps,success,distance,consumption,stream_count\n')

    sum_other_elements = 0
    for tup in write_store[number_threads:]:
        sum_other_elements += tup[7]
    avg_distance = sum_other_elements / (len(write_store) - number_threads)
    for i in range(number_threads):
        write_store[i] = write_store[i][:7] + (np.floor(avg_distance),) + write_store[i][8:]

    for (delta, ts, cpu, memory, pixel, fps, detected, distance, consumption, thread_number) in write_store:
        f.write(f'{delta},{ts},{cpu},{memory},{pixel},{fps},{detected},{distance},{consumption},{thread_number}\n')

    f.close()
    print("Performance file exported")

    # upload_file()


def get_center_from_box(box):
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


# TODO: Maybe revert and simply take the absolute pixel distance and find some use case where this matches
def get_relative_distance_between_points(p1, p2, img):
    p1_x, p1_y = p1
    p2_x, p2_y = p2

    # The intention was to get the relative distance in different resolutions. The results match quite, but are a bit off
    return np.ceil(np.linalg.norm(np.array([p1_x / img.shape[1], p1_y / img.shape[0]]) - np.array(
        [p2_x / img.shape[1], p2_y / img.shape[0]])) * 1000)
    # return math.ceil(math.sqrt(((p1_x - p2_x)/img.shape[1])**2 + ((p1_y - p2_y)/img.shape[0])**2) * 1000)


# TODO: Provide some regression based on existing data
def get_consumption():
    return 100


def upload_file():
    # The API endpoint to communicate with
    url_post = "http://192.168.1.153:5000/upload"

    # A POST request to tthe API
    files = {'file': open('../data/Performance.csv', 'rb')}
    post_response = requests.post(url_post, files=files)

    # Print the response
    post_response_json = post_response.content
    print(post_response_json)


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


def prepare_samples(samples):
    samples['bitrate'] = samples['fps'] * samples['pixel']
    samples['in_time'] = samples['execution_time'] <= (1000 / samples['fps'])
    # samples['in_time'] = samples['in_time'].astype(bool)
    samples['distance'] = samples['distance'].astype(int)
    samples['cpu_utilization'] = pd.cut(samples['cpu_utilization'], bins=[0, 50, 70, 90, 100],
                                        labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)
    samples['memory_usage'] = pd.cut(samples['memory_usage'], bins=[0, 50, 70, 90, 100],
                                     labels=['Low', 'Mid', 'High', 'Very High'], include_lowest=True)

    del samples['timestamp']
    del samples['execution_time']
    return samples


def print_BN(bn: BayesianNetwork | pgmpy.base.DAG, root=None, try_visualization=False, vis_ls=None, save=False,
             name=None, show=True, color_map=None):
    if vis_ls is None:
        vis_ls = ["fdp"]
    else:
        vis_ls = vis_ls

    if name is None:
        name = root

    if try_visualization:
        vis_ls = ['neato', 'dot', 'twopi', 'fdp', 'sfdp', 'circo']

    for s in vis_ls:
        pos = graphviz_layout(bn, root=root, prog=s)
        nx.draw(
            bn, pos, with_labels=True, arrowsize=20, node_size=1500,  # alpha=1.0, font_weight="bold",
            node_color=color_map
        )
        if save:
            plt.box(False)
            plt.savefig(f"{name}.png", dpi=400, bbox_inches="tight")  # default dpi is 100
        if show:
            plt.show()

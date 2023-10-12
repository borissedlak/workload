import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter

from FGCS import util_fgcs
from util_fgcs import prepare_samples, print_execution_time

# list1 = {
#     '10800', '10920', '12600', '1680', '2160', '2520', '2640', '3120',
#     '3240', '3360', '3600', '3960', '4200', '4320', '4680', '5040',
#     '5280', '5400', '5880', '6240', '6480', '6600', '7200', '7560',
#     '7800', '7920', '9000', '9240', '9360'
# }
#
# list2 = [
#     '10800', '10920', '12600', '2520', '2640', '3240', '3360', '3600',
#     '3960', '4200', '5040', '5280', '5400', '5880', '6240', '6480',
#     '6600', '7200', '7560', '7800', '7920', '9000', '9240', '9360'
# ]
#
# # Find elements that are only in the first list
# only_in_list1 = set(list1) - set(list2)
#
# # Print the elements that are only in the first list
# print(only_in_list1)


br = str(18 * 360)
stream = '1'

model_Xavier_CPU = XMLBIFReader("model_Xavier_CPU.xml").get_model()
raw_data_Xavier_CPU = pd.read_csv("backup_entire_data_Xavier_CPU.csv")
curated_data_Xavier_CPU = prepare_samples(raw_data_Xavier_CPU, t_distance=50, t_total_bitrate=(420 * 30 * 10))
curated_data_Xavier_CPU['consumption'] = '22'

inference = VariableElimination(model_Xavier_CPU)
ra = util_fgcs.get_true(
    inference.query(variables=["in_time", "network"], evidence={'bitrate': br, 'stream_count': stream}))
print(ra)

model_Laptop = XMLBIFReader("model_Laptop.xml").get_model()
raw_data_Laptop = pd.read_csv("backup_entire_data_Laptop.csv")
curated_data_Laptop = prepare_samples(raw_data_Laptop, t_distance=50, t_total_bitrate=(420 * 30 * 10))
curated_data_Laptop['consumption'] = '7'

inference = VariableElimination(model_Laptop)
ra = util_fgcs.get_true(
    inference.query(variables=["in_time", "network"], evidence={'bitrate': br, 'stream_count': stream}))
print(ra)

filtered_data = curated_data_Xavier_CPU[
    ~curated_data_Xavier_CPU['bitrate'].isin(['3120', '4680', '1680', '2160', '4320'])]
past_data_length = len(raw_data_Laptop)  # 19748


@print_execution_time
def merge():
    model_Laptop.fit_update(filtered_data, n_prev_samples=past_data_length)


merge()
inference = VariableElimination(model_Laptop)
ra = util_fgcs.get_true(
    inference.query(variables=["in_time", "network"], evidence={'bitrate': br, 'stream_count': stream}))
print(ra)

model_name = 'model_Xavier_GPU_merged.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

model_Laptop.fit(curated_data_Xavier_CPU)
model_name = 'model_Xavier_CPU.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

model_Laptop.fit(curated_data_Laptop)
model_name = 'model_Laptop.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

raw_data_Nano = pd.read_csv("backup_entire_data_Xavier_CPU.csv")
curated_data_Nano = prepare_samples(raw_data_Nano, t_distance=50, t_total_bitrate=(420 * 30 * 10))
model_Laptop.fit(curated_data_Nano)
model_name = 'model_Nano.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

raw_data_Orin = pd.read_csv("backup_entire_data_Orin.csv")
curated_data_Orin = prepare_samples(raw_data_Orin, t_distance=50, t_total_bitrate=(420 * 30 * 10))
model_Laptop.fit(curated_data_Orin)
model_name = 'model_Orin.xml'
XMLBIFWriter(model_Laptop).write_xmlbif(model_name)
print(f"Model exported as '{model_name}'")

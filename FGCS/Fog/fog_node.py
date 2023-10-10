import csv
from datetime import datetime

from flask import Flask, request

from ScalingModel import ScalingModel

app = Flask(__name__)
csv_file_path_app = "slo_stream_results.csv"
csv_file_path_system = "system_load_results.csv"

counter = 0
stream = 1

scm = ScalingModel()
scm.shuffle_load(25)


@app.route("/stats")
def hello():
    global counter, stream
    pixel = int(request.args.get('pixel'))
    fps = int(request.args.get('fps'))
    pv = round(float(request.args.get('pv')), 2)
    ra = round(float(request.args.get('ra')), 2)
    threads = int(request.args.get('threads'))
    device_name = request.args.get('device_name')
    gpu = int(request.args.get('gpu'))

    with open(csv_file_path_app, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([datetime.now(), pixel, fps, pv, ra, threads, device_name, gpu])

    # counter += 1
    # if counter >= 100:
    #     counter = 0
    #
    #     if stream <= 14:
    #         stream += 1
    #     else:
    #         stream = 1

    stream = scm.get_assigned_streams(device_name=device_name, gpu=gpu)

    return str(stream) + ",0"


@app.route("/system")
def system():
    cpu = int(request.args.get('cpu'))
    device_name = request.args.get('device_name')
    disabled_aci = request.args.get('disabled_aci')
    gpu_available = request.args.get('gpu_available')

    with open(csv_file_path_system, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([datetime.now(), cpu, device_name, disabled_aci, gpu_available])

    return "success"


app.run(host='0.0.0.0', port=8080)

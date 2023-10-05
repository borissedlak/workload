import csv
from datetime import datetime

from flask import Flask, request

app = Flask(__name__)
csv_file_path_app = "slo_stream_results.csv"
csv_file_path_system = "system_load_results.csv"


@app.route("/stats")
def hello():
    pixel = int(request.args.get('pixel'))
    fps = int(request.args.get('fps'))
    pv = round(float(request.args.get('pv')), 2)
    ra = round(float(request.args.get('ra')), 2)
    threads = int(request.args.get('threads'))
    device_name = request.args.get('device_name')

    with open(csv_file_path_app, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([datetime.now(), pixel, fps, pv, ra, threads, device_name])

    # TODO: Make the regression and divide optimal number of threads
    return "1,0"

@app.route("/system")
def system():
    cpu = int(request.args.get('cpu'))
    device_name = request.args.get('device_name')
    disabled_aci = request.args.get('disabled_aci')
    gpu_available = request.args.get('gpu_available')

    with open(csv_file_path_system, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([datetime.now(), cpu, device_name, disabled_aci,gpu_available])

    return "success"


app.run(host='0.0.0.0', port=8080)

import csv
import threading
from datetime import datetime

from flask import Flask, request

from ScalingModel import ScalingModel

app = Flask(__name__)
csv_file_path_app = "slo_stream_results.csv"
csv_file_path_system = "system_load_results.csv"

counter = 0
stream = 1

NUMBER_STREAMS = 25

scm = ScalingModel()
scm.shuffle_load(NUMBER_STREAMS)


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
    surprise = int(round(float(request.args.get('surprise')), 0))

    with open(csv_file_path_app, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([datetime.now(), pixel, fps, pv, ra, threads, device_name, gpu, surprise])

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


def run_server():
    app.run(host='0.0.0.0', port=8080)


background_thread = threading.Thread(target=run_server)
background_thread.daemon = True
background_thread.start()

while True:
    user_input = input()

    if user_input == "p":
        scm.print_current_assignment()
    elif user_input == "r":
        scm.shuffle_load(NUMBER_STREAMS)
    elif user_input.startswith("o: "):
        override_text = user_input[3:]
        real_assignment = eval(override_text)
        scm.override_assignment(real_assignment)
        scm.print_current_assignment()

#1 Inferred) o: {('Laptop', 0): 9, ('Orin', 1): 9, ('Xavier', 0): 1, ('Xavier', 1): 5, ('Nano', 0): 1}
#2 Single) o: {('Laptop', 0): 1, ('Orin', 1): 1, ('Xavier', 0): 1, ('Xavier', 1): 1, ('Nano', 0): 1}
#3 Random) o: {('Laptop', 0): 4, ('Orin', 1): 4, ('Xavier', 0): 5, ('Xavier', 1): 8, ('Nano', 0): 3}
#4 Equal) o: {('Laptop', 0): 5, ('Orin', 1): 5, ('Xavier', 0): 5, ('Xavier', 1): 5, ('Nano', 0): 5}
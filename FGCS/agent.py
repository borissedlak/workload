import os
import sys
import traceback

import ModelParser
import Models
import util_fgcs
from ACI import ACI
from VideoProcessor import *
from http_client import HttpClient

HTTP_SERVER = os.environ.get('HTTP_SERVER')
if HTTP_SERVER:
    print(f'Found ENV value for HTTP_SERVER: {HTTP_SERVER}')
else:
    HTTP_SERVER = "127.0.0.1"
    print(f"Didn't find ENV value for HTTP_SERVER, default to: {HTTP_SERVER}")

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

CLEAN_RESTART = os.environ.get('CLEAN_RESTART')
if CLEAN_RESTART:
    print(f'Found ENV value for CLEAN_RESTART: {CLEAN_RESTART}')
else:
    CLEAN_RESTART = False
    print(f"Didn't find ENV value for CLEAN_RESTART, default to: {CLEAN_RESTART}")

DISABLE_ACI = os.environ.get('DISABLE_ACI')
if DISABLE_ACI:
    print(f'Found ENV value for DISABLE_ACI: {DISABLE_ACI}')
else:
    DISABLE_ACI = False
    print(f"Didn't find ENV value for DISABLE_ACI, default to: {DISABLE_ACI}")

SEND_SYSTEM_STATS = os.environ.get('SEND_SYSTEM_STATS')
if SEND_SYSTEM_STATS:
    print(f'Found ENV value for SEND_SYSTEM_STATS: {SEND_SYSTEM_STATS}')
else:
    SEND_SYSTEM_STATS = False
    print(f"Didn't find ENV value for SEND_SYSTEM_STATS, default to: {SEND_SYSTEM_STATS}")

SHOW_IMG = os.environ.get('SHOW_IMG')
if SHOW_IMG:
    print(f'Found ENV value for SHOW_IMG: {SHOW_IMG}')
else:
    SHOW_IMG = False
    print(f"Didn't find ENV value for SHOW_IMG, default to: {SHOW_IMG}")

privacy_model = ModelParser.parseModel(Models.model_1)
chain = privacy_model.getChainForSource("video", "webcam")
detector = VideoProcessor(device_name=DEVICE_NAME, privacy_chain=chain, display_stats=False, simulate_fps=True)

model_name = None if CLEAN_RESTART else f"model_{DEVICE_NAME}.xml"
aci = ACI(distance_slo=100, network_slo=(420 * 30 * 10), load_model=model_name, device_name=DEVICE_NAME, show_img=SHOW_IMG)

c_pixel = ACI.pixel_list[1]
c_fps = ACI.fps_list[2]
c_mode = None

new_data = False
override_next_config = None

inferred_config_hist = []
util_fgcs.clear_performance_history('../data/Performance_History.csv')
video_path = "../video_data/"

http_client = HttpClient(HOST=HTTP_SERVER)
http_client.override_stream_config(1)

# Function for the background loop
def processing_loop():
    global c_pixel, c_fps, new_data
    while True:
        detector.processVideo(video_path=video_path,
                              video_info=(c_pixel, c_fps, http_client.get_latest_stream_config()),
                              show_result=False)
        if SEND_SYSTEM_STATS:
            http_client.send_system_stats(int(psutil.cpu_percent()), DEVICE_NAME, DISABLE_ACI, detector.gpu_available)
        new_data = True


background_thread = threading.Thread(target=processing_loop)
background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
background_thread.start()


class ACIBackgroundThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits

    def run(self):
        global c_pixel, c_fps, new_data, override_next_config, inferred_config_hist
        while True:
            try:
                if new_data:
                    new_data = False
                    d_threads = http_client.get_latest_stream_config()[0]
                    (new_pixel, new_fps, pv, ra, real, surprise) = aci.iterate(str(d_threads))
                    past_pixel, past_fps, past_pv, past_ra = real
                    http_client.send_app_stats(past_pixel, past_fps, past_pv, past_ra, d_threads, DEVICE_NAME,
                                               detector.gpu_available, surprise)
                    inferred_config_hist.append((new_pixel, new_fps))
                    if override_next_config:
                        c_pixel, c_fps = override_next_config
                        override_next_config = None
                    else:
                        c_pixel, c_fps = new_pixel, new_fps
                else:
                    time.sleep(0.2)
            except Exception as e:
                # Capture the traceback as a string
                error_traceback = traceback.format_exc()

                # Print the traceback
                print("Error Traceback:")
                print(error_traceback)

                util.print_in_red(f"ACI Background thread encountered an exception:{e}")
                # self.start()


if not DISABLE_ACI:
    background_thread = ACIBackgroundThread()
    background_thread.start()

# Main loop to read commands from the CLI
while True:
    user_input = input()

    # Check if the user entered a command
    if user_input:
        threads = http_client.get_latest_stream_config()[0]
        if user_input == "+":
            http_client.override_stream_config(threads + 1)
        elif user_input == "-":
            http_client.override_stream_config(1 if threads == 1 else (threads - 1))
        elif user_input == "i":
            aci.initialize_bn()
        elif user_input == "e":
            aci.export_model(DEVICE_NAME)
        # elif user_input == "q":
        #     aci.export_model()
        #     sys.exit()

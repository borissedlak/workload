import sys
import traceback

import ModelParser
import Models
import util_fgcs
from ACI import ACI
from VideoProcessor import *
from http_client import HttpClient

device_name = "Laptop"

privacy_model = ModelParser.parseModel(Models.model_1)
chain = privacy_model.getChainForSource("video", "webcam")
detector = VideoProcessor(device_name=device_name, privacy_chain=chain, display_stats=False, simulate_fps=True)

aci = ACI(distance_slo=30, network_slo=(420 * 30 * 4), load_model="model.xml")

c_pixel = ACI.pixel_list[4]
c_fps = ACI.fps_list[2]
c_mode = None

new_data = False
override_next_config = None

inferred_config_hist = []
util_fgcs.clear_performance_history('../data/Performance_History.csv')

http_client = HttpClient()

# Function for the background loop
def processing_loop():
    global c_pixel, c_fps, new_data
    while True:
        detector.processVideo(video_path="../video_data/",
                              video_info=(c_pixel, c_fps, http_client.get_latest_stream_config()),
                              show_result=False)
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
                    d_threads = http_client.get_latest_stream_config()
                    (new_pixel, new_fps, pv, ra) = aci.iterate(str(d_threads))
                    http_client.send_stats(new_pixel, new_fps, pv, ra, d_threads, device_name)
                    inferred_config_hist.append((new_pixel, new_fps))
                    if override_next_config:
                        c_pixel, c_fps = override_next_config
                        override_next_config = None
                    else:
                        1+1# c_pixel, c_fps = new_pixel, new_fps
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


background_thread = ACIBackgroundThread()
background_thread.start()

# Main loop to read commands from the CLI
while True:
    user_input = input()

    # Check if the user entered a command
    if user_input:
        threads = http_client.get_latest_stream_config()
        if user_input == "+":
            http_client.override_stream_config(threads + 1)
        elif user_input == "-":
            http_client.override_stream_config(1 if threads == 1 else (threads - 1))
        elif user_input == "i":
            aci.initialize_bn()
        elif user_input == "e":
            aci.export_model()
        elif user_input == "q":
            aci.export_model()
            # TODO: Upload file
            sys.exit()

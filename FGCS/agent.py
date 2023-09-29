import sys
import traceback

import ModelParser
import Models
from FGCS import util_fgcs
from FGCS.ACI import ACI
from VideoProcessor import *

privacy_model = ModelParser.parseModel(Models.model_1)
chain = privacy_model.getChainForSource("video", "webcam")
detector = VideoProcessor(privacy_chain=chain, display_stats=False, simulate_fps=True)

aci = ACI()

c_pixel = ACI.pixel_list[0]
c_fps = ACI.fps_list[0]
c_mode = None

d_threads = 1
new_data = False
override_next_config = None

inferred_config_hist = []
util_fgcs.clear_performance_history('../data/Performance_History.csv')

# Function for the background loop
def processing_loop():
    global c_pixel, c_fps, new_data
    while True:
        detector.processVideo(video_path="../video_data/",
                              video_info=(c_pixel, c_fps, d_threads),
                              show_result=False)
        new_data = True

        # (new_pixel, new_fps) = aci.iterate(c_pixel, c_fps)
        # c_pixel = new_pixel
        # c_fps = new_fps


background_thread = threading.Thread(target=processing_loop)
background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
background_thread.start()


class ACIBackgroundThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Set the thread as a daemon so it will exit when the main program exits

    def run(self):
        global c_pixel, c_fps, new_data, override_next_config, inferred_config_hist
        while True:
            try:
                if new_data:
                    new_data = False
                    (new_pixel, new_fps) = aci.iterate(d_threads)
                    inferred_config_hist.append((new_pixel, new_fps))

                    if override_next_config:
                        c_pixel, c_fps = override_next_config
                        override_next_config = None
                    else:
                        c_pixel, c_fps = new_pixel, new_fps
                else:
                    time.sleep(0.5)
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
        if user_input == "+":
            d_threads += 1
        elif user_input == "-":
            d_threads = 1 if d_threads == 1 else (d_threads - 1)
        elif user_input == "i":
            aci.initialize_bn()
            continue
        if user_input == "e":
            aci.export_model()
        if user_input == "q":
            aci.export_model()
            #TODO: Upload file
            sys.exit()
        # override_next_config = ACI.pixel_list[d_threads - 1], ACI.fps_list[d_threads - 1]

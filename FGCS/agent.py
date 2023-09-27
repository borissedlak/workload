import time

import ModelParser
import Models
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


# TODO: Maybe run the ACI in another thread
def aci_loop():
    global c_pixel, c_fps, new_data
    while True:
        if new_data:
            new_data = False
            (new_pixel, new_fps) = aci.iterate(c_pixel, c_fps)
            c_pixel = new_pixel
            c_fps = new_fps
        else:
            time.sleep(0.2)


background_thread = threading.Thread(target=aci_loop)
background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
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
            aci.initialize()

        c_fps = ACI.fps_list[d_threads-1]
        c_pixel = ACI.pixel_list[d_threads-1]

        print(f"Changed to {d_threads} Threads")

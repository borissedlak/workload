import sys

import Models
from VideoProcessor import *

privacy_model_1 = ModelParser.parseModel(Models.model_1)
chain_1 = privacy_model_1.getChainForSource("video", "webcam")
detector_1 = VideoProcessor(privacy_chain=chain_1, display_stats=False, write_stats=True, simulate_fps=True)

c_pixel = "240p"
c_fps = 12
c_mode = None

d_threads = 1


# Function for the background loop
def processing_loop():
    while True:
        detector_1.processVideo(video_path="../video_data/",
                                video_info=(c_pixel, c_fps, d_threads),
                                model_name="model_1", show_result=False)


background_thread = threading.Thread(target=processing_loop)
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
        print(f"Changed to {d_threads} Threads")

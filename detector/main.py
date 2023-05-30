import itertools
import sys

import ModelParser
import Models
from VideoDetector import *

# privacy_model = ModelParser.parseModel("video:{'tag':'recording'}-->Gender_Trigger:{'prob':0.85}-->Blur_Area_Simple:{'blocks':5}")
privacy_model_1 = ModelParser.parseModel(Models.model_1)
privacy_model_1_1 = ModelParser.parseModel(Models.model_1_1)
privacy_model_1_50 = ModelParser.parseModel(Models.model_1_50)
privacy_model_2 = ModelParser.parseModel(Models.model_2)
privacy_model_3 = ModelParser.parseModel(Models.model_3)
privacy_model_4 = ModelParser.parseModel(Models.model_4)

chain_1 = privacy_model_1.getChainForSource("video", "webcam")
chain_1_1 = privacy_model_1_1.getChainForSource("video", "webcam")
chain_1_50 = privacy_model_1_50.getChainForSource("video", "webcam")
chain_2 = privacy_model_2.getChainForSource("video", "webcam")
chain_3 = privacy_model_3.getChainForSource("video", "webcam")
chain_4 = privacy_model_4.getChainForSource("video", "webcam")

detector_1 = VideoDetector(privacy_chain=chain_1, display_stats=True, write_stats=True, simulate_fps=False)
detector_1_1 = VideoDetector(privacy_chain=chain_1_1, display_stats=True, write_stats=True)
detector_1_50 = VideoDetector(privacy_chain=chain_1_50, display_stats=True, write_stats=True)
detector_2 = VideoDetector(privacy_chain=chain_2, display_stats=True, write_stats=True)
detector_3 = VideoDetector(privacy_chain=chain_3, display_stats=True, write_stats=True)
detector_4 = VideoDetector(privacy_chain=chain_4, display_stats=True, write_stats=True)

# detector_1.processImage("../producer/demo_files/images/zoom_call.jpg", show_result=True)
# detector_1_1.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_1_1", show_result=True)
# detector_1_50.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_1_50", show_result=True)

# ("100p", 16), ("120p", 16), ("240p", 16), ("360p", 16), ("480p", 16), ("720p", 16),
#                                     ("360p", 30), ("360p", 45), ("360p", 60), ("480p", 30), ("480p", 45), ("480p", 60)

detector_1.processVideo(video_path="../producer/demo_files/videos/",
                        video_info=list(
                            itertools.product(["100p", "120p", "240p", "360p", "480p", "720p"], [16, 30, 45, 60])),
                        model_name="model_1", show_result=False)
# detector_2.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_1", show_result=False)
# detector_1.processVideo(video_path="../producer/demo_files/videos/", video_name="video_loop_1", model_name="model_1", show_result=False)
# detector_2.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_2", show_result=True)
# detector_3.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_3", show_result=True)
# detector_4.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_4", show_result=True)

# detector_1.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_1", show_result=True)
# detector_2.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_2", show_result=True)
# detector_3.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_3", show_result=True)
# detector_4.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_4", show_result=True)

sys.exit()

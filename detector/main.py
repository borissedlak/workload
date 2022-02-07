import sys

import ModelParser
from VideoDetector import *

# privacy_model = ModelParser.parseModel("video:{'tag':'recording'}-->Gender_Trigger:{'prob':0.85}-->Blur_Area_Simple:{'blocks':5}")
privacy_model = ModelParser.parseModel("video:{'tag':'webcam'}-->Max_Spec_Resize:{'max_width':800}-->Face_Trigger:{'prob':0.9}-->Blur_Area_Pixelate:{'blocks':5}")
chain = privacy_model.getChainForSource("video", "webcam")
detector = VideoDetector(privacy_chain=chain, display_stats=True, write_stats=True)

# detector.processImage("../producer/demo_files/images/scooter_vietnam.jpg", show_result=True)
detector.processVideo("../producer/demo_files/videos/webcam_480p_16fps.mp4", show_result=True)
# detector.processVideo("../producer/demo_files/videos/head-pose-detection.mp4", show_result=True)

sys.exit()

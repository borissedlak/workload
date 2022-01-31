import ModelParser
import Models
from VideoDetector import *

privacy_model = ModelParser.parseModel("video:{'tag':'recording'}-->Gender_Trigger:{'prob':0.85}-->Blur_Area_Simple:{'blocks':5}")
chain = privacy_model.getChainForSource("video", "recording")
detector = VideoDetector(privacy_chain=chain, show_stats=True)

detector.processImage("../producer/demo_files/images/uaf.jpg", show_result=True)
# detector.processVideo("../producer/demo_files/videos/lukas-detection.mp4", show=True)  # FPS: CPU 10, GPU 32
# detector.processVideo("../producer/demo_files/videos/head-pose-detection.mp4", show=True)  # FPS: CPU 12, GPU 52

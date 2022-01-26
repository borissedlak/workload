import ModelParser
import Models
from VideoDetector import *

privacy_model = ModelParser.parseModel(Models.faces_pixelate_with_resize)
privacy_chain = privacy_model.getChainForSource("video", "webcam")
detector = VideoDetector(use_cuda=True, privacy_chain=privacy_chain, show_stats=True)

detector.processImage("../producer/demo_files/images/bruder.jpg", show_result=True)
# detector.processVideo("../producer/demo_files/videos/lukas-detection.mp4", show=True)  # FPS: CPU 10, GPU 32
# detector.processVideo("../producer/demo_files/videos/head-pose-detection.mp4", show=True)  # FPS: CPU 12, GPU 52

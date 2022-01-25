import ModelParser
from VideoDetector import *

privacy_model = ModelParser.parseModel(ModelParser.test_string)
detector = VideoDetector(use_cuda=True, output_width=700, privacy_model=privacy_model)

detector.processImage("../producer/demo_files/images/bruder.jpg", show=True)
# detector.processVideo("../producer/demo_files/videos/lukas-detection.mp4", show=True)  # FPS: CPU 10, GPU 32
# detector.processVideo("../producer/demo_files/videos/head-pose-detection.mp4", show=True)  # FPS: CPU 12, GPU 52

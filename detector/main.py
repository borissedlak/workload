from VideoDetector import *

detector = VideoDetector(use_cuda=True, output_width=500, confidence_threshold=0.3)

detector.processImage("../producer/demo_files/images/bruder.jpg", show=True)
# detector.processVideo("../producer/demo_files/videos/lukas-detection.mp4", show=True)  # FPS: CPU 10, GPU 32
# detector.processVideo("../producer/demo_files/videos/head-pose-detection.mp4")  # FPS: CPU 12, GPU 52

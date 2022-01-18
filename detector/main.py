from Detector import *

detector = Detector(use_cuda=True, output_width=500, confidence_threshold=0.3)

# detector.processImage("../server/demo/uaf.jpg")
# detector.processVideo("../server/demo/lukas-detection.mp4") # FPS: CPU 10, GPU 32
detector.processVideo("../producer/demo_files/head-pose-detection.mp4")  # FPS: CPU 12, GPU 52

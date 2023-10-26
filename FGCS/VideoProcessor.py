# from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
import random
import threading
import time
from datetime import datetime
from os.path import join, dirname

import onnxruntime as ort

import cv2
import psutil
from imutils.video import FPS

import util_fgcs as util
from ConsumptionRegression import ConsRegression
from ModelParser import PrivacyChain
from Triggers import Face_Trigger
from util_fgcs import getExecutionTime, write_execution_times


class VideoProcessor:
    def __init__(self, device_name, privacy_chain: PrivacyChain = None, confidence_threshold=0.5,
                 display_stats=False, simulate_fps=False):
        self.img = None
        self.confidence_threshold = confidence_threshold
        self.privacy_chain = privacy_chain
        self.display_stats = display_stats
        self.write_store = None
        self.resolution = 0
        self.simulate_fps = simulate_fps
        self.old_center = (0, 0)
        self.distance = 0
        self.consumption_regression = ConsRegression(device_name)
        self.gpu_available = self.detect_gpu()

    def processVideo(self, video_path, video_info, show_result=False):

        self.write_store = []

        (source_res, source_fps, stream_config) = video_info
        (number_threads, latency_factor) = stream_config
        print(f"Now processing: {source_res}p{source_fps} with {number_threads} Thread(s)")

        available_time_frame = (1000 / source_fps)
        cap = cv2.VideoCapture(video_path + str(source_res) + "p_" + str(source_fps) + ".mp4")
        if not cap.isOpened():
            print("Error opening video ...")
            return

        skip_x_frames = source_fps * random.randint(0, 9)
        # start_time = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_x_frames)
        # print((time.time() - start_time) * 1000)

        (success, self.img) = cap.read()

        self.resolution = self.img.shape[0] * self.img.shape[1]
        self.old_center = (self.img.shape[1] / 2, self.img.shape[0] / 2)

        fps = FPS().start()

        for _ in range(source_fps):
            start_time = datetime.now()

            threads = []
            for _ in range(number_threads):
                thread = threading.Thread(target=self.processFrame_v3, args=(source_fps, number_threads, latency_factor))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

            # Adding one CPU Utilization for all entries in the thread
            cpu = psutil.cpu_percent()
            consumption = self.consumption_regression.predict(cpu, self.gpu_available)
            last_x_items = list(self.write_store)[-number_threads:]
            self.write_store = self.write_store[:-number_threads]
            for item in last_x_items:
                i = list(item)
                i[2] = cpu
                i[8] = consumption
                self.write_store.append(tuple(i))

            fps.update()
            (success, self.img) = cap.read()
            if self.img is not None and self.simulate_fps:
                overall_time = int((datetime.now() - start_time).microseconds / 1000)
                if overall_time < available_time_frame:
                    time.sleep((available_time_frame - overall_time) / 1000)

        cap.release()
        fps.stop()
        if self.display_stats:
            print("Elapsed time: {:.2f}s".format(fps.elapsed()))
            print("FPS: {:.2f}".format(fps.fps()))

        write_execution_times(self.write_store, number_threads)

    def processFrame_v3(self, fps=None, number_threads=1, latency_factor =1):
        boxes = None
        detected = False
        overall_time = datetime.now()

        for cmA in self.privacy_chain.cmAs:

            if cmA.isTrigger():
                args_with_boxes = {'boxes': boxes}
                args_with_boxes.update(cmA.args)
                _, boxes = cmA.commandFunction.check(self.img, options=args_with_boxes)

                if boxes is None or boxes.size == 0:
                    detected = False
                else:
                    detected = True
                    new_center = util.get_center_from_box(boxes[0])
                    d = util.get_relative_distance_between_points(new_center, self.old_center, self.img)
                    if d > 0:
                        self.distance = d
                        self.old_center = new_center

            if cmA.isTransformation():
                args_with_boxes = {'boxes': boxes}
                args_with_boxes.update(cmA.args)
                cmA.commandFunction.transform(self.img, options=args_with_boxes)  # self.img =
                boxes = None

        overall_delta = getExecutionTime(datetime.now(), overall_time)
        overall_delta *= latency_factor
        self.write_store.append((overall_delta, datetime.now(), -1,
                                 psutil.virtual_memory().percent,
                                 self.img.shape[0], fps, detected, self.distance,
                                 -1,
                                 number_threads
                                 ))

    def detect_gpu(self):
        face_detector_onnx = Face_Trigger()
        providers = face_detector_onnx.face_detector.get_providers()
        return 1 if "CUDAExecutionProvider" in providers else 0

# from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
import time
from datetime import datetime

import cv2
import imutils
from imutils.video import FPS

import util
from ModelParser import PrivacyChain
from util import printExecutionTime, write_execution_times
import psutil


class VideoDetector:
    def __init__(self, privacy_chain: PrivacyChain = None, output_width=None, confidence_threshold=0.5,
                 display_stats=False, write_stats=False, simulate_fps=False):
        self.img = None
        self.output_width = output_width
        self.confidence_threshold = confidence_threshold
        self.privacy_chain = privacy_chain
        self.display_stats = display_stats
        self.write_stats = write_stats
        self.write_store = None
        self.resolution = 0
        self.simulate_fps = simulate_fps
        self.old_center = (0, 0)
        self.distance = 0

    def processImage(self, img_path=None, img=None, show_result=False):
        if img_path is not None:
            self.img = cv2.imread(img_path)
        else:
            self.img = img

        self.write_store = {"Overall_Chain": []}
        if self.output_width is not None:
            self.img = imutils.resize(self.img, width=self.output_width)

        # (self.height, self.width) = self.img.shape[:2]

        self.processFrame_v3()

        if show_result:
            # cv2.startWindowThread()
            # cv2.imwrite("../evaluation/figures/test.jpg", self.img)
            cv2.imshow("output", self.img)
            cv2.waitKey(0)

    def processVideo(self, video_path, video_info, model_name, show_result=False, repeat=1):

        if self.write_stats:
            self.write_store = {"Overall_Chain": []}

        for (source_res, source_fps) in video_info:
            for x in range(repeat):
                print(f"Now processing: {source_res}{source_fps} Round {x+1}")
                available_time_frame = (1000 / source_fps)
                cap = cv2.VideoCapture(video_path + source_res + "_" + str(source_fps) + ".mp4")
                if not cap.isOpened():
                    print("Error opening video ...")
                    return
                (success, self.img) = cap.read()
                self.img = imutils.resize(self.img, width=self.output_width)
                self.resolution = self.img.shape[0] * self.img.shape[1]
                self.old_center = (self.img.shape[1] / 2, self.img.shape[0] / 2)
                # (self.height, self.width) = self.img.shape[:2]

                fps = FPS().start()

                while success:
                    start_time = datetime.now()
                    self.processFrame_v3(source_fps)
                    # if show_result:
                    #     cv2.imshow("output", self.img)
                    #
                    # key = cv2.waitKey(1) & 0xFF
                    # if key == ord("q"):
                    #     break

                    fps.update()
                    (success, self.img) = cap.read()
                    if self.img is not None:
                        self.img = imutils.resize(self.img, width=self.output_width)
                        # (self.height, self.width) = self.img.shape[:2]

                        if self.simulate_fps:
                            overall_time = int((datetime.now() - start_time).microseconds / 1000)
                            if overall_time < available_time_frame:
                                time.sleep((available_time_frame - overall_time) / 1000)

                cap.release()
                fps.stop()
                print("Elapsed time: {:.2f}s".format(fps.elapsed()))
                print("FPS: {:.2f}".format(fps.fps()))

        if self.write_stats:
            write_execution_times(self.write_store, "video_loop_1", model_name)

    def processFrame_v3(self, fps=None):
        boxes = None

        overall_time = None
        detected = False
        if self.display_stats:
            overall_time = datetime.now()

        for cmA in self.privacy_chain.cmAs:

            function_name = cmA.commandFunction.function_name
            start_time = None
            if self.display_stats:
                start_time = datetime.now()

            if cmA.isTrigger():
                args_with_boxes = cmA.args | {'boxes': boxes}
                self.img, boxes = cmA.commandFunction.check(self.img, options=args_with_boxes)

                if boxes is None or boxes.size == 0:
                    detected = False
                else:
                    detected = True
                    new_center = util.get_center_from_box(boxes[0])
                    self.distance = util.get_relative_distance_between_points(new_center, self.old_center, self.img)
                    self.old_center = new_center

            if cmA.isTransformation():
                args_with_boxes = cmA.args | {'boxes': boxes}
                self.img = cmA.commandFunction.transform(self.img, options=args_with_boxes)
                boxes = None

            delta = printExecutionTime(function_name, datetime.now(), start_time)

            # if self.write_stats:
            #     if function_name in self.write_store:
            #         self.write_store[function_name].append((delta, datetime.now(), 0, 0, 0, 0,0))
            #     else:
            #         self.write_store[function_name] = []
            #         self.write_store[function_name].append((delta, datetime.now(), 0, 0, 0, 0,0))

            # if self.simulate_fps:
            #     if delta < time_frame:
            #         time.sleep((time_frame - delta) / 1000)

        if self.write_stats:
            overall_delta = printExecutionTime("Overall Chain", datetime.now(), overall_time)
            # unfortunately this seems to slow down the FPS decisively, however, the execution time (delay) stays the same
            # celsius = util.get_cpu_temperature()
            self.write_store["Overall_Chain"].append((overall_delta, datetime.now(), psutil.cpu_percent(),
                                                      psutil.virtual_memory().percent,
                                                      self.resolution, fps, detected, self.distance,
                                                      util.get_consumption()))

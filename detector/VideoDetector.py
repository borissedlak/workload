# from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
from datetime import datetime

import cv2
import imutils
from imutils.video import FPS

from ModelParser import PrivacyChain
from util import printExecutionTime, write_execution_times


class VideoDetector:
    def __init__(self, privacy_chain: PrivacyChain = None, output_width=None, confidence_threshold=0.5,
                 display_stats=False, write_stats=False):
        self.img = None
        self.output_width = output_width
        self.confidence_threshold = confidence_threshold
        self.privacy_chain = privacy_chain
        self.display_stats = display_stats
        self.write_stats = write_stats
        self.write_store = None

    def processImage(self, img_path=None, img=None, show_result=False):
        if img_path is not None:
            self.img = cv2.imread(img_path)
        else:
            self.img = img

        if self.output_width is not None:
            self.img = imutils.resize(self.img, width=self.output_width)

        (self.height, self.width) = self.img.shape[:2]

        self.processFrame_v3()

        if show_result:
            cv2.imshow("output", self.img)
            cv2.waitKey(0)

    def processVideo(self, videoName, show_result=False):
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print("Error opening video ...")
            return
        (success, self.img) = cap.read()
        self.img = imutils.resize(self.img, width=self.output_width)
        (self.height, self.width) = self.img.shape[:2]

        fps = FPS().start()

        if self.write_stats:
            self.write_store = {"Overall_Chain": []}

        while success:
            self.processFrame_v3()
            if show_result:
                cv2.imshow("output", self.img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            fps.update()
            (success, self.img) = cap.read()
            if self.img is not None:
                self.img = imutils.resize(self.img, width=self.output_width)
                (self.height, self.width) = self.img.shape[:2]

        fps.stop()
        print("Elapsed time: {:.2f}".format(fps.elapsed()))
        print("FPS: {:.2f}".format(fps.fps()))
        if self.write_stats:
            write_execution_times(self.write_store)

        cap.release()

    def processFrame_v3(self):
        boxes = None

        overall_time = None
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
            if cmA.isTransformation():
                args_with_boxes = cmA.args | {'boxes': boxes}
                self.img = cmA.commandFunction.transform(self.img, options=args_with_boxes)
                boxes = None

            delta = printExecutionTime(function_name, datetime.now(), start_time)

            if self.write_stats:
                if function_name in self.write_store:
                    self.write_store[function_name].append((delta, datetime.now()))
                else:
                    self.write_store[function_name] = []
                    self.write_store[function_name].append((delta, datetime.now()))

        overall_delta = printExecutionTime("Overall Chain", datetime.now(), overall_time)

        if self.write_stats:
            self.write_store["Overall_Chain"].append((overall_delta, datetime.now()))

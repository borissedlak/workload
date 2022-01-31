# from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

import cv2
import imutils
from imutils.video import FPS

from ModelParser import PrivacyChain
from Transformations import Blur_Face_Pixelate
from Triggers import Face_Trigger, Age_Trigger

face_trigger = Face_Trigger()
age_trigger = Age_Trigger()
blur_pixelate = Blur_Face_Pixelate()


class VideoDetector:
    def __init__(self, privacy_chain: PrivacyChain = None, output_width=None, confidence_threshold=0.5,
                 show_stats=False):
        self.img = None
        self.output_width = output_width
        self.confidence_threshold = confidence_threshold
        self.privacy_chain = privacy_chain
        self.show_stats = show_stats

    def processImage(self, img_path=None, img=None, show_result=False):
        if img_path is not None:
            self.img = cv2.imread(img_path)
        else:
            self.img = img

        if self.output_width is not None:
            self.img = imutils.resize(self.img, width=self.output_width)

        (self.height, self.width) = self.img.shape[:2]

        self.processFrame_v3(stats=self.show_stats)

        if show_result:
            cv2.imshow("outpt", self.img)
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

        while success:
            self.processFrame_v3(stats=self.show_stats)
            if show_result:
                cv2.imshow("outpt", self.img)

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

        cap.release()

    def processFrame_v3(self, stats=False):
        boxes = None

        # TODO: Print total value that chain execution takes

        # TODO: Maybe switch from displaying stats in methods to calculating stats here, way easier!
        #  I can use the current util functions and the names that the function has attached
        #  What is the max time that it might take?? Also is it slower when doing only one image and not a whole video?

        for cmA in self.privacy_chain.cmAs:
            if cmA.isTrigger():
                args_with_boxes = cmA.args | {'boxes': boxes} | {'stats': stats}
                self.img, boxes = cmA.commandFunction.check(self.img, options=args_with_boxes)
            if cmA.isTransformation():
                args_with_boxes = cmA.args | {'boxes': boxes} | {'stats': stats}
                self.img = cmA.commandFunction.transform(self.img, options=args_with_boxes)
                boxes = None

        # self.img, boxes = face_trigger.check(self.img, options={'prob': 0.85, 'stats': stats})
        # self.img, boxes = age_trigger.check(self.img,
        #                                     options={'prob': 0.85, 'label': '(25-32)', 'boxes': boxes, 'debug': True,
        #                                              'stats': stats})
        # self.img = blur_pixelate.transform(self.img, options={'boxes': boxes, 'blocks': 5, 'stats': stats})

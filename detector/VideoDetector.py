# from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

from os.path import dirname, join

import cv2
import imutils
import numpy as np
from cv2 import dnn
from imutils.video import FPS

from Transformations import anonymize_face_pixelate
from age.levi_googlenet import process_frame_v2

protoPath = join(dirname(__file__), "models/res10_300x300_ssd_iter_140000.prototxt")
modelPath = join(dirname(__file__), "models/res10_300x300_ssd_iter_140000.caffemodel")


class VideoDetector:
    def __init__(self, use_cuda=False, output_width=None, confidence_threshold=0.5):
        # I can use caffe, tensorflow, or pytorch
        self.faceModel = cv2.dnn.readNetFromCaffe(protoPath, caffeModel=modelPath)
        self.img = None
        self.output_width = output_width
        self.confidence_threshold = confidence_threshold

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self, img_path=None, img=None, show=False):
        if img_path is not None:
            self.img = cv2.imread(img_path)
        else:
            self.img = img

        if self.output_width is not None:
            self.img = imutils.resize(self.img, width=self.output_width)

        (self.height, self.width) = self.img.shape[:2]

        # self.processFrame()
        self.img = process_frame_v2(self.img)

        if show:
            cv2.imshow("outpt", self.img)
            cv2.waitKey(0)

    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print("Error opening video ...")
            return
        (success, self.img) = cap.read()
        self.img = imutils.resize(self.img, width=self.output_width)
        (self.height, self.width) = self.img.shape[:2]

        fps = FPS().start()

        while success:
            process_frame_v2(self.img)
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
        # cv2.destroyAllWindows()

    def processFrame(self):
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123),
                                     swapRB=False, crop=False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            if (predictions[0, 0, i, 2]) > self.confidence_threshold:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                # highlight section
                # cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), -2)  # negative thickness means fill

                # blur section
                face_pixels = self.img[ymin:ymax, xmin:xmax]
                # face_blurred = self.anonymize_face_simple(face_pixels)
                face_blurred = anonymize_face_pixelate(face_pixels, blocks=5)
                self.img[ymin:ymax, xmin:xmax] = face_blurred

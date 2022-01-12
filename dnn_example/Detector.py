import math

import imutils
import numpy as np
import cv2
from cv2 import dnn
from imutils.video import FPS

from os.path import dirname, join

protoPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.prototxt")
modelPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")


class Detector:
    def __init__(self, use_cuda=False, output_width=200, confidence_threshold=0.5):
        self.faceModel = cv2.dnn.readNetFromCaffe(protoPath, caffeModel=modelPath)
        self.img = None
        self.output_width = output_width
        self.confidence_threshold = confidence_threshold

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self, img_path=None, img=None):
        if img_path is not None:
            raw_image = cv2.imread(img_path)
        else:
            raw_image = img

        raw_height, raw_width = raw_image.shape[:2]
        self.img = cv2.resize(raw_image,
                              (math.floor(600 * raw_width / raw_height), math.floor(600 * raw_height / raw_width)))
        (self.height, self.width) = self.img.shape[:2]

        self.processFrame()

        # cv2.imshow("outpt", self.img)
        # cv2.waitKey(0)

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
            self.processFrame()
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
        cv2.destroyAllWindows()

    def processFrame(self):
        # frame = imutils.resize(self.img, width=800)
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 177.0, 123),
                                     swapRB=False, crop=False)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            if (predictions[0, 0, i, 2]) > self.confidence_threshold:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int")

                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # cv2.blur(self.img)

    def anonymize_face_pixelate(self, image, blocks=3):
        # divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
        # loop over the blocks in both the x and y direction
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # compute the starting and ending (x, y)-coordinates
                # for the current block
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                # extract the ROI using NumPy array slicing, compute the
                # mean of the ROI, and then draw a rectangle with the
                # mean RGB values over the ROI in the original image
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (B, G, R), -1)
        # return the pixelated blurred image
        return image

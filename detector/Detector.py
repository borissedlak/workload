# from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

from os.path import dirname, join

import cv2
import imutils
import numpy as np
from cv2 import dnn
from imutils.video import FPS

protoPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.prototxt")
modelPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")


class Detector:
    def __init__(self, use_cuda=False, output_width=None, confidence_threshold=0.5):
        # I can use caffe, tensorflow, or pytorch
        self.faceModel = cv2.dnn.readNetFromCaffe(protoPath, caffeModel=modelPath)
        self.img = None
        self.output_width = output_width
        self.confidence_threshold = confidence_threshold

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self, img_path=None, img=None):
        if img_path is not None:
            self.img = cv2.imread(img_path)
        else:
            self.img = img

        if self.output_width is not None:
            self.img = imutils.resize(self.img, width=self.output_width)

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
            # cv2.imshow("outpt", self.img)

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
                face_blurred = self.anonymize_face_pixelate(face_pixels, blocks=5)
                self.img[ymin:ymax, xmin:xmax] = face_blurred

    # ##################################################################################################################
    # ########  Transformation methods  ################################################################################
    # ##################################################################################################################

    @staticmethod
    def anonymize_face_simple(image, factor=3.0):
        if image is None or image.size == 0:
            return image

        # automatically determine the size of the blurring kernel based
        # on the spatial dimensions of the input image
        (h, w) = image.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)
        # ensure the width of the kernel is odd
        if kW % 2 == 0:
            kW -= 1
        # ensure the height of the kernel is odd
        if kH % 2 == 0:
            kH -= 1

        # cv2.imshow("abcd", cv2.GaussianBlur(src=image, ksize=(kW, kH), sigmaX=0))
        # apply a Gaussian blur to the input image using our computed kernel size
        return cv2.GaussianBlur(src=image, ksize=(kW, kH), sigmaX=0)

    # more blocks mean more computation, but looks better
    @staticmethod
    def anonymize_face_pixelate(image, blocks=3):
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

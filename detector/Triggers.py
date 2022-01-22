from abc import ABCMeta, abstractmethod
from os.path import join, dirname
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from av.frame import Frame

from box_utils import predict
from util import cropFrameToBoxArea


class Trigger_Function(metaclass=ABCMeta):

    # @classmethod
    # def version(self): return "1.0"

    @abstractmethod
    def check(self, frame, probability, label, box=None) -> Tuple[Frame, np.ndarray]:
        raise NotImplementedError


class Age_Trigger(Trigger_Function):
    age_classifier_onnx = join(dirname(__file__), "models/age_googlenet.onnx")

    age_classifier = ort.InferenceSession(age_classifier_onnx, providers=['CUDAExecutionProvider'])
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def check(self, frame, probability, label, box=None):
        image = frame if box is None else cropFrameToBoxArea(frame, box)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image_mean = np.array([104, 117, 123])
        image = image - image_mean
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.age_classifier.get_inputs()[0].name
        ages = self.age_classifier.run(None, {input_name: image})
        age = self.ageList[ages[0].argmax()]
        return frame, age


class Face_Trigger(Trigger_Function):
    face_detector_onnx = join(dirname(__file__), "models/version-RFB-320.onnx")
    face_detector = ort.InferenceSession(face_detector_onnx, providers=['CUDAExecutionProvider'])

    def check(self, frame, probability, label, box=None) -> Tuple[Frame, np.ndarray]:
        _image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _image = cv2.resize(_image, (320, 240))
        image_mean = np.array([127, 127, 127])
        _image = (_image - image_mean) / 128
        _image = np.transpose(_image, [2, 0, 1])
        _image = np.expand_dims(_image, axis=0)
        _image = _image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: _image})
        #TODO: Do I only return the ones with prob > x here?
        boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, boxes, probability)
        return frame, boxes

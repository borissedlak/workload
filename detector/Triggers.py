from abc import ABCMeta, abstractmethod
from os.path import join, dirname
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from av.frame import Frame

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
    face_detector_onnx = join(dirname(__file__), "../models/version-RFB-320.onnx")
    face_detector = ort.InferenceSession(face_detector_onnx, providers=['CUDAExecutionProvider'])

    def check(self, frame, probability, label, box=None) -> Tuple[Frame, np.ndarray]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        return boxes, labels, probs

        color = (255, 128, 0)

        # orig_image = cv2.imread(img_path)
        # orig_image = imutils.resize(orig_image, width=700)
        boxes, labels, probs = faceDetector(orig_image)

        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            # cropped = cropFrameToBoxArea(orig_image, box)
            # gender = genderClassifier(cropped)
            frame, age = age_trigger.check(orig_image, 0.7, '(38-43)', box=box)
            # age = ageClassifier(cropped)
            gender = "???"
            # age = "???"
            # print(f'Box {i} --> {gender}, {age}')

            orig_image = blur_pixelate.transform(orig_image, options={'box': box, 'blocks': 5})
            # orig_image[box[1]:box[3], box[0]:box[2]] = face_blurred

            # cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)
            cv2.putText(orig_image, f'{gender}, {age}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 2,
                        cv2.LINE_AA)
            # cv2.imshow('', orig_image)
        return orig_image

from abc import ABCMeta, abstractmethod
from datetime import datetime
from os.path import join, dirname
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from av.frame import Frame

from box_utils import predict
from util import cropFrameToBoxArea, printExecutionTime


class Trigger_Function(metaclass=ABCMeta):

    # @classmethod
    # def version(self): return "1.0"

    @abstractmethod
    def check(self, frame, options=None) -> Tuple[Frame, np.ndarray]:
        raise NotImplementedError


# ##################################################################################################################
# ########  Trigger Functions  #####################################################################################
# ##################################################################################################################

class Face_Trigger(Trigger_Function):
    function_name = 'Face_Trigger'
    face_detector_onnx = join(dirname(__file__), "models/version-RFB-320.onnx")
    face_detector = ort.InferenceSession(face_detector_onnx, providers=['CUDAExecutionProvider'])

    def check(self, frame, options=None) -> Tuple[Frame, np.ndarray]:
        self.start_time = None
        if 'stats' in options and options['stats']:
            self.start_time = datetime.now()

        _image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _image = cv2.resize(_image, (320, 240))
        image_mean = np.array([127, 127, 127])
        _image = (_image - image_mean) / 128
        _image = np.transpose(_image, [2, 0, 1])
        _image = np.expand_dims(_image, axis=0)
        _image = _image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: _image})
        boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, boxes, options['prob'])

        printExecutionTime(self.function_name, datetime.now(), self.start_time)
        return frame, boxes


class Age_Trigger(Trigger_Function):
    function_name = 'Age_Trigger'
    age_classifier_onnx = join(dirname(__file__), "models/age_googlenet.onnx")
    age_classifier = ort.InferenceSession(age_classifier_onnx, providers=['CUDAExecutionProvider'])
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def check(self, frame, options=None):
        boxes_hit = []
        self.start_time = None
        if 'stats' in options and options['stats']:
            self.start_time = datetime.now()

        for i in range(options['boxes'].shape[0]):
            box = options['boxes'][i]
            _image = cropFrameToBoxArea(frame, box)

            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            _image = cv2.resize(_image, (224, 224))
            image_mean = np.array([104, 117, 123])
            _image = _image - image_mean
            _image = np.transpose(_image, [2, 0, 1])
            _image = np.expand_dims(_image, axis=0)
            _image = _image.astype(np.float32)

            input_name = self.age_classifier.get_inputs()[0].name
            ages = self.age_classifier.run(None, {input_name: _image})
            age = self.ageList[ages[0].argmax()]
            if age == options['label'] and ages[0].argmax() >= options['prob']:
                boxes_hit.append(box)

            if 'debug' in options and options['debug']:
                cv2.putText(frame, f'{age}', (box[0] - 50, box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            1.25, (0, 0, 0), 2, cv2.LINE_AA)

        printExecutionTime(self.function_name, datetime.now(), self.start_time)
        return frame, np.array(boxes_hit)

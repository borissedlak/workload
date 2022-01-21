# SPDX-License-Identifier: Apache-2.0

import sys
from os.path import join, dirname

import cv2
import imutils
import numpy as np
import onnxruntime as ort

import Detector
from Transformations import anonymize_face_pixelate
from util import cropFrameToBoxArea, Age_Trigger

sys.path.append('..')
from box_utils import predict

ag = Age_Trigger()

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-640 onnx model
face_detector_onnx = join(dirname(__file__), "version-RFB-320.onnx")
plate_detector_onnx = join(dirname(__file__), "az_plate_ssdmobilenetv1.onnx")

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
face_detector = ort.InferenceSession(face_detector_onnx, providers=['CUDAExecutionProvider'])


# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes




# face detection method
def faceDetector(orig_image, threshold=0.85):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (300, 300))
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face gender classification using GoogleNet onnx model
gender_classifier_onnx = join(dirname(__file__), "gender_googlenet.onnx")

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
gender_classifier = ort.InferenceSession(gender_classifier_onnx, providers=['CUDAExecutionProvider'])
genderList = ['Male', 'Female']


# gender classification method
def genderClassifier(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = gender_classifier.get_inputs()[0].name
    genders = gender_classifier.run(None, {input_name: image})
    gender = genderList[genders[0].argmax()]
    return gender


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face age classification using GoogleNet onnx model
age_classifier_onnx = join(dirname(__file__), "age_googlenet.onnx")

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
age_classifier = ort.InferenceSession(age_classifier_onnx, providers=['CUDAExecutionProvider'])
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


# age classification method
def ageClassifier(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    ages = age_classifier.run(None, {input_name: image})
    age = ageList[ages[0].argmax()]
    return age


# ------------------------------------------------------------------------------------------------------------------------------------------------

def process_frame_v2(orig_image):
    # img_path = "../../producer/demo_files/boris.jpg"
    color = (255, 128, 0)

    # orig_image = cv2.imread(img_path)
    # orig_image = imutils.resize(orig_image, width=700)
    boxes, labels, probs = faceDetector(orig_image)

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        cropped = cropFrameToBoxArea(orig_image, box)
        # gender = genderClassifier(cropped)
        # TODO: Returns in the new format here
        # age = ag.check(orig_image, 0.7, '(38-43)', box=box)
        # age = ageClassifier(cropped)
        gender = "???"
        age = "???"
        # print(f'Box {i} --> {gender}, {age}')

        face_blurred = anonymize_face_pixelate(cropped, blocks=5)
        orig_image[box[1]:box[3], box[0]:box[2]] = face_blurred

        # cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)
        # cv2.putText(orig_image, f'{gender}, {age}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 2,
        #             cv2.LINE_AA)
        # cv2.imshow('', orig_image)
    return orig_image

    # cv2.waitKey(0)
    # sys.exit()
# ------------------------------------------------------------------------------------------------------------------------------------------------

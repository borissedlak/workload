from abc import ABCMeta, abstractmethod
from datetime import datetime

import cv2
import imutils
import numpy as np
from av.frame import Frame

from util import cropFrameToBoxArea, printExecutionTime


class Transformation_Function(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, image, options=None) -> Frame:
        raise NotImplementedError


# ##################################################################################################################
# ########  Transformation Methods  ################################################################################
# ##################################################################################################################

class Blur_Area_Pixelate(Transformation_Function):
    start_time = None
    function_name = 'Blur_Area_Pixelate'

    # Always requires a box as input, even if everything is to blur it must specify the whole area
    def transform(self, image, options=None) -> Frame:

        # Return the image if there is no more boxes to blur
        if options['boxes'].size == 0:
            return image
        else:
            box = options['boxes'][0]

        box_area = cropFrameToBoxArea(image, box)
        (h, w) = box_area.shape[:2]
        xSteps = np.linspace(0, w, options['blocks'] + 1, dtype="int")
        ySteps = np.linspace(0, h, options['blocks'] + 1, dtype="int")
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
                roi = box_area[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(box_area, (startX, startY), (endX, endY),
                              (B, G, R), -1)
        composed = image
        composed[box[1]:box[3], box[0]:box[2]] = box_area
        # Recursively call again without the current box
        return self.transform(composed, options={'boxes': options['boxes'][1:], 'blocks': options['blocks']})

#TODO: Make
class Blur_Area_Simple(Transformation_Function):
    start_time = None
    function_name = 'Blur_Area_Simple'

    # Always requires a box as input, even if everything is to blur it must specify the whole area
    def transform(self, image, options=None) -> Frame:

        # Return the image if there is no more boxes to blur
        if options['boxes'].size == 0:
            return image
        else:
            box = options['boxes'][0]

        box_area = cropFrameToBoxArea(image, box)
        (h, w) = box_area.shape[:2]
        xSteps = np.linspace(0, w, options['blocks'] + 1, dtype="int")
        ySteps = np.linspace(0, h, options['blocks'] + 1, dtype="int")
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
                roi = box_area[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(box_area, (startX, startY), (endX, endY),
                              (B, G, R), -1)
        composed = image
        composed[box[1]:box[3], box[0]:box[2]] = box_area
        # Recursively call again without the current box
        return self.transform(composed, options={'boxes': options['boxes'][1:], 'blocks': options['blocks']})


#TODO: Make
class Fill_Area_Box(Transformation_Function):
    start_time = None
    function_name = 'Fill_Area_Box'

    # Always requires a box as input, even if everything is to blur it must specify the whole area
    def transform(self, image, options=None) -> Frame:

        # Return the image if there is no more boxes to blur
        if options['boxes'].size == 0:
            return image
        else:
            box = options['boxes'][0]

        box_area = cropFrameToBoxArea(image, box)
        (h, w) = box_area.shape[:2]
        xSteps = np.linspace(0, w, options['blocks'] + 1, dtype="int")
        ySteps = np.linspace(0, h, options['blocks'] + 1, dtype="int")
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
                roi = box_area[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(box_area, (startX, startY), (endX, endY),
                              (B, G, R), -1)
        composed = image
        composed[box[1]:box[3], box[0]:box[2]] = box_area
        # Recursively call again without the current box
        return self.transform(composed, options={'boxes': options['boxes'][1:], 'blocks': options['blocks']})

class Max_Spec_Resize(Transformation_Function):
    start_time = None
    function_name = 'Max_Spec_Resize'

    # Always requires a box as input, even if everything is to blur it must specify the whole area
    def transform(self, image, options=None) -> Frame:

        resized = image
        if 'max_width' in options and resized.shape[1] > options['max_width']:
            resized = imutils.resize(resized, width=options['max_width'])
        elif 'max_height' in options and resized.shape[0] > options['max_height']:
            resized = imutils.resize(resized, width=options['max_height'])

        return resized

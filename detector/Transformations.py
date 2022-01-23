from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from av.frame import Frame

from util import cropFrameToBoxArea


class Transformation_Function(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, image, options=None) -> Frame:
        raise NotImplementedError


# ##################################################################################################################
# ########  Transformation Methods  ################################################################################
# ##################################################################################################################

class Blur_Face_Pixelate(Transformation_Function):

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

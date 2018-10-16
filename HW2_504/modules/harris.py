#!/usr/bin/env python
'''
A module for determining the locations of corners in an image using
the Harris Corner Detection algorithm.

Info:
    type: eta.core.types.Module
    version: 0.1.0
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np

from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import matplotlib.pyplot as plt


class HarrisCornerConfig(etam.BaseModuleConfig):
    '''Harris corner configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(HarrisCornerConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)
        self._validate()

    def _validate(self):
        '''Ensures the right inputs are given to the module before
        actual computation begins.
        '''
        for data in self.data:
            # if data.visualized_corners and not data.input_image:
            if not data.input_image:
                raise ConfigError(
                    "ERROR! Original image required for visualization.")


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): [None] The input image
        sobel_horizontal_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).
    Outputs:
        corner_locations (eta.core.types.NpzFile): [None] The location of every
            corner detected by the Harris Corner algorithm, sorted by
            confidence
        corners_img_before_sup (eta.core.types.ImageFile): [None] The corner
            response image before suppression
        corners_img_after_sup (eta.core.types.ImageFile): [None] The corner
            response image after suppression
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(
            d, "input_image", default=None)
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.corner_locations = self.parse_string(
            d, "corner_locations", default=None)
        self.corners_img_before_sup = self.parse_string(
            d, "corners_img_before_sup", default=None)
        self.corners_img_after_sup = self.parse_string(
            d, "corners_img_after_Sup", default=None)


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        window_size (eta.core.types.Number): The size of the window
        threshold (eta.core.types.Number): The lower-bound threshold for
            corner "confidence".
    '''

    def __init__(self, d):
        self.window_size = self.parse_number(d, "window_size")
        self.threshold = self.parse_number(d, "threshold")


def _get_harris_corner(Gx, Gy, window_size, threshold):
    '''Performs Harris corner detection and outputs an Nx2 matrix of corner
    locations, where each row of the matrix is the location of a corner,
    represented as a 1x2 array [x,y].

    Args:
        Gx: The x-derivative of the image, which is the output of convolving
            with the horizontal sobel kernel
        Gy: the y-derivative of the image, which is the output of convolving
            with the vertical sobel kernel
        window_size: the size of the window to consider when computing
            the structure tensor
        threshold: the lower-bound threshold to use when choosing corners

    Returns:
        corner_location_matrix: the locations of each detected corner in the
            image, stored in an Nx2 matrix
        corner_response: image map with value of harris corner detection score
    '''
    m, n = Gx.shape
    edge = int((window_size - 1) / 2)
    corner_response = np.zeros((m - 2 * edge, n - 2 * edge))

    corner_location_matrix = []
    for i in range(edge, m - edge):
        for j in range(edge, n - edge):
            x_range = Gx[i - edge:i + edge + 1, j - edge:j + edge + 1]
            x_range = np.reshape(x_range, (1, window_size**2))
            y_range = Gy[i - edge:i + edge + 1, j - edge:j + edge + 1]
            y_range = np.reshape(y_range, (1, window_size**2))
            tmp = np.concatenate((x_range, y_range), axis=0)
            tensor = np.dot(tmp, tmp.T)
            w, v = np.linalg.eig(tensor)
            min_eig = w.min()
            corner_response[i - edge, j - edge] = min_eig
            if min_eig > threshold:
                corner_location_matrix.append(np.array([i - edge, j - edge]))
    print(corner_location_matrix)
    plt.imshow(corner_response)
    plt.show()


def non_max_suppression(corner_response, radius):
    """Finds corners in a given response map.

    This method uses a circular region to define the non-maxima suppression
    area. For example, let c1 be a corner representing a peak in the Harris
    response map, any corners in the area determined by the circle of radius
    'radius' centered in c1 should not be returned in the peaks array.
    Make sure you account for duplicate and overlapping points.

    Args:
        corner_response (numpy.array): floating-point response map,
            e.g. output from the Harris detector.
        radius (int): radius of circular region for non-maximal suppression.

    Returns:
        corners_sup: peaks found in response map R, each row must be defined
            as [x, y]. Array size must be N x 2, where N are the number of
            points found.
    """
    #(r_map, threshold, radius) = (r_maps["trans_a"], threshold["trans_a"], radius["trans_a"])

    #    radius  = 5
    #    threshold = .5
    r_map1 = np.copy(corner_response)

    #    print len (np.nonzero(r_map1)[0])
    data_max = np.zeros(r_map1.shape)
    ind = np.nonzero(r_map1)
    for n in range(len(ind[0])):
        i = ind[0][n]
        j = ind[1][n]
#        print 'i,j=',(i,j), 'cond 1', ( r_map1 [i,j]< np.max(frame) ), 'cond 2', np.max(frame_max)>0
        frame = r_map1[max((i - radius), 0):min((i + radius), r_map1.shape[0]), max((j - radius), 0):min((j + radius), r_map1.shape[1])]
        if r_map1[i, j] < np.max(frame):  # not local max
            data_max[i, j] = 0
        elif np.max(data_max[max((i - radius), 0):min((i + radius), data_max.shape[0]), max((j - radius), 0):min((j + radius), data_max.shape[1])]) > 0:
            # Tie, and already as a max t
            data_max[i, j] = 0
        else:
            data_max[i, j] = r_map1[i, j]

    col_ind, row_ind = np.nonzero(data_max)
#    You can use the distance as a conditional measure to merge the points.
#   Average of the x and y coordinates of the close points to merge.
    corners = []
    for i in range(len(row_ind)):
        corners.append((row_ind[i], col_ind[i]))
    corners = tuple(corners)

    return np.array(corners)


def _visualize_corners(in_img, corner_locations):
    '''Creates an image that shows the detected corners.

    Args:
        in_img: the original image
        corner_locations: the locations of each detected corner in the
            image, stored in an Nx2 matrix

    Returns:
        out_img: an image with the detected corners colored in red

    '''
    out_img = in_img.copy()
    for i in corner_locations.shape[0]:
        cv2.circle(img=out_img, center=corner_locations[i], radius=2, colour=[0, 0, 255])
    return out_img


def _find_corners(harris_corner_config):
    for data in harris_corner_config.data:
        sobel_horiz = np.load(data.sobel_horizontal_result)["filtered_matrix"]
        sobel_vert = np.load(data.sobel_vertical_result)["filtered_matrix"]
        corner_locations, corner_confidence = _get_harris_corner(
            sobel_horiz, sobel_vert,
            harris_corner_config.parameters.window_size,
            harris_corner_config.parameters.threshold)
        if data.corner_locations:
            np.savez(data.corner_locations, corner_locations=corner_locations)
        if data.visualized_corners:
            in_img = etai.read(data.input_image)
            corners_before_sup = _visualize_corners(in_img, corner_locations)
            etai.write(corners_visualized, data.visualized_corners)
            corners_after_sup = non_max_suppression(corner_confidence)
            corners_visualized = _visualize_corners(in_img, corners_sup)
            etai.write(corners_visualized, data.visualized_corners)


def run(config_path, pipeline_config_path=None):
    '''Run the Harris Corner module.

    Args:
        config_path: path to a HarrisCornerConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    harris_corner_config = HarrisCornerConfig.from_json(config_path)
    etam.setup(harris_corner_config, pipeline_config_path=pipeline_config_path)
    _find_corners(harris_corner_config)


if __name__ == "__main__":
    run(*sys.argv[1:])

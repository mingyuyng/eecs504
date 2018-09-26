#!/usr/bin/env python
'''
A module for determining the edges of an image using the Canny Edge Detector.

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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam


# CONSTANTS
STRONG = 10
WEAK = 5
SUPPRESSED = 0


class CannyEdgeConfig(etam.BaseModuleConfig):
    '''CannyEdge configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(CannyEdgeConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): the input image
        sobel_horizontal_result (eta.core.types.Image): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.Image): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).

    Outputs:
        image_edges (eta.core.types.ImageFile): A new image displaying
            the edges of the original image.
        gradient_orientation (eta.core.types.ImageFile): [None] An image
            displaying the gradient orientation for each pixel in the
            input image
        gradient_intensity (eta.core.types.ImageFile): [None] An image
            displaying the gradient intensity for each pixel in the input
            image
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(d, "input_image")
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.image_edges = self.parse_string(d, "image_edges")
        self.gradient_orientation = self.parse_string(
            d, "gradient_orientation", default=None)
        self.gradient_intensity = self.parse_string(
            d, "gradient_intensity", default=None)


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        high_threshold (eta.core.types.Number): The upper threshold to use
            during double thresholding
        low_threshold (eta.core.types.Number): The lower threshold to use
            during double thresholding
    '''

    def __init__(self, d):
        self.high_threshold = self.parse_number(d, "high_threshold")
        self.low_threshold = self.parse_number(d, "low_threshold")


def _create_intensity_orientation_matrices(Gx, Gy):
    '''Creates two matrices: one for intensity and one for orientation.
    The intensity at each pixel is defined as sqrt(Gx^2 + Gy^2) and
    the orientation of each pixel is defined as arctan(Gy/Gx).

    Args:
        Gx: the result of convolving with the "sobel_horizontal" kernel
        Gy: the result of convolving with the "sobel_vertical" kernel

    Returns:
        (g_intensity, orientation): a tuple with the first element as
            the intensity matrix and the second element as the
            orientation matrix.
    '''
    @TODO
    # ADD CODE HERE


def _non_maximum_suppression(g_intensity, orientation, input_image):
    '''Performs non-maximum suppression. If a pixel is not a local maximum
    (not bigger than it's neighbors with the same orientation), then
    suppress that pixel.

    Args:
        g_intensity: the gradient intensity of each pixel
        orientation: the gradient orientation of each pixel
        input_image: the input image

    Returns:
        g_sup: the gradient intensity of each pixel, with some intensities
            suppressed to 0 if the corresponding pixel was not a local
            maximum
    '''
    @TODO
    # ADD CODE HERE


def _double_thresholding(g_suppressed, low_threshold, high_threshold):
    '''Performs a double threhold. All pixels with gradient intensity larger
    than 'high_threshold' are considered strong edges, and all pixels
    with gradient intensity smaller than 'low_threshold' are suppressed
    to 0.

    Args:
        g_suppressed: the gradient intensities of all pixels, after
            non-maxiumum suppression
        low_threshold: the lower threshold in double thresholding
        high_threshold: the higher threshold in double thresholding

    Returns:
        g_thresholded: the result of double thresholding
    '''
    @TODO
    # ADD CODE HERE


def _hysteresis(g_thresholded):
    '''Performs hysteresis. If a weak pixel is connected to a strong pixel,
    then the weak pixel is marked as strong. Otherwise, it is suppressed.
    The result will be an image with only strong pixels.

    Args:
        g_thresholded: the result of double thresholding

    Returns:
        g_strong: an image with only strong edges
    '''
    @TODO
    # ADD CODE HERE


def _perform_canny_edge_detection(canny_edge_config):
    for data in canny_edge_config.data:
        in_img = etai.read(data.input_image)
        sobel_horiz = etai.read(data.sobel_horizontal_result)
        sobel_vert = etai.read(data.sobel_vertical_result)
        (g_intensity, orientation) = _create_intensity_orientation_matrices(
                                        sobel_horiz,
                                        sobel_vert)
        if data.gradient_intensity is not None:
            etai.write(g_intensity, data.gradient_intensity)
        if data.gradient_orientation is not None:
            etai.write(orientation, data.gradient_orientation)
        g_suppressed = _non_maximum_suppression(g_intensity, orientation,
                                                in_img)
        g_thresholded = _double_thresholding(
                            g_suppressed,
                            canny_edge_config.parameters.low_threshold,
                            canny_edge_config.parameters.high_threshold)
        g_strong = _hysteresis(g_thresholded)
        g_strong = g_strong.astype(int)

        etai.write(g_strong, data.image_edges)


def run(config_path, pipeline_config_path=None):
    '''Run the canny edge detector module.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    canny_edge_config = CannyEdgeConfig.from_json(config_path)
    etam.setup(canny_edge_config, pipeline_config_path=pipeline_config_path)
    _perform_canny_edge_detection(canny_edge_config)


if __name__ == "__main__":
    run(*sys.argv[1:])

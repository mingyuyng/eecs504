#!/usr/bin/env python
'''
A module for finding the line segments of an image, given the output
of Canny Edge Detection.

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

from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam


class FindSegmentsConfig(etam.BaseModuleConfig):
    '''Find line segments module configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(FindSegmentsConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): The input image
        canny_edge_output (eta.core.types.Image): The output of canny
            edge detection
        gradient_intensity (eta.core.types.Image): The gradient intensity
            for each pixel in the input image
        gradient_orientation (eta.core.types.Image): The gradient orientation
            for each pixel in the input image

    Outputs:
        line_segments (eta.core.types.JSONFile): A list of coordinate tuples,
            specifying the start and end of each line segment in the image
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(d, "input_image")
        self.canny_edge_output = self.parse_string(d, "canny_edge_output")
        self.gradient_intensity = self.parse_string(d, "gradient_intensity")
        self.gradient_orientation = self.parse_string(
            d, "gradient_orientation")
        self.line_segments = self.parse_string(d, "line_segments")


@TODO
'''ADD CODE HERE AND SPECIFY ANY PARAMETERS ABOVE. There can be many
solutions to this problem, so we are leaving it open-ended.
'''


def _find_line_segments(find_segments_config):
    @TODO
    '''ADD CODE HERE'''


def run(config_path, pipeline_config_path=None):
    '''Run this module to find line segments in the image.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    find_segments_config = FindSegmentsConfig.from_json(config_path)
    etam.setup(find_segments_config, pipeline_config_path=pipeline_config_path)
    _find_line_segments(find_segments_config)


if __name__ == "__main__":
    run(*sys.argv[1:])

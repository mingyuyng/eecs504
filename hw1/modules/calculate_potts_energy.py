#!/usr/bin/env python
'''
A module for calculating the Pott's Energy of an image.

This module calculates the Pott's Energy of an image, after convolving
with the x-derivative and y-derivative kernels.

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
import matplotlib.pyplot as plt
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam
import eta.core.serial as etas


class PottsEnergyConfig(etam.BaseModuleConfig):
    '''Pott's Energy module configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(PottsEnergyConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        x_derivative_path (eta.core.types.Image): The result of convolving
            the original image with the "x_derivative" kernel
        y_derivative_path (eta.core.types.Image): The result of convolving
            the original image with the "y_derivative" kernel

    Outputs:
        potts_energy_out (eta.core.types.JSONFile): The Pott's Energy of the
            image, written to a JSON file
    '''

    def __init__(self, d):
        self.x_derivative_path = self.parse_string(d, "x_derivative_path")
        self.y_derivative_path = self.parse_string(d, "y_derivative_path")
        self.potts_energy_out = self.parse_string(d, "potts_energy_out")


def _calculate_potts_energy(data):
    '''Calculates the pott's energy of an image, given the x-derivative
    and y-derivative of the image at each pixel. Beta is assumed to be 1.

    Args:
        data: the data configuration values

    Returns:
        potts_energy: the Pott's Energy for the original image
    '''
  
    # ADD CODE HERE
    x_dev = etai.read(data.x_derivative_path)
    y_dev = etai.read(data.y_derivative_path)
    plt.imshow(x_dev)
    plt.show()
    if x_dev.ndim > 2: 
        x_dev_eff = x_dev[1:-1,:,:]
        y_dev_eff = y_dev[:,1:-1,:]
    else:
        x_dev_eff = x_dev[1:-1,:]
        y_dev_eff = y_dev[:,1:-1]

    x_dev_eff[x_dev_eff!=0] = 1
    y_dev_eff[y_dev_eff!=0] = 1

    return np.sum(x_dev_eff) + np.sum(y_dev_eff)


    


def run(config_path, pipeline_config_path=None):
    '''Run the Pott's Energy module.

    Args:
        config_path: path to a ConvolutionConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    potts_config = PottsEnergyConfig.from_json(config_path)
    etam.setup(potts_config, pipeline_config_path=pipeline_config_path)
    for data in potts_config.data:
        energy = _calculate_potts_energy(data)
        potts_energy = defaultdict(lambda: defaultdict())
        potts_energy["result"]["energy"] = energy
        etas.write_json(potts_energy, data.potts_energy_out)


if __name__ == "__main__":
    run(*sys.argv[1:])

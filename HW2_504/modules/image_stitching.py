#!/usr/bin/env python
'''
A module for stitching two parts of an image into one whole image.

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
import cv2

from eta.core.config import Config
import eta.core.module as etam


class ImageStitchingConfig(etam.BaseModuleConfig):
    '''Image stitching configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ImageStitchingConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        corners_image_1 (eta.core.types.NpzFile): An Nx2 matrix
            containing (x,y) locations of all corners in image 1,
            detected by the Harris Corner algorithm
        corners_image_2 (eta.core.types.NpzFile): A Mx2 matrix
            containing (x,y) locations of all corners in image 2,
            detected by the Harris Corner algorithm
        image_1 (eta.core.types.Image): the first input image
        image_2 (eta.core.types.Image): the second input image
    Outputs:
        stitched_image (eta.core.types.ImageFile): The final stitched image
    '''

    def __init__(self, d):
        self.corners_image_1 = self.parse_string(d, "corners_image_1")
        self.corners_image_2 = self.parse_string(d, "corners_image_2")
        self.stitched_image = self.parse_string(d, "stitched_image")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        no_correspondence (eta.core.types.Number): the number of
            points to use when computing the homography
    '''

    def __init__(self, d):
        self.no_correspondence = self.parse_string(d, "no_correspondence")


def _get_HOG_descriptors(corner_locs, in_img):
    '''Return a MxN matrix that contains the M-dimensional HOG feature vectors
        for all N corners.

    Args:
        corner_locs: the location of Harris corners, given as a
           Nx2 2-dimensional matrix
        in_img: the input image

    Returns:
        hog_features: a MxN matrix containing HOG feature vectors for every
            detected corner
    '''
    hog = cv2.HOGDescriptor()
    winStride = (8,8)
    padding = (8,8)
    new_locations = []
    for i in range(corner_locs.shape[0]):
        new_locations.append(tuple(corner_locs[i]))
    new_locations = tuple(new_locations)
    N = corner_locs.shape[0]
    hog_descrp = hog.compute(in_img, winStride, padding, new_locations)
    hog_features = np.reshape(hog_descrp,(N,3780))
    return hog_features


def _match_keypoints(hog_features_1, hog_features_2, img1_corners, img2_corners ):
    '''Match the HOG features of the two images and return a list of matched
    keypoints.

    Args:
        hog_features_1: the HOG features for the first image
        hog_features_2: the HOG features for the second image
        img1_corners: the corners detected in the first image, from which the
            HOG features were computed
        img2_corners: the corners detected in the second image, from which the
            HOG feautures were computed

    Returns:
       img1_matched_points: a list of corner locations in the first image that
            match with those in the second image
       img2_matched_points: a list of corner locations in the second image that
            match with those in the first image
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(hog_features_1,hog_features_2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    img1_matched_pts = []
    img2_matched_pts = []
    for match in matches:
        img1_matched_pts.append(img1_corners[match.queryIdx])
        img2_matched_pts.append(img2_corners[match.trainIdx])
    # Draw first 10 matches.
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
    return img1_matched_pts, img2_matched_pts


def _get_homography(img1_keypoints, img2_keypoints):
    '''Calculate the homography matrix that relates the first image with the
    second image, using the matched keypoints.

    Args:
        img1_keypoints: the matched keypoints in image 1
        img2_keypoints: the matched keypoints in image 2

    Returns:
        homog_matrix: the homography matrix that relates image 1 and image 2
    '''
    # @TODO


def _overlap(img1, img2, homog_matrix):
    '''Applies a homography transformation to img2 to stitch img1 and img2
    togther.

    Args:
        img1: the first image
        img2: the second image
        homog_matrix: the homography matrix

    Returns:
        stitched_image: the final stitched image
    '''
    # @TODO


def _stitch_images(image_stitching_config):
    for data in image_stitching_config.data:
        img1_corners = np.load(data.corners_image_1)["corner_locations"]
        img2_corners = np.load(data.corners_image_2)["corner_locations"]
        img1 = etai.read(data.image_1)
        img2 = etai.read(data.image_2)
        hog_features_1 = _get_HOG_descriptors(img1_corners, img1)
        hog_features_2 = _get_HOG_descriptors(img2_corners, img2)
        img_1_pts, img_2_pts = _match_keypoints(hog_features_1, hog_features_2,
                                    img1_corners, img2_corners)
        # Tune this parameter to specify the number of corresponding points
        no_correspondence = image_stitching_config.parameters.no_correspondence
        H = _get_homography(img_1_pts[:no_correspondence], img_2_pts[:no_correspondence])
        final_img = _overlap(img1, img2, H)
        etai.write(final_img, data.stitched_image)


def run(config_path, pipeline_config_path=None):
    '''Run the Image Stitching module.

    Args:
        config_path: path to a ImageStitchingConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    image_stitching_config= ImageStitchingConfig.from_json(config_path)
    etam.setup(image_stitching_config,
               pipeline_config_path=pipeline_config_path)
    _stitch_images(image_stitching_config)


if __name__ == "__main__":
    run(*sys.argv[1:])

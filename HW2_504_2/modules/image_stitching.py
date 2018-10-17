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
import eta.core.image as etai
import eta.core.module as etam
import matplotlib.pyplot as plt


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
        corner_locs_1 (eta.core.types.NpzFile): An Nx2 matrix
            containing (x,y) locations of all corners in image 1,
            detected by the Harris Corner algorithm
        corner_locs_2 (eta.core.types.NpzFile): A Mx2 matrix
            containing (x,y) locations of all corners in image 2,
            detected by the Harris Corner algorithm
        image_1 (eta.core.types.Image): the first input image
        image_2 (eta.core.types.Image): the second input image
    Outputs:
        stitched_image (eta.core.types.ImageFile): The final stitched image
    '''

    def __init__(self, d):
        self.corner_locs_1 = self.parse_string(d, "corner_locs_1")
        self.corner_locs_2 = self.parse_string(d, "corner_locs_2")
        self.image_1 = self.parse_string(d, "image_1")
        self.image_2 = self.parse_string(d, "image_2")
        self.stitched_image = self.parse_string(d, "stitched_image")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        no_correspondence (eta.core.types.Number): [4] the number of
            points to use when computing the homography
    '''

    def __init__(self, d):
        self.no_correspondence = self.parse_number(
            d, "no_correspondence", default=4)


def _get_HOG_descriptors(corner_locs, in_img):
    '''Return a MxN matrix that contains the M-dimensional HOG feature vectors
        for all N corners.

    Args:
        corner_locs: the location of Harris corners, given as a
            Nx2 2-dimensional matrix
        in_img: the input image

    Returns:
        hog_features: a N x 3780 matrix containing HOG feature vectors for every
            detected corner
    '''
    # Setting parameters
    win_size = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    # Initializing descriptor
    hog = cv2.HOGDescriptor(win_size, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # Setting compute parameters
    win_stride = (8, 8)
    padding = (8, 8)
    new_locations = []

    # Gathering all corner locations
    # NOTE: This will not work until you successfully implement the Harris
    #       Corner Detector in 'modules/harris.py'.
    for i in range(corner_locs.shape[0]):
        new_locations.append((int(corner_locs[i][0]), int(corner_locs[i][1])))
    N = len(new_locations)

    # Computing HOG feature vectors for all corners and concatenating them
    # together
    hog_descrp = hog.compute(in_img, win_stride, padding, new_locations)
    feat_size = int((((win_size[0] / 8) - 1) * ((win_size[1] / 8) - 1)) * 36)
    hog_features = np.asarray(hog_descrp)

    # Reshaping as a N x 3780 array
    hog_features = np.reshape(hog_descrp, (N, feat_size))

    return hog_features


def _match_keypoints(hog_features_1, hog_features_2, img1_corners, img2_corners, img1, img2):
    '''Match the HOG features of the two images and return a list of matched
    keypoints.

    Args:
        hog_features_1: the HOG features for the first image
        hog_features_2: the HOG features for the second image
        img1_corners: the corners detected in the first image, from which the
            HOG features were computed
        img2_corners: the corners detected in the second image, from which the
            HOG feautures were computed
        img1: the first image, in case you want to visualize the matches.
        img2: the second image, in case you want to visualize the matches

    Returns:
       img1_matched_points: a list of corner locations in the first image that
            match with those in the second image
       img2_matched_points: a list of corner locations in the second image that
            match with those in the first image
    '''
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(hog_features_1, hog_features_2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    img1_matched_pts = []
    img2_matched_pts = []
    for match in matches:
        img1_matched_pts.append(img1_corners[match.queryIdx])
        img2_matched_pts.append(img2_corners[match.trainIdx])

    # Draw the first 20 matches(the blue dots are the matches)
    out_img = img1.copy()
    out_img_2 = img2.copy()
    for i in range(20):
        cv2.circle(out_img,
                   (img1_matched_pts[i][0], img1_matched_pts[i][1]),
                   4,
                   (0, 0, 255),
                   -1)
    for i in range(20):
        cv2.circle(out_img_2,
                   (img2_matched_pts[i][0], img2_matched_pts[i][1]),
                   4,
                   (0, 0, 255),
                   -1)
    # The images below will be stored in your current working directory
    etai.write(out_img, "out1.png")
    etai.write(out_img_2, "out2.png")

    return img1_matched_pts, img2_matched_pts


def _get_homography(img1_keypoints, img2_keypoints):
    '''Calculate the homography matrix that relates the first image with the
    second image, using the matched keypoints.

    Args:
        img1_keypoints: a list of matched keypoints in image 1. The number
            of keypoints is indicated by the parameter, 'no_correspondence'.
        img2_keypoints: a list of matched keypoints in image 2. The number
            of keypoints is indicated by the parameter, 'no_correspondence'.

    Returns:
        homog_matrix: the homography matrix that relates image 1 and image 2
    '''
    # TODO
    # REPLACE THE CODE BELOW WITH YOUR IMPLEMENTATION
    n = len(img1_keypoints)
    A = []
    y = []
    for i in range(n):
        tmp = np.kron(np.eye(2), np.append(img1_keypoints[i], 1))
        tmp = np.concatenate((tmp, -np.outer(img2_keypoints[i], img1_keypoints[i])), axis=1)
        if i == 0:
            A = tmp
            y = img2_keypoints[i]
        else:
            A = np.append(A, tmp, axis=0)
            y = np.append(y, img2_keypoints[i], axis=0)

    # Solve the least square problem and regenerate the matrix H
    x = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(y)
    x = np.append(x, 1)
    homog_matrix = np.reshape(x, (3, 3))

    for i in range(n):
        tmp = np.kron(np.eye(2), np.append(img2_keypoints[i], 1))
        tmp = np.concatenate((tmp, -np.outer(img1_keypoints[i], img2_keypoints[i])), axis=1)
        if i == 0:
            A = tmp
            y = img1_keypoints[i]
        else:
            A = np.append(A, tmp, axis=0)
            y = np.append(y, img1_keypoints[i], axis=0)

    # Solve the least square problem and regenerate the matrix H
    x = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(y)
    x = np.append(x, 1)
    homog_matrix_inv = np.reshape(x, (3, 3))

    return homog_matrix, homog_matrix_inv


def _overlap(img1, img2, homog_matrix, homog_matrix_inv):
    '''Applies a homography transformation to img2 to stitch img1 and img2
    togther.

    Args:
        img1: the first image
        img2: the second image
        homog_matrix: the homography matrix

    Returns:
        stitched_image: the final stitched image
    '''
    # TODO
    # REPLACE THE CODE BELOW WITH YOUR IMPLEMENTATION

    # Fugure out the size of whole image after stitching
    m1, n1, channel1 = img1.shape
    m2, n2, channel2 = img2.shape
    x = np.arange(0, n1)
    y = np.arange(0, m1)
    gx, gy = np.meshgrid(x, y)
    gx_1d = np.reshape(gx, (1, gx.size))
    gy_1d = np.reshape(gy, (1, gy.size))

    loc_1 = np.concatenate((gx_1d, gy_1d), axis=0)
    loc_11 = np.concatenate((loc_1, np.ones((1, gx_1d.size))), axis=0)
    loc_2 = homog_matrix.dot(loc_11)

    loc_2 = loc_2 / loc_2[-1]
    loc_2 = loc_2[0:2]

    bound_x_r, bound_y_r = np.ceil(np.max(loc_2, axis=1))
    bound_x_l, bound_y_l = np.floor(np.min(loc_2, axis=1))
    bound_x_l = int(np.min([bound_x_l, 0]))
    bound_y_l = int(np.min([bound_y_l, 0]))
    bound_x_r = int(np.max([bound_x_r, n2]))
    bound_y_r = int(np.max([bound_y_r, m2]))
    h = bound_y_r - bound_y_l
    w = bound_x_r - bound_x_l
    new_image = np.zeros((h, w, 3))
    H_inv = homog_matrix_inv

    # Backward Wrapping
    for i in range(h):
        for j in range(w):
            pos = np.array([j + bound_x_l, i + bound_y_l, 1])
            pos_old = H_inv.dot(pos)
            pos_old = pos_old / pos_old[-1]
            pos_old = pos_old[0:2]
            # Bilinear Interpolation
            if (pos_old[0] < 0) or (pos_old[0] >= n1 - 1) or (pos_old[1] < 0) or (pos_old[1] >= m1 - 1):
                new_image[i, j, :] = np.zeros(3)
            else:
                ix1, iy1 = np.floor(pos_old)
                ix1 = int(ix1)
                iy1 = int(iy1)
                ix2 = ix1 + 1
                iy2 = iy1 + 1
                tmp = img1[iy1, ix1, :] * (ix2 - pos_old[0]) * (iy2 - pos_old[1]) + img1[iy1, ix2, :] * (pos_old[0] - ix1) * (iy2 - pos_old[1])
                tmp = tmp + img1[iy2, ix1, :] * (ix2 - pos_old[0]) * (pos_old[1] - iy1) + img1[iy2, ix2, :] * (pos_old[0] - ix1) * (pos_old[1] - iy1)
                new_image[i, j, :] = tmp

            if (j + bound_x_l >= 0) and (j + bound_x_l < n2 - 1) and (i + bound_y_l >= 0) and (i + bound_y_l < m2 - 1):
                new_image[i, j, :] = (new_image[i, j, :] + img2[i + bound_y_l, j + bound_x_l, :]) / 2

    plt.imshow(new_image)
    plt.show()
    return new_image


def _stitch_images(image_stitching_config):
    for data in image_stitching_config.data:
        # Load the corner locations
        img1_corners = np.load(data.corner_locs_1)["corner_locations"]
        img2_corners = np.load(data.corner_locs_2)["corner_locations"]

        # Read in the input images
        img1 = etai.read(data.image_1)
        img2 = etai.read(data.image_2)

        # Compute HOG feature vectors for every detected corner
        hog_features_1 = _get_HOG_descriptors(img1_corners, img1)
        hog_features_2 = _get_HOG_descriptors(img2_corners, img2)

        # Match the feature vectors
        img_1_pts, img_2_pts = _match_keypoints(hog_features_1, hog_features_2,
                                                img1_corners, img2_corners, img1, img2)

        # Tune this parameter in "requests/image_stitching_request.json"
        # to specify the number of corresponding points to use when computing
        # the homography matrix
        no_correspondence = image_stitching_config.parameters.no_correspondence

        # Compute the homography matrix that relates image 1 and image 2
        H, H_inv = _get_homography(img_1_pts[1:no_correspondence + 1], img_2_pts[1:no_correspondence + 1])

        # Stitching the images by applying the homography matrix to image 2
        final_img = _overlap(img1, img2, H, H_inv)

        # Write the final stitched image
        etai.write(final_img, data.stitched_image)


def run(config_path, pipeline_config_path=None):
    '''Run the Image Stitching module.

    Args:
        config_path: path to a ImageStitchingConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    image_stitching_config = ImageStitchingConfig.from_json(config_path)
    etam.setup(image_stitching_config,
               pipeline_config_path=pipeline_config_path)
    _stitch_images(image_stitching_config)


if __name__ == "__main__":
    run(*sys.argv[1:])

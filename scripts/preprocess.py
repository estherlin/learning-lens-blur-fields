import sys
import json
import pickle

import cv2
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from util.preprocess import *
from util.printarray import *

def undistort_image(img, camera_matrix, distortion_coefficients):
    """ Undistort an image using the camera matrix and distortion coefficients"""
    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients)
    return undistorted_img


def config_parser():
    """ Parse the command line arguments"""
    # Create the parser
    parser = argparse.ArgumentParser(description='Preprocessing arguments')
    # Add the arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--camera-params-path', type=str, required=True, 
                            help='Path to the json file with camera parameters')
    parser.add_argument('--results-path', type=str, required=True,
                            help='Path to the directory to save the results')
    parser.add_argument('--data-path', type=str, required=True,
                            help='Path to the dataset of blurry images')
    parser.add_argument('--sharp-path', type=str, required=True,
                            help='Path to all of the sharp frequency patterns')
    parser.add_argument('--calibration-path', type=str, required=True,
                            help='Path to all of the calibration patterns')

    # Execute the parse_args() method
    return parser.parse_args()


if __name__ == '__main__':

    # grab params from the command line
    args = config_parser()
    camera_params_filename     = args.camera_params_path  # add path to config file
    results_dir     	       = args.results_path # add path to results directory
    data_dir     	           = args.data_path    # add path to results directory

    # Load in data and camera parameters
    with open(f'{data_dir}/lens_positions.pickle', 'rb') as handle:
        lens_positions = pickle.load(handle)
        lens_positions = np.array(lens_positions)
    with open(f'{data_dir}/patterns.pickle', 'rb') as handle:
        patterns = pickle.load(handle)

    blurry_np = np.load(f'{data_dir}/blurry-original.npy')

    # load in a json file
    with open(f'{camera_params_filename}') as f:
        cameraParams = json.load(f)

    # Get camera matrix and distortion coefficients
    camera_matrix = np.array(cameraParams['camera_matrix'])
    distortion_coefficients = np.array(cameraParams['distortion_coefficients'])
    if distortion_coefficients.size == 2:
        distortion_coefficients = np.pad(distortion_coefficients, ((0, 3)), mode='constant', constant_values=0)
    camera_matrix, distortion_coefficients


    # Undistort the blurry images
    # Perform undistortion
    undistorted = np.zeros(blurry_np.shape)
    up_dim = (2*blurry_np.shape[3], 2*blurry_np.shape[2])
    dim = (blurry_np.shape[3], blurry_np.shape[2])
    for pat in range(blurry_np.shape[0]):
        for lens_pos in range(blurry_np.shape[1]):
            upsampled = cv2.resize(blurry_np[pat, lens_pos, ...], up_dim, interpolation=cv2.INTER_LINEAR)
            temp = undistort_image(upsampled, camera_matrix, distortion_coefficients)
            undistorted[pat, lens_pos, ...] = cv2.resize(temp, dim, interpolation=cv2.INTER_LINEAR)
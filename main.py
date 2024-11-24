import cv2
import numpy as np
import glob
from utils import *

output_path = "output/"

if __name__ == "__main__":
    # Camera calibration
    matrix_coeffs, dist_coeffs = calibrate_camera("camera_cal/")

    # Example distortion correction
    img = cv2.imread('camera_cal/calibration3.jpg')
    undistorted = cv2.undistort(img, matrix_coeffs, dist_coeffs, None, matrix_coeffs)
    cv2.imwrite(f'{output_path}undistorted_image.jpg', undistorted)

    
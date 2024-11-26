import cv2
import numpy as np
import glob
import os
from utils import *

output_path = "output/"
calibration_path = "camera_cal/"

if __name__ == "__main__":
    # Camera calibration.
    matrix_coeffs, dist_coeffs = calibrate_camera(calibration_path)

    # Example distortion correction.
    img = cv2.imread('test_images/whiteCarLaneSwitch.jpg')
    undistorted = cv2.undistort(img, matrix_coeffs, dist_coeffs, None, matrix_coeffs)
    cv2.imwrite(f'{output_path}undistorted_image.jpg', undistorted)

    # Create a binary image.
    binary_image = create_binary_image(undistorted)
    cv2.imwrite(f'{output_path}/binary_image.jpg', (binary_image * 255).astype(np.uint8))

    # Apply a perspective transform - Get a bird's-eye view of the road.
    warped, inverse_matrix = perspective_transform(binary_image)
    cv2.imwrite('output/warped_image.jpg', (warped * 255).astype(np.uint8) )

    # Identify lane pixels and fit lane lines using a polynomial.
    color_fit_lines = detect_lane_pixels_and_fit(warped)
    
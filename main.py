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
    img = cv2.imread(f'{calibration_path}calibration3.jpg')
    undistorted = cv2.undistort(img, matrix_coeffs, dist_coeffs, None, matrix_coeffs)
    cv2.imwrite(f'{output_path}undistorted_image.jpg', undistorted)

    # Create a binary image.
    binary_image = create_binary_image(undistorted)
    cv2.imwrite(f'{output_path}/binary_undistorted.jpg', (binary_image * 255).astype(np.uint8))

    # Generate a binary image for every image inside test_image dir.
    images = glob.glob('test_images/' + "*.jpg")
    for fname in images:  
        image = cv2.imread(fname)
        binary = create_binary_image(image)
        cv2.imwrite(f'{output_path}/binary_{os.path.basename(fname)}', (binary * 255).astype(np.uint8))

    # Apply a perspective transform - Get a bird's-eye view of the road.
    image = cv2.imread('test_images/whiteCarLaneSwitch.jpg')
    warped, inverse_matrix = perspective_transform(image)
    cv2.imwrite('output/warped.jpg', warped )
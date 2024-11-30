import cv2
import numpy as np
import glob
import os
from utils import *

output_path = "output/"
calibration_path = "camera_cal/"
example_image = "test_images/whiteCarLaneSwitch.jpg"
example_video = "test_videos/project_video01.mp4"

if __name__ == "__main__":
    # Camera calibration.
    matrix_coeffs, dist_coeffs = calibrate_camera(calibration_path)

    # Example distortion correction.
    img = cv2.imread(example_image)
    undistorted = cv2.undistort(img, matrix_coeffs, dist_coeffs, None, matrix_coeffs)
    cv2.imwrite(f'{output_path}undistorted_image.jpg', undistorted)

    # Create a binary image.
    binary_image = create_binary_image(undistorted)
    cv2.imwrite(f'{output_path}/binary_image.jpg', (binary_image * 255).astype(np.uint8))

    # Apply a perspective transform - Get a bird's-eye view of the road.
    warped, inverse_matrix = perspective_transform(binary_image)
    cv2.imwrite(f'{output_path}/warped_image.jpg', (warped * 255).astype(np.uint8) )

    # Identify lane pixels and fit lane lines using a polynomial.
    left_fit, right_fit, color_fit_lines_image = detect_lane_pixels_and_fit(warped)
    
    # Calculate curvature and vehicle position
    left_curverad, right_curverad, center_dist = calculate_curvature_and_position(warped, left_fit, right_fit)
    result = draw_lane_lines(undistorted, warped, (left_fit, right_fit), inverse_matrix)

    process_video(example_video, matrix_coeffs)
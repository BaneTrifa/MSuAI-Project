import cv2
import numpy as np
import glob

def calibrate_camera(calibration_images_path):
    objpoints = []  # 3D points in real-world space.
    imgpoints = []  # 2D points in image plane.

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    images = glob.glob(calibration_images_path + "*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def create_binary_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelX)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    return binary_output

def perspective_transform(image, display_frame=False):
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([[175, image.shape[0]], [900, image.shape[0]], [550, 350], [425, 350]])
    dst = np.float32([[200, image.shape[0]], [900, image.shape[0]], [900, 0], [200, 0]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, matrix, img_size)

    # Visualize boundaries for the transformations.
    if display_frame:
        # Convert the points to integer format for OpenCV functions.
        pts_src = src.astype(int)
        pts_dst = dst.astype(int)

        # Draw the polygon outline.
        cv2.polylines(image, [pts_src], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(warped, [pts_dst], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the image.
        cv2.imshow("Polygon Area", cv2.hconcat([image, warped]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped, inverse_matrix

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def detect_lane_pixels_and_fit(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    plot_fit_lines(left_fit, right_fit, out_img)
    
    return cv2.imread('output/color_fit_lines.jpg')

def plot_fit_lines(left_curvature, right_curvature, image):

    # Image dimensions.
    height, _, _ = image.shape

    # Generate y-values (vertical positions).
    y_values = np.linspace(0, height - 1, num=height)

    # A quadratic curve: y = Ax^2 + Bx + C.
    A_left, B_left, C_left = left_curvature
    A_right, B_right, C_right = right_curvature

    # Calculate x-values for the left and right lanes.
    x_left = A_left * y_values**2 + B_left * y_values + C_left
    x_right = A_right * y_values**2 + B_right * y_values + C_right

    # Plot and save the image.
    plt.imshow(image)
    plt.plot(x_left, y_values, color='red', linewidth=5, label='Left Lane')
    plt.plot(x_right, y_values, color='blue', linewidth=5, label='Right Lane')
    plt.legend()
    plt.title("Lane Curvatures on Image")
    plt.axis("off")
    plt.savefig('output/color_fit_lines.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
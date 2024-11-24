**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration

In this step, the camera calibration is performed using chessboard images. Specifically:

1. Chessboard Corners Detection:

The findChessboardCorners function from OpenCV is used to detect the inner corners of a chessboard pattern in calibration images.
These detected corners represent the observed image points.

2. Object Points Generation:

A corresponding set of 3D points in the real world (assuming the chessboard lies flat on the z=0 plane) is generated. These are referred to as object points.

3. Camera Matrix and Distortion Coefficients Calculation:

The calibrateCamera function is used with the object points and image points. This computes the camera matrix and distortion coefficients, which correct lens distortion.

4. Distortion Correction:

Using the computed matrix and coefficients, images are undistorted with cv2.undistort, removing distortion and producing geometrically accurate images.


| Distorted  | Undistorted |
| ------------- | ------------- |
| ![Distorted](./camera_cal/calibration3.jpg)   | ![Undistorted](./output/undistorted_image.jpg)  |

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!


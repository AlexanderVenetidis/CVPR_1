"""
parameters about the camera are required to determine an accurate relationship
between a 3D point in the real world and its corresponding 2D projection (pixel) 
in the image captured by that calibrated camera.
"""

import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import pickle

# parameters (in mm)
square_size = 23
aperture_width = 35.8
aperture_height = 23.9

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

for fname in glob.glob("calib_imgs/*.jpg"):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        drawing = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        # plt.imshow(drawing)
        # plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# choosing image to undistort
img = cv2.imread("calib_imgs/IMG_9144.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# saving calibration arrays
mtx.dump("mtx.npy")
newcameramtx.dump("newcameramtx.npy")
dist.dump("cam_dist.npy")
with open("cam_roi.pkl", 'wb') as f: pickle.dump(roi, f)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]

cv2.imwrite("calib.jpg", dst)
cv2.imwrite("uncalib.jpg", img)

print(newcameramtx)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

### Real unit focal length
Fx = newcameramtx[0,0] * aperture_width / img.shape[1]
Fy = newcameramtx[1,1] * aperture_height / img.shape[0]
print(f'Fx = {Fx}mm, Fy = {Fy}mm')
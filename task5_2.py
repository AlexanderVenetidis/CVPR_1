
# Start writing code here...
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import random
from undistort import undistort_img


img1_big = (cv.imread('./fd_jpg/IMG_9125.jpg')) # queryImage
img2_big = (cv.imread('./fd_jpg/IMG_9129.jpg')) # trainImage
img1 = img1_big#cv.resize(img1_big, (1368, 912))
img2 = img2_big#cv.resize(img2_big, (1368, 912))

def drawMatches(img1, kp1, img2, kp2, matches, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 7
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = ( int (c [ 0 ]), int (c [ 1 ]), int (c [ 2 ]))


        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.

        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv.line(new_img, end1, end2, c, thickness)
        cv.circle(new_img, end1, r, c, thickness)
        cv.circle(new_img, end2, r, c, thickness)

    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()

#Then using SIFT method

# img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.35*n.distance:
        good.append(m)

pts1 = []
pts2 = []
# for x in [good[20]] + [good[3]] + [good[34]] + [good[64]] + [good[17]] + [good[13]] + [good[56]] + [good[101]]:
for x in good:
    pts1.append(list(np.round(kp1[x.queryIdx].pt).astype(int)))
    pts2.append(list(np.round(kp2[x.trainIdx].pt).astype(int)))

# print(pts1)

# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# drawMatches(img1, kp1, img2, kp2, [good[20]] + [good[3]] + [good[34]] + [good[64]] + [good[17]] + [good[13]] + [good[56]] + [good[101]], color=None)
# plt.imshow(img3),plt.show()

#
# #images 9125 and 9127
# img1 = cv.imread('./fd_jpg/IMG_9125.jpg')
# img1 = cv.resize(img1, (1368, 912))
# pts1 = [[409,212], [412,359], [470,498], [715,461], [688,560], [997,268], [1168,192], [1171,339]]
#
# img2 = cv.imread('./fd_jpg/IMG_9127.jpg')
# img2 = cv.resize(img2, (1368, 912))
# pts2 = [[190,191], [197,364], [231,526], [648,461], [647,575], [676,251], [900,166], [903,316]]
#
#
#
#
# #corresponding points from image 1 and image 2

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# print(pts1,pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# print(F)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
#
#
# #Drawing the epipolar lines
#
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    # print(img1.shape)
    r,c,_ = img1.shape
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,4)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
#
#
#
# Find epilines corresponding to points in right image (second image) and

# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1[:20],pts1[:20],pts2[:20])
# Find epilines corresponding to points in left image (first image) and

# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2[:20],pts2[:20],pts1[:20])

#
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()
#
#
# #using the fundamental matrix calculated in task 4_2 (shown above)
#
# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1,_ = img1.shape
h2, w2,_ = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
)
#
#What we get back are the transformations encoded by the homography matrices H1 and H2

# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
# cv.imwrite("rectified_1.png", img1_rectified)
# cv.imwrite("rectified_2.png", img2_rectified)

# Draw the rectified images
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img2_rectified, cmap="gray")
# axes[1].imshow(img1_rectified, cmap="gray")
# axes[0].axhline(250)
# axes[1].axhline(250)
# axes[0].axhline(450)
# axes[1].axhline(450)
# plt.suptitle("Rectified images")
# # plt.savefig("rectified_images.png")
# plt.show()

img2_gray = cv.cvtColor(img2_rectified, cv.COLOR_BGR2GRAY)
img1_gray = cv.cvtColor(img1_rectified, cv.COLOR_BGR2GRAY)

window_size = 3
min_disp = 0
num_disp = 110-min_disp

args = dict(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 12,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32)
stereo = cv.StereoSGBM_create(args)
disparity = stereo.compute(img1_gray,img2_gray)

# sigma = 1.8
# lmbda = 1000.0

# left_matcher = cv.StereoSGBM_create(**args)
# right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
# left_disp = left_matcher.compute(img1_gray, img2_gray)
# right_disp = right_matcher.compute(img2_gray,img1_gray)

# Now create DisparityWLSFilter
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
filtered_disp = wls_filter.filter(left_disp, img1_gray, disparity_map_right=right_disp)

plt.imshow(filtered_disp,cmap = 'plasma')
# plt.imshow(img1_rectified)
# plt.show()
plt.colorbar(shrink=.7)
plt.show()

plt.imshow(disparity,cmap = 'plasma')
plt.colorbar(shrink=.7)
plt.show()

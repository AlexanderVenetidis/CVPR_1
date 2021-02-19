# Start writing code here...
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

#images 9125 and 9127
img1 = cv.imread('./fd_jpg/IMG_9128.jpg')
img1 = cv.resize(img1, (1368, 912))
# pts1 = [
# [407,651], [368,675], [540,179], [401,316], [446,361], [555,302], [558,363], [416,444],
# [789, 154], [847, 157], [904, 155]    ,  [807,332],[920,338],
# [698, 497],[686,590], [759,524], [757,558], [929, 570], [661,493], [650, 586], [814,513], [798,574]
# ]

img2 = cv.imread('./fd_jpg/IMG_9129.jpg')
img2 = cv.resize(img2, (1368, 912))
# pts2 = [
# [356,641], [286,645], [680,191], [411,272], [486,340], [686,333], [688,401], [417,417],
# [932, 188], [968, 186], [1009,182]    , [938, 408], [1015, 411],
#  [526, 460],[489,564], [545, 494],[534,528],[716,581], [494,453], [453,552], [596,485], [562,550]
# ]


row, col = img2.shape[:2]
bottom = img2[row-2:row, 0:col]
mean = cv.mean(bottom)[0]

bordersize = 2000
img1 = cv.copyMakeBorder(
    img1,
    top=0,
    bottom=0,
    left=0,
    right=bordersize,
    borderType=cv.BORDER_CONSTANT,
    value=[255, 255, 255]
)



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
    if m.distance < 0.38*n.distance:
        good.append(m)
# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.imshow(img3),plt.show()

pts1 = []
pts2 = []

# drawMatches(img1, kp1, img2, kp2, good, color=None)
print(pts1)

for x in good:
    pts1.append(list(np.round(kp1[x.queryIdx].pt).astype(int)))
    pts2.append(list(np.round(kp2[x.trainIdx].pt).astype(int)))
print(pts1)
print(len(good), len(pts1))

# pts1 = [ m.pt for m in kp1]
# pts2 = [ m.pt for m in kp2]

#
# pts1 = np.reshape(np.array(pts1),(-1,2))
# pts2 = np.reshape(np.array(pts2),(-1,2))
# print(pts1.shape, pts2.shape)









# for x in pts1:
#     x[0] += 1000
# for x in pts2:
#     x[0] += 10000

#corresponding points from image 1 and image 2
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
# print(pts1.shape)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.RANSAC)
print(F)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# F, mask = cv.findFundamentalMat(pts1,pts2,cv.RANSAC)
# print(F)
# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]
#

#Drawing the epipolar lines

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
        img1 = cv.circle(img1,tuple(pt1),10,color,-1)
        img2 = cv.circle(img2,tuple(pt2),10,color,-1)
    return img1,img2

def drawpoints(img1,img2,lines,pts1,pts2):
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
        # img1 = cv.line(img1, (x0,y0), (x1,y1), color,4)
        img1 = cv.circle(img1,tuple(pt1),10,color,-1)
        img2 = cv.circle(img2,tuple(pt2),10,color,-1)
    return img1,img2



# Find epilines corresponding to points in right image (second image) and




# drawing its lines on left image

# Find epilines corresponding to points in left image (first image) and

# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawpoints(img2,img1,lines2,pts2,pts1)


lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)



plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()


#using the fundamental matrix calculated in task 4_2 (shown above)

# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1,_ = img1.shape
h2, w2,_ = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
)

#What we get back are the transformations encoded by the homography matrices H1 and H2

# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
cv.imwrite("rectified_1.png", img1_rectified)
cv.imwrite("rectified_2.png", img2_rectified)

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

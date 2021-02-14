# Start writing code here...
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

#images 9125 and 9127
img1 = cv.imread('./fd_jpg/IMG_9125.jpg')
img1 = cv.resize(img1, (1368, 912))
pts1 = [[409,212], [412,359], [470,498], [715,461], [688,560], [997,268], [1168,192], [1171,339]]

img2 = cv.imread('./fd_jpg/IMG_9127.jpg')
img2 = cv.resize(img2, (1368, 912))
pts2 = [[190,191], [197,364], [231,526], [648,461], [647,575], [676,251], [900,166], [903,316]]




#corresponding points from image 1 and image 2
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
print(pts1.shape)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


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
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2



# Find epilines corresponding to points in right image (second image) and

# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and

# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()


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

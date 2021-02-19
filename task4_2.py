# Start writing code here...
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

#images 9125 and 9127
img1 = cv.imread('./fd_jpg/IMG_9133.jpg')
img1 = cv.resize(img1, (1368, 912))
pts1 = [
[407,651], [368,675], [540,179], [401,316], [446,361], [555,302], [558,363], [416,444],
[789, 154], [847, 157], [904, 155]    #  [807,332],[920,338],
# [698, 497],[686,590], [759,524], [757,558], [929, 570], [661,493], [650, 586], [814,513], [798,574]
]

img2 = cv.imread('./fd_jpg/IMG_9130.jpg')
img2 = cv.resize(img2, (1368, 912))
pts2 = [
[356,641], [286,645], [680,191], [411,272], [486,340], [686,333], [688,401], [417,417],
[932, 188], [968, 186], [1009,182]    # [938, 408], [1015, 411],
# [526, 460],[489,564], [545, 494],[534,528],[716,581], [494,453], [453,552], [596,485], [562,550]
]


# for x in pts1:
#     x[0] += 1000
# for x in pts2:
#     x[0] += 10000

#corresponding points from image 1 and image 2
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
# print(pts1.shape)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
print(F)
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
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawpoints(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and

# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)

# row, col = img2.shape[:2]
# bottom = img2[row-2:row, 0:col]
# mean = cv.mean(bottom)[0]
#
# bordersize = 0
# img2 = cv.copyMakeBorder(
#     img2,
#     top=0,
#     bottom=0,
#     left=bordersize,
#     right=0,
#     borderType=cv.BORDER_CONSTANT,
#     value=[mean, mean, mean]
# )

img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


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

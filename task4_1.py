# Start writing code here...
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from undistort import undistort_img

def draw_matches(img1, kp1, img2, kp2, color=None):
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
    thickness = 2
    if color:
        c = color
    for m in range(len(kp1)):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = ( int (c [ 0 ]), int (c [ 1 ]), int (c [ 2 ]))
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m]).astype(int))
        end2 = tuple(np.round(kp2[m]).astype(int) + np.array([img1.shape[1], 0]))
        # print(end1,end2)
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()



# Read source image.
im_src = undistort_img(cv2.imread('./hg_jpg/IMG_9135.jpg'))
im_src = cv2.resize(im_src, (1368, 912))
# Four keypoints source image (or use SIFT from task2)
pts_src = np.array([[445, 290], [413, 517], [989, 277],[989, 500], [489,339],[861,328]])

pts_dst = np.array([[333, 318],[362, 620],[1023, 131],[1095, 414], [404,364], [876,234]])


# Read destination image.
im_dst = undistort_img(cv2.imread('./hg_jpg/IMG_9138_2.jpg'))
im_dst = cv2.resize(im_dst, (1368, 912))
# Four same keypoints in destination image. (find using mouse)

# Calculate Homography (h is homography matrix)
h, status = cv2.findHomography(pts_src, pts_dst)
print(h)

# print(h.shape, pts_src.shape)

proj_pts = np.float32(pts_src).reshape(-1,1,2)

proj_pts =  cv2.perspectiveTransform(proj_pts, h).reshape(6,2)


# print(pts_dst, proj_pts)


# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

# #drawing correspondence between points used
draw_matches(im_src, pts_src, im_dst, proj_pts, color=None)

# Display images
# cv2.imshow("Source Image", im_src)
# cv2.imshow("Destination Image", im_dst)
# cv2.imshow("Warped Source Image", im_out)
# #
# cv2.waitKey(0)


# #plot destination image and warp image side by side
# new_shape = (max(im_dst.shape[0], im_out.shape[0]), im_out.shape[1]+im_out.shape[1], im_dst.shape[2])
# # plt.show()
# new_img = np.zeros(new_shape, type(im_dst.flat[0]))
# # # Place images onto the new image.
# new_img[0:im_dst.shape[0],0:im_dst.shape[1]] = im_dst
# new_img[0:im_out.shape[0],im_dst.shape[1]:im_dst.shape[1]+im_out.shape[1]] = im_out
# plt.figure(figsize=(15,15))
# plt.imshow(new_img)
# plt.show()

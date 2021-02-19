from matplotlib import pyplot as plt
import cv2 as cv

# img1 = cv.imread('./hg_jpg/IMG_9135.jpg')
img1 = cv.imread('./hg_jpg/IMG_9138_2.jpg')

img1 = cv.resize(img1, (1368, 912))

plt.imshow(img1)
plt.show()

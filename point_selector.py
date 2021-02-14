import numpy as np
import cv2

img=cv2.imread('./fd_jpg/IMG_9127.jpg')
img = cv2.resize(img, (1368, 912))

# cv2.imshow('Test',img)


cv2.imshow('image', img)

#define the events for the
# mouse_click.
def mouse_click(event, x, y,
                flags, param):

    # to check if left mouse
    # button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:

        # font for left click event
        font = cv2.FONT_HERSHEY_DUPLEX
        LB = str(x) + ' ' + str(y)

        # display that left button
        # was clicked.
        cv2.putText(img, LB, (x, y),
                    font, 1,
                    (255, 255, 0),
                    2)
        cv2.imshow('image', img)


    # to check if right mouse
    # button was clicked
    if event == cv2.EVENT_RBUTTONDOWN:

        # font for right click event
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        RB = 'Right Button'

        # display that right button
        # was clicked.
        cv2.putText(img, RB, (x, y),
                    font, 1,
                    (0, 255, 255),
                    2)
        cv2.imshow('image', img)

cv2.setMouseCallback('image', mouse_click)

cv2.waitKey(0)

# close all the opened windows.
cv2.destroyAllWindows()

# def draw_circle(event,x,y,flags,param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#         mouseX,mouseY = x,y
#
# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
#
# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#     elif k == ord('a'):
#         print(mouseX,mouseY)

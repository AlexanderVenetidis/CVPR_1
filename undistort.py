import cv2
import numpy as np
import pickle

dist = np.load('cam_dist.npy')
mtx = np.load('mtx.npy')
newcameramtx = np.load('newcameramtx.npy')

with open('cam_roi.pkl', 'rb') as f: 
	roi = pickle.load(f)

def undistort_img(img):
	img = cv2.undistort(img, mtx, dist, None, newcameramtx)
	x, y, w, h = roi
	return img[y : y + h, x : x + w]


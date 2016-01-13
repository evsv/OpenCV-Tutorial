import cv2
from sklearn import datasets

digits = datasets.load_digits()

cv2.imshow('test',digits.images[3])
cv2.waitKey(0)

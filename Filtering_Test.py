import cv2
import numpy as np
from matplotlib import pyplot as plt

K_124 = cv2.imread('KKNDHLI_124.jpg')
K_118 = cv2.imread('KKNDHLI_118.jpg')
cv2.imshow('test_118', K_124)
cv2.imshow('test_124', K_118)
cv2.waitKey(0)

#Testing 2D convolution
kernel_5x5 = np.ones((5,5),np.float32)/25
kernel_3x3 = np.ones((3,3), np.float32)/9
dst_5x5 = cv2.filter2D(K_118,-1,kernel_5x5)
dst_3x3 = cv2.filter2D(K_118,-1,kernel_3x3)
cv2.imshow('orig', K_124)
cv2.imshow('filtertest_5x5', dst_5x5)
cv2.imshow('filtertest_3x3', dst_3x3)
cv2.waitKey(0)

#Testing bilateral filtering
K_118_bilateral = cv2.bilateralFilter(K_118,9,75,75)
cv2.imshow('orig', K_118)
cv2.imshow('test_bilateral', K_118_bilateral)
cv2.waitKey(0)


#This is a test commit
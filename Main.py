import cv2
import numpy as np

KKNDHLI_L0 = cv2.imread('KKNDHLI_L0.jpg')
K_124 = cv2.imread('KKNDHLI_124.jpg')
K_118 = cv2.imread('KKNDHLI_118.jpg')
Q1_KRIE = cv2.imread('Q1_KRIE.jpg')

cv2.imshow('test', KKNDHLI_L0)
cv2.imshow('test_118', K_124)
cv2.imshow('test_124', K_118)
cv2.waitKey(0)

KKN_ROI = KKNDHLI_L0[0:133, 0:422]

KKNDHLI_L0_Test = KKNDHLI_L0
KKNDHLI_L0[133:266, 422:844] = KKN_ROI
cv2.imshow('test2', KKNDHLI_L0_Test)
cv2.waitKey(0)

#Lines to test image splitting
b,g,r = cv2.split(KKNDHLI_L0_Test)
cv2.imshow('btest',b)
cv2.imshow('gtest',g)
cv2.imshow('rtest',r)
cv2.waitKey(0)

#Lines to test the HSV colour conversion and masking
KKN_Test2 = KKNDHLI_L0
KKN_HSV = cv2.cvtColor(KKN_Test2,cv2.COLOR_BGR2HSV)
cv2.imshow('hsvtest', KKN_HSV)
cv2.waitKey(0)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

mask = cv2.inRange(KKN_HSV, lower_blue, upper_blue)
cv2.imshow('hsvtest', mask)
cv2.waitKey(0)

KKN_Blue = cv2.bitwise_and(KKN_Test2,KKN_Test2, mask= mask)
cv2.imshow('hsvtest', KKN_Blue)
cv2.waitKey(0)

#Code for image thresholding
k_118_b,k_118_g,k_118_r= cv2.split(K_118)
cv2.imshow('btest',k_118_b)
cv2.imshow('gtest',k_118_g)
cv2.imshow('rtest',k_118_r)
cv2.waitKey(0)
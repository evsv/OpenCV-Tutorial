import cv2
import numpy as np
from matplotlib import pyplot as plt

K_124 = cv2.imread('KKNDHLI_124.jpg')
K_118 = cv2.imread('KKNDHLI_118.jpg')
cv2.imshow('test_118', K_124)
cv2.imshow('test_124', K_118)
cv2.waitKey(0)

#Code for image splitting
k_118_b,k_118_g,k_118_r= cv2.split(K_118)
cv2.imshow('btest',k_118_b)
cv2.imshow('gtest',k_118_g)
cv2.imshow('rtest',k_118_r)
cv2.waitKey(0)

#Code for thresholding the blue layer
ret,thresh110 = cv2.threshold(k_118_b,110,255,cv2.THRESH_BINARY)
ret,thresh130 = cv2.threshold(k_118_b,130,255,cv2.THRESH_BINARY)
ret,thresh150 = cv2.threshold(k_118_b,150,255,cv2.THRESH_BINARY)
ret,thresh160 = cv2.threshold(k_118_b,160,255,cv2.THRESH_BINARY)
ret,thresh170 = cv2.threshold(k_118_b,170,255,cv2.THRESH_BINARY)

#Code for displaying images
cv2.imshow('thresh170',thresh170) #Too discriminating
cv2.imshow('thresh160',thresh160) #Seems OK
cv2.imshow('thresh150',thresh150) #Seems OK
#cv2.imshow('thresh130',thresh130)
#cv2.imshow('thresh110',thresh110)
cv2.imshow('orig',K_118)
cv2.waitKey(0)

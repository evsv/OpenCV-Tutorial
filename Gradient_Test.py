import cv2
import numpy as np

KKNDHLI_L0 = cv2.imread('KKNDHLI_L0.jpg')
K_124 = cv2.imread('KKNDHLI_124.jpg')
K_118 = cv2.imread('KKNDHLI_118.jpg')
#cv2.imshow('test_118', K_124)
#cv2.imshow('test_124', K_118)
#cv2.waitKey(0)

#Splitting the given images into BGR intensity images
K_118_B, K_118_G, K_118_R = cv2.split(K_118)

#Applying the filters on the B image
K_118_B_laplacian = cv2.Laplacian(K_118_B,cv2.CV_64F)
K_118_B_sobelx = cv2.Sobel(K_118_B,cv2.CV_64F,1,0,ksize=5)
K_118_B_sobely = cv2.Sobel(K_118_B,cv2.CV_64F,0,1,ksize=5)

K_118_laplacian = cv2.Laplacian(K_118,cv2.CV_64F)
K_118_sobelx = cv2.Sobel(K_118,cv2.CV_64F,1,0,ksize=3)
K_118_sobely = cv2.Sobel(K_118,cv2.CV_64F,0,1,ksize=3)
cv2.imshow('Laplacian',K_118_laplacian)
cv2.imshow('SobelX', K_118_sobelx)
cv2.imshow('SobelY', K_118_sobely)
cv2.imshow('Original', K_118)
cv2.waitKey(0)



cv2.imshow('BlueChannel',K_118_B)
cv2.imshow('GreenChannel',K_118_G)
cv2.imshow('RedChannel',K_118_R)
cv2.waitKey(0)

cv2.imshow('Laplacian',K_118_B_laplacian)
cv2.imshow('SobelX', K_118_B_sobelx)
cv2.imshow('SobelY', K_118_B_sobely)
cv2.imshow('Original', K_118)
cv2.imshow('BlueChannel',K_118_B)
cv2.waitKey(0)

#Using the Canny Edge Detection
K_118_B_Canny_50 = cv2.Canny(K_118_B,50,100,L2gradient=True)
K_118_B_Canny_100 = cv2.Canny(K_118_B,100,100,L2gradient=True)
K_118_Canny_50 = cv2.Canny(K_118,50,100,L2gradient=True)
K_118_Canny_100 = cv2.Canny(K_118,100,100,L2gradient=True)
#cv2.imwrite('CannyTest_50_B.png', K_118_B_Canny_50)
#cv2.imwrite('CannyTest_100_B.png', K_118_B_Canny_100)
cv2.imwrite('CannyTest_50.png', K_118_Canny_50)
cv2.imwrite('CannyTest_100.png', K_118_Canny_100)
#cv2.imshow('Original', K_118_B)
#cv2.waitKey(0)


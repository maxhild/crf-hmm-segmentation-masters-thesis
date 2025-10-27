from erode_dilate import opening, erode, dilate, closing, binarize
from masks import sobel
import cv2
import numpy as np


def kernels(x):
    kernel1 = np.array([[0, x, 1], [0, 1, x], [0, x, 1]])
    kernel2 = np.array([[0, 0, 0], [x, 1, x], [1, x, 1]])
    kernel3 = np.array([[0, 0, x], [0, 1, 1], [x, 1, x]])
    kernel4 = np.array([[x, 0, 0], [1, 1, 0], [x, 1, x]])
    kernel5 = np.array([[1, x, 0], [x, 1, 0], [1, x, 0]])
    kernel6 = np.array([[x, 1, x], [1, 1, 0], [x, 0, 0]])
    kernel7 = np.array([[1, x, 1], [x, 1, x], [0, 0, 0]])
    kernel8 = np.array([[x, 1, x], [0, 1, 1], [0, 0, x]])
    
    return [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8]

    
    
def thinning(img, x=1):
    kernels_list = kernels(x)
    output_img = img.copy()
    
    while True:
        marker = np.zeros_like(output_img)
        #print(f"marker:{marker}")
        for kernel in kernels_list:
            hit_or_miss = cv2.morphologyEx(output_img, cv2.MORPH_HITMISS, kernel)
            marker = cv2.bitwise_or(marker, hit_or_miss)
            #print(f"marker:{marker}kernel:{kernel}")
        
        temp_img = cv2.bitwise_and(output_img, cv2.bitwise_not(marker))
        if np.array_equal(output_img, temp_img):
            break
        output_img = temp_img.copy()
    
    return output_img

img=cv2.imread("frame1.jpg", cv2.IMREAD_GRAYSCALE)
#img=binarize(img)
#img=sobel(img) # für Kantendetektion
cv2.imwrite("thinning.png", thinning(img, x=1))

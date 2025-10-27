# erode_dilate.py
from masks import laplace, sobel, LOG, gauss_kernel, gen_gauss_image, Difference_of_Gaussians
import cv2
import numpy as np

def binarize(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")

    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return binary_img.astype(np.uint8)

def erode(img, kernel):
    return cv2.erode(img, kernel)

def dilate(img, kernel):
    return cv2.dilate(img, kernel)

def opening(img, kernel):
    return cv2.dilate(cv2.erode(img, kernel), kernel)

def closing(img, kernel):
    return cv2.erode(cv2.dilate(img, kernel), kernel)

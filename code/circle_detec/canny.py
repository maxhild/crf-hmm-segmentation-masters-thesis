import cv2
import numpy as np

img = cv2.imread("frame3.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Denoise & flatten illumination
img_blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# 2. Edge detection
edges = cv2.Canny(img_blur, 40, 120)

# 3. Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("edges_cleaned.png", edges_closed)

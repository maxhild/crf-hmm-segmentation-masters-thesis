import cv2
import numpy as np

img = cv2.imread("frame1.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Denoise & flatten illumination
img_blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# 3. Edge detection (lower thresholds so less is "lost")
edges = cv2.Canny(img_blur, 30, 90)  # instead of 40,120

# 4. Morphological cleanup (smaller kernel and/or just a dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("canny2.png", edges_closed)

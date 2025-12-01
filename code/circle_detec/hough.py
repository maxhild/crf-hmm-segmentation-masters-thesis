import cv2
import numpy as np

img = cv2.imread("frame1.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Denoise & flatten illumination
img_blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

import cv2
import numpy as np

img = cv2.imread("frame1.jpg", cv2.IMREAD_GRAYSCALE)
img_blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

h, w = img.shape[:2]
min_r = int(0.1 * min(h, w))   # tune this
max_r = int(0.1 * min(h, w))   # tune this

img = img[200:800, 540:200000]


circles = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,           # accumulator resolution
    minDist=100,       # min distance between circle centers
    param1=80,        # high threshold for the internal Canny
    param2=45,        # accumulator threshold (lower -> more circles)
    minRadius=1,     # tune!
    maxRadius=0       # 0 = no upper bound (or set a value)
)


# Visualize circles on the original image
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if circles is not None:
    print(len(circles))

    circles = np.round(circles[0, :]).astype(int)
    for (x, y, r) in circles:
        # draw outer circle
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        # draw center point
        cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)

cv2.imwrite("circles_visualization3.png", vis)

import cv2
import numpy as np

def detect_red_regions(img):
    """Detect significant red presence in HSV color space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red can appear near 0° or near 180° hue
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    ratio_red = np.sum(mask > 0) / mask.size
    return ratio_red > 0.03, mask, ratio_red  # >3% red pixels = red region


def detect_nbi_filter(img):
    """Detect Narrow Band Imaging (NBI) presence.
    NBI typically has muted reds, enhanced green/blue, and overall darker tone."""
    mean_bgr = np.mean(img, axis=(0, 1))
    b, g, r = mean_bgr

    # Simple heuristic: NBI tends to have strong G/B and weak R
    nbi_like = (g > r * 1.2) and (b > r * 1.2) and np.mean(mean_bgr) < 140
    return nbi_like, (b, g, r)


def detect_circles(img_gray):
    """Detect circular features using Hough Transform."""
    blurred = cv2.medianBlur(img_gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=150,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        output = img.copy()
        for (x, y, r) in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        return True, output, len(circles[0])
    else:
        return False, img, 0


def detect_lines(img_gray):
    """Detect straight lines using probabilistic Hough transform."""
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=10,
    )

    output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return True, output, len(lines)
    else:
        return False, output, 0


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Detect red regions
    red_present, red_mask, red_ratio = detect_red_regions(img)

    # 2️⃣ Detect NBI filtering
    nbi_present, mean_bgr = detect_nbi_filter(img)

    # 3️⃣ Detect circles
    circles_present, img_circles, num_circles = detect_circles(gray)

    # 4️⃣ Detect lines
    lines_present, img_lines, num_lines = detect_lines(gray)

    # Combine masks for visualization
    red_overlay = cv2.bitwise_and(img, img, mask=red_mask)

    print("==== Image Analysis Summary ====")
    print(f"Red regions present: {red_present} ({red_ratio*100:.2f}% of image)")
    print(f"NBI-like filter detected: {nbi_present} (mean BGR = {mean_bgr})")
    print(f"Circles detected: {circles_present} ({num_circles} found)")
    print(f"Lines detected: {lines_present} ({num_lines} found)")

    # Visualization
    cv2.imshow("Original", img)
    cv2.imshow("Red Mask", red_overlay)
    cv2.imshow("Detected Circles", img_circles)
    cv2.imshow("Detected Lines", img_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_features.py <image_path>")
        sys.exit(1)
    analyze_image(sys.argv[1])

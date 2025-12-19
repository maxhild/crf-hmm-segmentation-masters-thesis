import cv2
import numpy as np

def extract_features(prev_frame, current_frame):
    # 1. Motion (using Optical Flow on grayscale)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Reduce to global motion (Median is robust against noise)
    vx = np.median(flow[..., 0])
    vy = np.median(flow[..., 1])
    
    # 2. Color (Convert to HSV for better separation)
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    # Calculate means of H, S, and V channels
    mu_h, mu_s, mu_v = cv2.mean(hsv)[:3]
    
    # 3. Texture (Variance of Laplacian)
    # High value = Sharp/Texture; Low value = Blurry/Flat
    texture_energy = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
    
    # Final Measurement Vector y_t
    y_t = np.array([vx, vy, mu_h, mu_s, mu_v, texture_energy])
    
    return y_t
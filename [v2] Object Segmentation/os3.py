import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# loading image
os.chdir("F:\\_VScode1\\CV_2_Object Segmentation\\threshold")
img_path = r'F:\_VScode1\CV_2_Object Segmentation\image2.jpg' 
img = cv2.imread(img_path)

# Converting it to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Otsu for rough foreground/background separation
ret, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
foreground_mask = (otsu_thresh == 255)  # True for foreground pixels

# Step 2: Multi-layering on current foreground only, for filtering out actual foreground
gray_foreground = gray.copy()
gray_foreground[~foreground_mask] = 0  # Zero out background

# Apply multi-level thresholds
gray_multi = np.zeros_like(gray, dtype=np.uint8)
mean_val = gray_foreground[foreground_mask].mean()
gray_multi = (gray_foreground > gray_foreground.mean()).astype(np.uint8) * 255
gray_multi[(gray_foreground > mean_val)] = 3
gray_multi[(gray_foreground <= mean_val) & (gray_foreground > 0.5 * 255)] = 2
gray_multi[(gray_foreground <= 0.5 * 255) & (gray_foreground > 0.25 * 255)] = 1
kernel = np.ones((2,2),np.uint8)
dilated_foreground = cv2.dilate(gray_multi, kernel, iterations=2)

plt.imshow(dilated_foreground, cmap='gray')
plt.show()

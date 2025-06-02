import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Set working directory
os.chdir("F:\\_VScode1\\CV_2_Object Segmentation")
print("Current working directory:", os.getcwd())

# Check for files in the threshold directory
target_dir = r'F:\_VScode1\CV_2_Object Segmentation'
print("Files in threshold directory:", os.listdir(target_dir))

img_path = r'F:\_VScode1\CV_2_Object Segmentation\image2.jpg'  
# Verify the image
try:
    with Image.open(img_path) as img:
        img.verify()
    print("Image is valid")
except Exception as e:
    print(f"Image is corrupted: {e}")

# Load the image
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    print("Check if the path is correct and the file exists.")
    exit()


#b, g, r = cv2.split(img)  
#rgb_img = cv2.merge([r, g, b])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# First thresholding (from rgb to binary)
mean_val=gray.mean()
gray_binary=(gray>mean_val).astype(np.uint8)*255 #vectorization using NumPy's optimized C-based backend enables faster operation on entire arrays instead of using explicit for loops

# Second thresholding (multi-level)
gray_multi=np.zeros_like(gray, dtype=np.uint8)
gray_multi[gray > mean_val] = 3
gray_multi[(gray > 0.5 * 255) & (gray <= mean_val)] = 2
gray_multi[(gray > 0.25 * 255) & (gray <= 0.5 * 255)] = 1

plt.imshow(gray_multi, cmap='gray')
plt.show()

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
#from PIL import Image

# Set working directory
os.chdir("F:\\_VScode1\\CV_2_Object Segmentation\\threshold")
#print("Current working directory:", os.getcwd())

# Use an existing image (update this if needed)
img_path = r'F:\_VScode1\CV_2_Object Segmentation\image2.jpg'  # Change if necessary

# Load the image
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    print("Check if the path is correct and the file exists.")
    exit()

# Proceed with image processing
b, g, r = cv2.split(img)  
rgb_img = cv2.merge([r, g, b])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# THRESH_OTSU automatically computes the optimal threshold
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)
# Finding sure background area
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations = 2)
sure_bg = cv2.dilate(closing,kernel,iterations=3) #dilating 3 iterations, adds pixels to boundaries, expanding bright regions (foreground)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
# Dealing with unknown areas (marking the pixels to 0)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.subplot(211),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(thresh, 'gray')
plt.imsave(r'thresh.png',thresh)
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

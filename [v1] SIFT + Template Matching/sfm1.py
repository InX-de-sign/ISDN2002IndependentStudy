import cv2
import matplotlib.pyplot as plt

import os
os.chdir("F:\\_VScode1\\CV_1_SFM")

# Step 1: Load the images
image1 = cv2.imread('query_image.jpg', cv2.IMREAD_GRAYSCALE)  # Query image
image2 = cv2.imread('template_image.jpg', cv2.IMREAD_GRAYSCALE)  # Train image


if image1 is None:
    print("Error loading query_image.jpg")
if image2 is None:
    print("Error loading template_image.jpg")

# Step 2: Initialize the SIFT detector
sift = cv2.SIFT_create()

# Step 3: Detect SIFT keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Step 4: Use the Brute-Force Matcher to find matches
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Step 5: Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Step 6: Draw the top N matches
N = 50  # Number of matches to display
matched_image = cv2.drawMatches(
    image1, keypoints1, image2, keypoints2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Step 7: Display the result
plt.figure(figsize=(20, 10))
plt.imshow(matched_image, cmap='gray')
plt.title(f'Top {N} SIFT Matches')
plt.axis('off')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the images
query_image = cv2.imread('query_image1.jpg', cv2.IMREAD_GRAYSCALE)  # Larger image
template_image = cv2.imread('template_image2.jpg', cv2.IMREAD_GRAYSCALE)  # Smaller image (template)

# Ensure the images are loaded correctly
if query_image is None or template_image is None:
    raise ValueError("Could not load images. Check the file paths.")

# Step 2: Initialize the SIFT detector
sift = cv2.SIFT_create()

# Step 3: Detect SIFT keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(query_image, None)
keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)

# Step 4: Perform Dense Matching using SIFT Descriptors
# Create a similarity map
similarity_map = np.zeros((query_image.shape[0] - template_image.shape[0] + 1,
                          (query_image.shape[1] - template_image.shape[1] + 1)))

# Slide the template over the query image
for i in range(similarity_map.shape[0]):
    for j in range(similarity_map.shape[1]):
        # Extract the region of interest (ROI) from the query image
        roi = query_image[i:i + template_image.shape[0], j:j + template_image.shape[1]]
        
        # Compute SIFT descriptors for the ROI
        roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
        
        if roi_descriptors is not None and descriptors2 is not None:
            # Use a Brute-Force Matcher to compute the similarity between descriptors
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(roi_descriptors, descriptors2)
            
            # Compute the average distance of the matches
            avg_distance = np.mean([m.distance for m in matches])
            
            # Store the inverse of the average distance as the similarity score
            similarity_map[i, j] = 1 / (avg_distance + 1e-6)  # Add a small epsilon to avoid division by zero

# Step 5: Find the Best Match Location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity_map)
top_left = max_loc  # Best match location
bottom_right = (top_left[0] + template_image.shape[1], top_left[1] + template_image.shape[0])

# Step 6: Draw a Rectangle Around the Matched Region
template_matched_image = query_image.copy()
cv2.rectangle(template_matched_image, top_left, bottom_right, 255, 2)

# Step 7: Display the Results
plt.figure(figsize=(20, 10))

# Display the query image with the matched region
plt.subplot(1, 2, 1)
plt.imshow(template_matched_image, cmap='gray')
plt.title('Template Matching with SIFT Descriptors')
plt.axis('off')

# Display the similarity map
plt.subplot(1, 2, 2)
plt.imshow(similarity_map, cmap='jet')
plt.title('Similarity Map')
plt.axis('off')

plt.tight_layout()
plt.show()

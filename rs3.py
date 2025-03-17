# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:08:45 2024

@author: jonas
"""

import os #to navigate directories
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% Points location
angled_points = np.array([[1264, 558], [383, 178], [990, 219], [590, 129]], dtype=np.float32)
topdown_points = np.array([[791, 18], [1087, 951], [303, 606], [721, 1127]], dtype=np.float32)

# Compute the homography matrix using the points
homography_matrix, status = cv2.findHomography(angled_points, topdown_points)


# %% Image transformation
patches_area = []
dates = []

# Load the mask (replace with your actual mask image)
mask_path = "C:/Users/jonas/Documents/uni/TM/RS/img/mask1.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

path = "C:/Users/jonas/Documents/uni/TM/RS/img/asdaf/"
for image_filename in os.listdir(path):
    
    image_path = os.path.join(path, image_filename)
    image = cv2.imread(image_path)
    
    # Extract date from filename (format: Cam1-DD-MM-HH-MM)
    date_part = image_filename.split('-')[1:3]  # Extract ['07', '04']
    date_text = f"{date_part[1]}.{date_part[0]}"  # Format as '04.07'
    dates.append(date_text)
    
    # Apply the mask to the original image
    isolated_lake = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the result
    result_path = os.path.abspath(os.path.join(path, "../results/lake_isolated.jpg"))
    cv2.imwrite(result_path, isolated_lake)
    cv2.imshow("Isolated lake", isolated_lake) 
    
    # Apply Perspective Transform Algorithm
    Lake_warped = cv2.warpPerspective(isolated_lake, homography_matrix, (1280, 1280))
    result_path = os.path.abspath(os.path.join(path, "../results/lake_warped.jpg"))
    cv2.imwrite(result_path, Lake_warped)
    cv2.imshow("Warped lake", Lake_warped)
    
    # Splitting channels
    
    blue, green, red = cv2.split(Lake_warped)
    np.seterr(divide='ignore', invalid='ignore')
    blue[np.isinf(blue)] = np.max(blue[np.isfinite(blue)])
    blue[np.isinf(green)] = np.max(green[np.isfinite(green)])
    GBratio = np.where(blue > 0, green / blue, 0) #avoids nan and inf
    #cv2.imshow('Green/blue ratio', GBratio)
    
    # Blooming areas
    
    # GBratio histogram
    threshold_value = np.percentile(GBratio, 98) #threshold at xth percentile
    hist, bins = np.histogram(GBratio, bins=100, range=(0.01, np.max(GBratio)))
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist, width=0.1, color='b', alpha=0.7)
    plt.vlines(threshold_value, min(hist), max(hist), color='black', linestyles="--")
    plt.xlabel("Green/Blue Ratio")
    plt.ylabel("Frequency")
    plt.title("Histogram of Green/Blue Ratio with 95th percentile threshold")
    plt.grid(True)
    plt.show()
    
    #Map blooming areas
    green_mask = (GBratio > threshold_value).astype(np.uint8) * 255 #white areas on black
    green_areas = cv2.bitwise_and(Lake_warped, Lake_warped, mask=green_mask) #blooming areas in the picture
    #cv2.imshow("Blooming areas", green_mask)
    cv2.imwrite(result_path, green_mask )
    
    # Compute areas and number of patches
    _, binary_mask = cv2.threshold(green_mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_bloom_area = 0
    num_patches = len(contours)
    patch_areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        total_bloom_area += area
        #patch_areas.append(area)
    
    # converting from px to m2
    Lake_grey = cv2.cvtColor(Lake_warped, cv2.COLOR_BGR2GRAY) 
    px_lake = np.count_nonzero(Lake_grey)
    lake_area = 98490 #m2, measured with geoportail
    px_area = lake_area/px_lake
    
    total_bloom_area = total_bloom_area*px_area
    #patch_areas = np.array(patch_areas)*px_area
    patches_area.append(total_bloom_area)
    #print(f"Total white area: {total_bloom_area} m2")
    #print(f"Ratio: {100*total_bloom_area/lake_area:.2f}%")
    #print(f"Number of distinct patches: {num_patches}")
    #print("Area of each patch [m2]:")
    #for area in patch_areas:
    #    print(f"{area:.2f}")
    
patches_ratio = [100 * area / lake_area for area in patches_area]

#%% Plotting
plt.figure(figsize=(10, 6))
plt.plot(dates, patches_ratio, marker='o', linestyle='-', color='b')
plt.title('Green algae patches ratio over time')
plt.xlabel('Date')
plt.ylabel('Patches ratio (%)')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


#%% Points visualisation
# Load the first or last image (change 'first' to 'last' if needed)
image_filenames = sorted(os.listdir(path))  # Sort to ensure order
selected_image_filename = image_filenames[0]  # Change to [-1] for the last image

# Load the selected image
selected_image_path = os.path.join(path, selected_image_filename)
selected_image = cv2.imread(selected_image_path)

# Define font settings for the labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
text_color = (0, 255, 255)  # Yellow color

# Draw red dots and GPS coordinate labels
for i, (point, gps) in enumerate(zip(angled_points, topdown_points)):
    x, y = int(point[0]), int(point[1])
    cv2.circle(selected_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot
    
    # Format GPS coordinates as text
    gps_label = f"({int(gps[0])}, {int(gps[1])})"
    
    # Position the text to the left of the dot
    text_position = (x - 100, y-15)  # Adjust -100 to move it further left if needed
    
    # Add the text to the image
    cv2.putText(selected_image, gps_label, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Save and display the result
output_path = os.path.abspath(os.path.join(path, "../results/image_with_dots.jpg"))
cv2.imwrite(output_path, selected_image)
cv2.imshow("Image with Red Dots and Labels", selected_image)

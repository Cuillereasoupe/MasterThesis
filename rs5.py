# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:08:45 2024

@author: jonas
"""

import os #to navigate directories
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy.spatial.distance import cdist
from skimage.exposure import match_histograms

# %% Points location
angled_points = np.array([[1264, 558], [383, 178], [990, 219], [590, 129]], dtype=np.float32)
topdown_points = np.array([[791, 18], [1087, 951], [303, 606], [721, 1127]], dtype=np.float32)

# Compute the homography matrix using the points
homography_matrix, status = cv2.findHomography(angled_points, topdown_points)

# %% Colour informations

def load_algae_data(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found. Verify the file path.")
        return {}

algae_locations = load_algae_data("algae-locations.json")

def preprocess_color_data(algae_data):
    """Convert JSON data to a structured format with color and spatial information"""
    color_data = {}
    for filename, points_list in algae_data.items():
        # Extract date from filename (Cam1-DD-MM-HH-MM.png)
        date_part = filename.split('-')[1:3]  # Get [DD, MM]
        date_key = f"{date_part[1]}.{date_part[0]}"  # Format as MM.DD
        
        color_data[date_key] = {
            'colors': [],
            'coords': [],
            'categories': []
        }
        
        for point in points_list:
            color_data[date_key]['colors'].append(point['color'])
            color_data[date_key]['coords'].append([point['x'], point['y']])
            color_data[date_key]['categories'].append(point['type'])
    
    # Convert lists to numpy arrays for efficient computation
    for date in color_data:
        color_data[date]['colors'] = np.array(color_data[date]['colors'])
        color_data[date]['coords'] = np.array(color_data[date]['coords'])
    
    return color_data

color_data = preprocess_color_data(algae_locations)

def classify_pixels(image, color_data_entry, spatial_weight=0.5, color_weight=0.5):
    """
    Classify pixels using combined color and spatial distance
    Args:
        image: 3-channel BGR image (H,W,3)
        color_data_entry: Preprocessed color data for the closest date
        spatial_weight: Importance of spatial proximity (0-1)
        color_weight: Importance of color proximity (0-1)
    """
    # Get reference data
    ref_colors = color_data_entry['colors']
    ref_coords = color_data_entry['coords']
    categories = color_data_entry['categories']
    category_codes = {"middle": 1, "algae": 2, "non-algae": 3}
    
    # Prepare image data
    h, w = image.shape[:2]
    flat_image = image.reshape(-1, 3)
    pixel_coords = np.indices((h, w)).reshape(2, -1).T
    
    # Normalize coordinates to 0-1 range for distance calculation
    norm_coords = ref_coords / np.array([w, h])
    norm_pixel_coords = pixel_coords / np.array([w, h])
    
    # Calculate color distances (Euclidean in BGR space)
    color_dists = cdist(flat_image, ref_colors, metric='euclidean')
    
    # Calculate spatial distances (Euclidean)
    spatial_dists = cdist(norm_pixel_coords, norm_coords, metric='euclidean')
    
    # Combine distances with weights
    combined_dists = (color_weight * color_dists) + (spatial_weight * spatial_dists)
    
    # Find nearest neighbor for each pixel
    nearest_idx = np.argmin(combined_dists, axis=1)
    
    # Create classification map
    classification = np.zeros(h*w, dtype=np.uint8)
    for i, idx in enumerate(nearest_idx):
        classification[i] = category_codes.get(categories[idx], 0)
    
    return classification.reshape(h, w)

def find_closest_date(target_date_str, date_keys):
    target = datetime.strptime(target_date_str, "%d.%m")
    min_diff = float('inf')
    closest_date = None
    
    for date_str in date_keys:
        date = datetime.strptime(date_str, "%d.%m")
        diff = abs((date - target).days)
        if diff < min_diff:
            min_diff, closest_date = diff, date_str
            
    return closest_date

# Lighting calibration: Use this with a well-lit image
def normalize_lighting(source_img, reference_img): 
    # Convert images to RGB if they are BGR
    source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

    matched_rgb = match_histograms(source_rgb, reference_rgb, channel_axis=-1)
    return cv2.cvtColor(np.uint8(matched_rgb), cv2.COLOR_RGB2BGR)

# %% Image transformation
patches_area = []
dates = []

# Load the mask (replace with your actual mask image)
mask_path = "C:/Users/jonas/Documents/uni/TM/RS/img/mask1.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

path = "C:/Users/jonas/Documents/uni/TM/RS/img/final-img/"
for image_filename in os.listdir(path):
    
    image_path = os.path.join(path, image_filename)
    image = cv2.imread(image_path)
    
    # Extract date from filename (format: Cam1-DD-MM-HH-MM)
    date_part = image_filename.split('-')[1:3]  # Extract [MM, DD]
    date_text = f"{date_part[1]}.{date_part[0]}"  # Format as 'DD.MM'
    dates.append(date_text)
    
    # Apply the mask to the original image
    isolated_lake = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the result
    # result_path = os.path.abspath(os.path.join(path, "../results/lake_isolated.jpg"))
    # cv2.imwrite(result_path, isolated_lake)
    # cv2.imshow("Isolated lake", isolated_lake) 
    
    # Apply Perspective Transform Algorithm
    Lake_warped = cv2.warpPerspective(isolated_lake, homography_matrix, (1280, 1280))
    # result_path = os.path.abspath(os.path.join(path, "../results/lake_warped.jpg"))
    # cv2.imwrite(result_path, Lake_warped)
    # cv2.imshow("Warped lake", Lake_warped)
    
    # Splitting channels
    
    blue, green, red = cv2.split(Lake_warped)
    np.seterr(divide='ignore', invalid='ignore')
    blue[np.isinf(blue)] = np.max(blue[np.isfinite(blue)])
    blue[np.isinf(green)] = np.max(green[np.isfinite(green)])
    GBratio = np.where(blue > 0, green / blue, 0) #avoids nan and inf
    # cv2.imshow('Green/blue ratio', GBratio)
    
    # Blooming areas
    
    # Extract date as MM.DD
    date_part = image_filename.split('-')[1:3]
    date_text = f"{date_part[1]}.{date_part[0]}"
    
    # Find closest date
    closest_date = find_closest_date(date_text, color_data.keys())
    current_data = color_data[closest_date]
    
    # Classify pixels
    classification_map = classify_pixels(Lake_warped, current_data)
    
    # Apply non-black mask
    warped_gray = cv2.cvtColor(Lake_warped, cv2.COLOR_BGR2GRAY)
    final_classification = np.where(warped_gray > 0, classification_map, 0)
    
    # Visualize
    plt.imshow(final_classification, cmap='viridis', vmin=0, vmax=3)
    plt.title(f"Classification {date_text} (closest: {closest_date})")
    plt.colorbar(ticks=[0,1,2,3], label='0: Unclassified\n1: Middle\n2: Algae\n3: Non-algae')
    plt.show()
 


#%% Points visualisation
# Load the first or last image (change 'first' to 'last' if needed)
image_filenames = sorted(os.listdir(path))  # Sort to ensure order
selected_image_filename = image_filenames[0]  # -1 for the last image

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
# output_path = os.path.abspath(os.path.join(path, "../results/image_with_dots.jpg"))
# cv2.imwrite(output_path, selected_image)
# cv2.imshow("Image with Red Dots and Labels", selected_image)

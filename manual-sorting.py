# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:34:15 2025

@author: jonas
"""

import cv2
import os
import shutil
import ctypes

# Paths
source_folder = 'C:/Users/jonas/Documents/uni/TM/RS/img/hourly'
destination_folder = 'C:/Users/jonas/Documents/uni/TM/RS/img/final-img'

# Make sure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get sorted list of image files
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
image_files.sort()

#Introduction
ctypes.windll.user32.MessageBoxW(0, "Browse the images (N and M keys) and select (Y key) the good ones based on the following criterias.\n - Entire lake is visible\n - No shades on the lake\n - No reflections on the lake's surface\n - The water isn't too milky", "Info", 0)

window_name = "Picture selection (Y=keep, N=discard, ESC=quit, N=prev, M=next, Z=Undo)"

index = 0
while index < len(image_files):
    image_file = image_files[index]
    image_path = os.path.join(source_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_file}, skipping.")
        index += 1
        continue

    cv2.imshow(window_name, image)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('y'):
        shutil.copy(image_path, os.path.join(destination_folder, image_file))
        ctypes.windll.user32.MessageBoxW(0, f"Copied {image_file} to {destination_folder}", "Info", 0)
        index += 1
    elif key == 109:  # Right arrow key (→)
        if index < len(image_files)-1:
            index += 1
        else:
            ctypes.windll.user32.MessageBoxW(0, "All images reviewed.", "Info", 0)
    elif key == 110:  # Left arrow key (←)
        if index > 0:
            index -= 1
    elif key == 27:  # Esc key
        print("Aborted by user.")
        break

cv2.destroyAllWindows()
print("Done processing images.")
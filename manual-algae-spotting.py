# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:27:40 2025

@author: jonas
"""
import cv2
import os
import numpy as np
from sys import exit
import json
import ctypes

# Path to images
folder = 'C:/Users/jonas/Documents/uni/TM/RS/tests-spring2025/afidus/img/'
pict_qty = 3 #Amount of pictures

# Get sorted list of image files
image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
image_files.sort()
indices = np.linspace(0, len(image_files) - 1, pict_qty, dtype=int)
image_subset = [image_files[i] for i in indices]

#Introduction
ctypes.windll.user32.MessageBoxW(0, "Browse this subset of image. On each one, place markers for:\n - Colour of the middle of the lake (D key)\n - Algae spots (A key)\n - Lake areas were we can see the bottom but it's not covered by algae (S key)", "Info", 0)

window_name = "Manual colour calibration (A=algae, S=non-algae, D=middle of the lake, ESC=quit, N=prev, M=next, Z=Undo)"

# Zoom/pan state and click storage
zoom_factor = 1.0
pan_x, pan_y = 0, 0
dragging = False
start_x, start_y = 0, 0
click_data = {}  # Format: {filename: [{'type': 'algae/non-algae', 'x': x, 'y': y, 'color': (B,G,R)}, ...]}
current_image_file = ""
mouseX = 0
mouseY = 0
screen_width, screen_height = 1920, 1080  # Your actual screen resolution
display_scale = 1.0  # Initial scaling factor for display
original_img = None  # Store the original image


def load_image(image_path):
    global current_img, original_img, display_scale
    
    # Load the original high-resolution image
    original_img = cv2.imread(image_path)
    if original_img is None:
        return False
    
    # Calculate initial scaling to fit screen
    h, w = original_img.shape[:2]
    width_scale = screen_width / w
    height_scale = screen_height / h
    display_scale = min(width_scale, height_scale, 1.0)  # Don't upscale small images
    
    # Create a scaled version for display
    if display_scale < 1.0:
        current_img = cv2.resize(original_img, 
                                (int(w * display_scale), int(h * display_scale)), 
                                interpolation=cv2.INTER_AREA)
    else:
        current_img = original_img.copy()
    
    return True

def update_display():
    global window_name, current_img, original_img, display_scale
    
    # Create a clean copy of the display image
    display_img = current_img.copy()
    
    # Apply zoom and pan
    display_img = zoom_pan_image(display_img)
    
    # Draw the mini-map
    if display_scale < 1.0:
        mini_map_size = 200  # Size of the mini-map in pixels
        h, w = original_img.shape[:2]
        mini_scale = min(mini_map_size / w, mini_map_size / h)
        mini_w, mini_h = int(w * mini_scale), int(h * mini_scale)
        
        # Create the mini-map
        mini_map = cv2.resize(original_img, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
        
        # Calculate the viewport rectangle
        viewport_x = int(pan_x * mini_scale / display_scale)
        viewport_y = int(pan_y * mini_scale / display_scale)
        viewport_w = int(display_img.shape[1] * mini_scale / zoom_factor / display_scale)
        viewport_h = int(display_img.shape[0] * mini_scale / zoom_factor / display_scale)
        
        # Draw the viewport rectangle on the mini-map
        cv2.rectangle(mini_map, 
                     (max(0, viewport_x), max(0, viewport_y)), 
                     (min(mini_w, viewport_x + viewport_w), min(mini_h, viewport_y + viewport_h)), 
                     (0, 255, 255), 2)
        
        # Add the mini-map to the corner of the display image
        padding_w = 10
        padding_h = 75
        display_img[padding_h:padding_h+mini_h, padding_w:padding_w+mini_w] = mini_map
    
    # Draw markers
    if current_image_file in click_data:
        h, w = original_img.shape[:2]
        dh, dw = display_img.shape[:2]
        
        for marker in click_data[current_image_file]:
            # Convert original coordinates to display coordinates
            ox, oy = marker['x'], marker['y']
            
            # Scale to display image
            dx = int(ox * display_scale)
            dy = int(oy * display_scale)
            
            # Adjust for zoom and pan
            new_w = dw / zoom_factor
            new_h = dh / zoom_factor
            x1 = int((dw - new_w)/2 + pan_x)
            y1 = int((dh - new_h)/2 + pan_y)
            
            if x1 <= dx < x1 + new_w and y1 <= dy < y1 + new_h:
                screen_x = int((dx - x1) * zoom_factor)
                screen_y = int((dy - y1) * zoom_factor)
                
                if marker['type'] == 'algae':
                    color = (0, 255, 0)
                elif marker['type'] == 'non-algae':
                    color = (0, 0, 255)
                elif marker['type'] == 'middle':
                    color = (255, 0, 0)
                cv2.circle(display_img, (screen_x, screen_y), 8, color, -1)
    
    legend_y = 210
    cv2.circle(display_img, (20, legend_y), 8, (0, 255, 0), -1)
    cv2.putText(display_img, "Algae (A)", (35, legend_y+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    legend_y += 25
    cv2.circle(display_img, (20, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(display_img, "Non-algae (S)", (35, legend_y+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    legend_y += 25
    cv2.circle(display_img, (20, legend_y), 8, (255, 0, 0), -1)
    cv2.putText(display_img, "Middle (D)", (35, legend_y+5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow(window_name, display_img)

def new_pin(type, x, y):
    global current_img, original_img, current_image_file, display_scale
    
    # Convert screen coordinates back to original image coordinates
    h, w = current_img.shape[:2]
    original_h, original_w = original_img.shape[:2]
    
    new_w = w / zoom_factor
    new_h = h / zoom_factor
    x1 = int((w - new_w)/2 + pan_x)
    y1 = int((h - new_h)/2 + pan_y)
    
    # Convert clicked position to display image coordinates
    display_x = int(x1 + x / zoom_factor)
    display_y = int(y1 + y / zoom_factor)
    
    # Convert display coordinates to original image coordinates
    original_x = int(display_x / display_scale)
    original_y = int(display_y / display_scale)
    
    # Clamp to image dimensions
    original_x = max(0, min(original_w-1, original_x))
    original_y = max(0, min(original_h-1, original_y))
    
    # Get pixel color from original image
    color = tuple(map(int, original_img[original_y, original_x]))
    
    # Create marker data
    marker_data = {
        'image': current_image_file,
        'type': type,
        'x': original_x,
        'y': original_y,
        'color': color
    }
    
    # Store coordinates
    if current_image_file not in click_data:
        click_data[current_image_file] = []
    click_data[current_image_file].append(marker_data)
    print(f"Stored {type} at ({original_x}, {original_y}) with color {color}")
    update_display()

def zoom_pan_image(img):
    global zoom_factor, pan_x, pan_y
    h, w = img.shape[:2]
    
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    
    # Clamp pan values
    max_pan_x = max(0, (w - new_w) // 2)
    max_pan_y = max(0, (h - new_h) // 2)
    pan_x = min(max_pan_x, max(-max_pan_x, pan_x))
    pan_y = min(max_pan_y, max(-max_pan_y, pan_y))
    
    x1 = max(0, int((w - new_w)/2 + pan_x))
    y1 = max(0, int((h - new_h)/2 + pan_y))
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def reset_image():
    global zoom_factor, pan_x, pan_y, current_image_file, click_data
    zoom_factor = 1.0
    pan_x, pan_y = 0, 0
    if current_image_file not in click_data:
        click_data[current_image_file] = []

def mouse_callback(event, x, y, flags, param):
    global pan_x, pan_y, dragging, start_x, start_y, zoom_factor, current_img, mouseX, mouseY
    
    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_speed = 0.1
        if flags > 0:
            zoom_factor = max(1, zoom_factor - zoom_speed)
        else:
            zoom_factor = min(3.0, zoom_factor + zoom_speed)
        update_display()
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - start_x
        dy = y - start_y
        pan_x += int(dx * zoom_factor)
        pan_y += int(dy * zoom_factor)
        start_x, start_y = x, y
        update_display()
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
    
    mouseX = x
    mouseY = y



# Main loop
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)
index = 0

while index < len(image_subset):
    current_image_file = image_subset[index]
    image_path = os.path.join(folder, current_image_file)
    
    if not load_image(image_path):
        print(f"Failed to load {current_image_file}, skipping.")
        index += 1
        continue
    
    reset_image()
    update_display()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('m'):  # Next
            if index < pict_qty-1:
                index += 1
                reset_image()
            break
        elif key == ord('n'):  # Previous
            if index > 0:
                index -= 1
                reset_image()
            break
        elif key == ord('z'):  # Undo
            if bool(click_data[current_image_file]):
                click_data[current_image_file].pop()
            break
        elif key == ord('a'):
            new_pin('algae', mouseX, mouseY)
            break
        elif key == ord('s'):
            new_pin('non-algae', mouseX, mouseY)
            break
        elif key == ord('d'):
            new_pin('middle', mouseX, mouseY)
            break
        elif key == 27:  # Esc
            print("\nStored coordinates:")
            for img_file, markers in click_data.items():
                print(f"\n{img_file}:")
                for i, marker in enumerate(markers, 1):
                    print(f"  Marker {i}:")
                    print(f"    Type: {marker['type']}")
                    print(f"    Position: X={marker['x']}, Y={marker['y']}")
                    print(f"    Color (BGR): {marker['color']}")
            cv2.destroyAllWindows()
            with open("algae-locations.json", "w") as f:
                json.dump(click_data, f)
            exit()
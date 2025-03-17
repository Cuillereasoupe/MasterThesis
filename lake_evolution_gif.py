import os
import imageio
import cv2

# Directory containing the images
results_folder = "C:/Users/jonas/Documents/uni/TM/img/results/"

# List to hold images
images = []

# Define font and position for the date text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White text
font_thickness = 2
margin = 10  # Margin from the bottom-right corner

# Loop through all the files in the directory
for file_name in sorted(os.listdir(results_folder)):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        file_path = os.path.join(results_folder, file_name)
        
        # Extract date from filename (format: Cam1-DD-MM-HH-MM)
        date_part = file_name.split('-')[1:3]  # Extract ['07', '04']
        date_text = f"{date_part[1]}.{date_part[0]}.2024"  # Format as '04.07'

        # Read the image
        image = cv2.imread(file_path)
        
        # Get the size of the image
        height, width = image.shape[:2]

        # Calculate text size and position
        text_size = cv2.getTextSize(date_text, font, font_scale, font_thickness)[0]
        text_x = width - text_size[0] - margin
        text_y = height - margin

        # Add text to the image
        cv2.putText(image, date_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Convert from BGR to RGB (imageio expects RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Append the image to the list
        images.append(image_rgb)

# Save as GIF
gif_path = os.path.abspath(os.path.join(results_folder, "lake_evolution.gif"))
imageio.mimsave(gif_path, images, duration=5)  # duration in seconds between frames


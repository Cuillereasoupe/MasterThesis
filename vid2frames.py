# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:43:07 2024

@author: jonas
"""

import cv2
import os
from datetime import datetime, timedelta

def save_all_frames(video_path, dir_path, start_time, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
     
    frame_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    while True:
        ret, frame = cap.read()
        timestamp = frame_time.strftime('Cam2-%m-%d-%H-%M-%S')
        if ret:
            cv2.imwrite(os.path.join(dir_path, f'{timestamp}.{ext}'), frame)
            # if(frame_time.strftime("%H") == "23"):
            #     frame_time = frame_time + timedelta(days=1)
            #     frame_time = frame_time.replace(hour=0)
            # else:
            frame_time = frame_time + timedelta(seconds=2)
        else:
            return

# Corrected function call with a valid start_time
save_all_frames('C:/Users/jonas/Documents/uni/TM/RS/tests-spring2025/Brinno/TLC00000.AVI', 
                'C:/Users/jonas/Documents/uni/TM/RS/tests-spring2025/Brinno/images', 
                '2025-03-05 15:43:08')


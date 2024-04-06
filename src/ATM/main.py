import cv2
import datetime
from ultralytics import YOLO
import logging
import os
from utils import *
from atm_cleanliness import *
from atm_suspecious import *
from atm_functionality import *
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()

logger = logging.getLogger(__name__)

person_presence_start_time = None
elapsed_time = None 

VIDEO_NAME = "videos/atm_func.mp4"
ATM_MODEL = os.getenv('ATM_MODEL')
VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

atm_model = load_model(ATM_MODEL)
cap = load_video(VIDEO_PATH)

# Get current FPS
fps = get_fps(cap)
print("Current FPS:", fps)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = process_frame(atm_model, frame, conf=0.8)[0]

    # wait_thresould is in sec
    persons, elapsed_time, all_sus_flags, person_presence_start_time = atm_suspecious(results,person_presence_start_time, elapsed_time, wait_threshould = 2) 
    atm_statuse, atm_trash_count, mess_level = atm_cleanliness(frame, results)
    

    atm_functions = get_atm_functions(results)

    atm_overly(frame, atm_statuse, atm_trash_count, mess_level, atm_functions, elapsed_time, person_presence_start_time ,persons
               ,all_sus_flags['time_exceeded_flag'], all_sus_flags['num_persons_flag'], all_sus_flags['helmet_detected_flag'])

    r = results.plot()

    cv2.imshow('ATM', r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
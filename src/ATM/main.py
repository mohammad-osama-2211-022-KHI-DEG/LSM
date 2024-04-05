import cv2
import datetime
from ultralytics import YOLO
import logging
import os
from utils import *
from src.ATM.atm_cleanliness import *
from atm_suspecious import *
from atm_functionality import *
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()

logger = logging.getLogger(__name__)

person_presence = None
elapsed_time = None 

ATM_MODEL = os.getenv('ATM_MODEL')
VIDEO_PATH = os.getenv('VIDEO_PATH')

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

    atm_functions = get_atm_functions(results)
    #atm_transaction_status = atm_transaction(atm_functions['res'], atm_functions['class_name'])
    sus_activity = suspecious_cases(results)
    person_presence, elapsed_time = check_person_duration(sus_activity['num_persons'], person_presence, elapsed_time)
    atm_trash_count = trash_count(frame, results)
    mess_level = calculate_mess_level(atm_trash_count)
    atm_statuses = atm_cleanliness_status(atm_trash_count, mess_level)

    atm_overly(frame, atm_statuses, atm_trash_count, mess_level, atm_functions, elapsed_time, person_presence, persons = sus_activity['num_persons'], helmet = sus_activity['helmet_detected'])

    r = results.plot()

    cv2.imshow('ATM', r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
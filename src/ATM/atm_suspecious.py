import cv2
from ultralytics import YOLO
import logging
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from utils import *

def suspecious_cases(results):
    helmet_detected = False
    class_ids = results.boxes.cls.numpy()
    num_persons = (class_ids == 7).sum() # 7 is person id
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results.names[int(class_id)].lower() == 'helmet':
            helmet_detected = True
    return {"helmet_detected": helmet_detected, "num_persons": num_persons}


def check_person_duration(num_persons, person_presence, elapsed_time):
    start_time = None
    if num_persons != 0 and person_presence is None:
        start_time = datetime.now()
        return start_time, None
          
    elif num_persons == 0 and person_presence is not None:
        elapsed_time = datetime.now() - person_presence
        return None, elapsed_time
    
    else:
        return person_presence, elapsed_time


def main():
    load_dotenv()

    person_presence = None
    elapsed_time = None 

    VIDEO_NAME = "videos/suspecious.mp4"
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

    atm_model = load_model(ATM_MODEL)

    cap = load_video(VIDEO_PATH)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = process_frame(atm_model, frame, conf=0.8)[0]
        sus_activity = suspecious_cases(results)
        person_presence, elapsed_time= check_person_duration(sus_activity['num_persons'], person_presence, elapsed_time)
        if elapsed_time is not None:
            cv2.putText(frame, f"Presence Time: {elapsed_time.total_seconds()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        cv2.putText(frame, f"Helmet: {sus_activity['helmet_detected']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(frame, f"No of Persons: {sus_activity['num_persons']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Start Time: {person_presence}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        

        r = results.plot()

        cv2.imshow('atm', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
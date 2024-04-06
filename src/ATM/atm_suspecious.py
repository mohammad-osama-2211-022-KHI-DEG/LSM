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
    num_persons_flag = False
    class_ids = results.boxes.cls.numpy()
    num_persons = (class_ids == 7).sum() # 7 is person id
    if num_persons > 2:
        num_persons_flag = True
    else:
        num_persons_flag = False
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results.names[int(class_id)].lower() == 'helmet':
            helmet_detected = True

    return {"helmet_detected": helmet_detected, "num_persons": num_persons, "num_persons_flag": num_persons_flag}


def check_person_duration(num_persons, person_presence_start_time, elapsed_time):
    start_time = None
    if num_persons != 0 and person_presence_start_time is None:
        start_time = datetime.now()
        return start_time, None
          
    elif num_persons == 0 and person_presence_start_time is not None:
        elapsed_time = datetime.now() - person_presence_start_time
        return None, elapsed_time
    
    else:
        return person_presence_start_time, elapsed_time
    
def presence_threshold_flag(person_presence, wait_threshould, num_persons):
    flag = False
    if num_persons != 0 and person_presence is not None:
        wait_time = datetime.now() - person_presence
        if wait_time.total_seconds() >= wait_threshould:
            flag = True
            return flag
    elif num_persons == 0 and person_presence is None:
        flag = False    
        return flag 
    else:
        return flag

def atm_suspecious(results, person_presence_start_time, elapsed_time, wait_threshould): # wait_threshold is in sec
    time_exceeded_flag = False
    sus_activity = suspecious_cases(results)
    person_presence_start_time, elapsed_time = check_person_duration(sus_activity['num_persons'], person_presence_start_time, elapsed_time)
    time_exceeded_flag = presence_threshold_flag(person_presence_start_time, wait_threshould, num_persons=sus_activity['num_persons']) 
    persons = sus_activity['num_persons']
    flags = {'num_persons_flag': sus_activity['num_persons_flag'],
             'time_exceeded_flag': time_exceeded_flag,
             'helmet_detected_flag': sus_activity['helmet_detected']}
    
    suspicious = any(flags.values())    

    suspicious_label = 'SUSPICIOUS' if suspicious else 'NORMAL'
    
    return persons, elapsed_time, flags, person_presence_start_time, suspicious_label

def post_sus_data(suspecious, previous_suspecious):
    status = int()
    if suspecious == 'SUSPICIOUS' and previous_suspecious == 'NORMAL':
        data = {
            'status': suspecious,
            'timestamp': formatted_timestamp
        }
        #send_data_to_endpoint(data, SUSPECIOUS_TARGET_URL, JWT_TOKEN)
        print("sent suspecious data")
        status = 200

    return status


def main():
    load_dotenv()

    person_presence_start_time = None
    elapsed_time = None 
    suspecious = False
    previous_suspecious = None


    VIDEO_NAME = "videos/atm_func.mp4"
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

    atm_model = load_model(ATM_MODEL)

    cap = load_video(VIDEO_PATH)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = process_frame(atm_model, frame, conf=0.8)[0]
        persons, elapsed_time, all_sus_flags, person_presence_start_time, suspecious= atm_suspecious(results,person_presence_start_time, elapsed_time, wait_threshould = 2) # wait_thresould is in sec
        post_sus_data(suspecious, previous_suspecious)
        previous_suspecious = suspecious

        if elapsed_time is not None:
            cv2.putText(frame, f"Presence elapsed time: {elapsed_time.total_seconds()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        cv2.putText(frame, f"Helmet: {all_sus_flags['helmet_detected_flag']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(frame, f"No of Persons: {persons}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"No of Persons Flag: {all_sus_flags['num_persons_flag']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Start Time: {person_presence_start_time}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Person Time Excedded: {all_sus_flags['time_exceeded_flag']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"suspecious: {suspecious}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


        

        r = results.plot()

        cv2.imshow('atm', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
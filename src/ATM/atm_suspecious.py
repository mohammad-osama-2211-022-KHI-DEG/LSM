import cv2
from ultralytics import YOLO
import logging
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Any, Dict, List
from utils import *
from router import *
import numpy as np
import requests

def suspecious_cases(results) -> dict[str: Any]:
    if results is None: raise TypeError("results can not be empty")
    helmet_detected: bool = False
    num_persons_flag: bool = False
    class_ids: np.ndarray = results.boxes.cls.numpy()
    num_persons: int = (class_ids == 7).sum()  # 7 is person id
    if num_persons > 2:
        num_persons_flag: bool = True
    for result in results.boxes.data.tolist():
        _, _, _, _, _, class_id = result
        if results.names[int(class_id)].lower() == 'helmet':
            helmet_detected: bool = True
    return {"helmet_detected": helmet_detected, "num_persons": num_persons, "num_persons_flag": num_persons_flag}


def check_person_duration(num_persons: int, person_presence_start_time: datetime, elapsed_time: timedelta) -> tuple[datetime, timedelta]:
    if num_persons < 0:
        raise ValueError("num_persons must be non-negative")
    if (type(num_persons) == int) is False:
        raise ValueError("num_persons must be integer")
    if not isinstance(person_presence_start_time, datetime):
        raise TypeError("person_presence_start_time must be a datetime object")
    if not isinstance(elapsed_time, timedelta):
        raise TypeError("elapsed_time must be a timedelta object")
    
    if num_persons != 0 and person_presence_start_time == datetime(1970, 1, 1, 0, 0, 0):
        start_time: datetime = datetime.now()
        return start_time, timedelta(0)
    elif num_persons == 0 and person_presence_start_time != datetime(1970, 1, 1, 0, 0, 0):
        elapsed_time = datetime.now() - person_presence_start_time
        return datetime(1970, 1, 1, 0, 0, 0), elapsed_time
    else:
        return person_presence_start_time, elapsed_time


def presence_threshold_flag(person_presence: datetime, wait_threshould: int, num_persons: int) -> bool:
    if not isinstance(person_presence, datetime):
        raise TypeError("person_presence must be a datetime object")
    if not isinstance(wait_threshould, int):
        raise TypeError("wait_threshould must be an integer")
    if num_persons < 0:
        raise ValueError("num_persons must be non-negative")
    if (type(num_persons) == int) is False:
        raise ValueError("num_persons must be integer")

    if num_persons != 0 and person_presence != datetime(1970, 1, 1, 0, 0, 0):
        wait_time: timedelta = datetime.now() - person_presence
        if wait_time.total_seconds() >= wait_threshould:
            return True
    elif num_persons == 0 and person_presence == datetime(1970, 1, 1, 0, 0, 0):
        return False
    else:
        return False

def atm_suspecious(results, person_presence_start_time: datetime, elapsed_time: timedelta, wait_threshould: int) -> tuple[int, timedelta, dict[str: bool], datetime, str]:
    if results is None:
        raise TypeError("results can not be empty")
    if not isinstance(person_presence_start_time, datetime):
        raise TypeError("person_presence_start_time must be a datetime object")
    if not isinstance(elapsed_time, timedelta):
        raise TypeError("elapsed_time must be a timedelta object")
    if (type(wait_threshould) == int) is False:
        raise TypeError("wait_threshould must be integer")
    
    time_exceeded_flag: bool = False
    sus_activity: dict = suspecious_cases(results)
    num_persons: int = int(sus_activity['num_persons'])
    person_presence_start_time, elapsed_time = check_person_duration(num_persons, person_presence_start_time, elapsed_time)
    time_exceeded_flag: bool = presence_threshold_flag(person_presence_start_time, wait_threshould, num_persons)
    persons: int = sus_activity['num_persons']
    flags: dict[str: bool] = {'num_persons_flag': sus_activity['num_persons_flag'],
                'time_exceeded_flag': time_exceeded_flag,
                'helmet_detected_flag': sus_activity['helmet_detected']}
    
    suspicious: bool = any(flags.values())   
    suspicious_label: int = 1 if suspicious else 0  # 1 for SUSPECIOUS 0 for NORMAL
    
    return persons, elapsed_time, flags, person_presence_start_time, suspicious_label


def post_sus_data(suspicious: int, previous_suspicious: int) -> int:
    if not isinstance(suspicious, int):
        raise TypeError("suspicious must be an integer")
    if not isinstance(previous_suspicious, int):
        raise TypeError("previous_suspicious must be an integer")
    status = 0
    if suspicious == 1 and previous_suspicious == 0:
        data = {
            'status': 'SUSPICIOUS' if suspicious == 1 else 'NORMAL',
            'timestamp': formatted_timestamp  
        }
        #status = send_data_to_endpoint(data, SUSPICIOUS_TARGET_URL, JWT_TOKEN)
        #print("sent suspicious data")
        status = 200

    return status

def main() -> None:
    try:
        load_dotenv()

        VIDEO_NAME = "videos/atm_func.mp4"
        ATM_MODEL = os.getenv('ATM_MODEL')
        VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

        person_presence_start_time: datetime = datetime(1970, 1, 1, 0, 0, 0)
        elapsed_time: timedelta = timedelta(0)
        suspicious: int = 0  # 1 for SUSPECIOUS 0 for NORMAL
        previous_suspicious: int = 0

        atm_model: YOLO = load_model(ATM_MODEL)
        cap: cv2.VideoCapture = load_video(VIDEO_PATH)

        while True:
            ret: bool
            frame: np.ndarray

            ret, frame = cap.read()
            if not ret:
                break

            results = process_frame(atm_model, frame, conf=0.8)[0]
            persons, elapsed_time, all_sus_flags, person_presence_start_time, suspicious = atm_suspecious(results, person_presence_start_time, elapsed_time, wait_threshould=2)
            post_sus_data(suspicious, previous_suspicious)
            previous_suspicious = suspicious

            if elapsed_time is not timedelta(0):
                cv2.putText(frame, f"Presence elapsed time: {elapsed_time.total_seconds()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            cv2.putText(frame, f"Helmet: {all_sus_flags['helmet_detected_flag']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            cv2.putText(frame, f"No of Persons: {persons}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.putText(frame, f"No of Persons Flag: {all_sus_flags['num_persons_flag']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.putText(frame, f"Start Time: {person_presence_start_time}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.putText(frame, f"Person Time Exceeded: {all_sus_flags['time_exceeded_flag']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.putText(frame, f"Suspicious: {suspicious}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            r = results.plot()
            cv2.imshow('atm', r)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == '__main__':
    main()

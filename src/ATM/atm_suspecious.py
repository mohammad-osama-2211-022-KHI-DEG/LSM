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

suspicious: int = 1
normal: int = 0

def atm_suspecious(results, person_presence_start_time: datetime, elapsed_time: timedelta, initial_time: datetime, wait_threshould: int) -> tuple[int, timedelta, dict[str: bool], datetime, str]:
    if results is None: raise ValueError("Results cannot be None")
    time_exceeded_flag: bool = False
    sus_activity: dict = suspecious_cases(results)
    num_persons: int = int(sus_activity['num_persons'])
    person_presence_start_time, elapsed_time = check_person_duration(num_persons, person_presence_start_time, elapsed_time, initial_time)
    time_exceeded_flag: bool = presence_threshold_flag(person_presence_start_time, wait_threshould, num_persons, initial_time)
    persons: int = sus_activity['num_persons']
    flags: dict[str: bool] = {'num_persons_flag': sus_activity['num_persons_flag'],
                'time_exceeded_flag': time_exceeded_flag,
                'helmet_detected_flag': sus_activity['helmet_detected']}
    
    suspicious_state: bool = any(flags.values())   
    suspicious_label: int = suspicious if suspicious_state else normal  # 1 for SUSPECIOUS 0 for NORMAL
    
    return persons, elapsed_time, flags, person_presence_start_time, suspicious_label

def suspecious_cases(results) -> dict[str: Any]:
    if results is None: raise ValueError("Results cannot be None")
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


def check_person_duration(num_persons: int, person_presence_start_time: datetime, elapsed_time: timedelta, initial_time: datetime) -> tuple[datetime, timedelta]:
    if num_persons < 0:
        raise ValueError("num_persons must be non-negative")
    if num_persons != 0 and person_presence_start_time == initial_time:
        start_time: datetime = datetime.now()
        return start_time, timedelta(0)
    elif num_persons == 0 and person_presence_start_time != initial_time:
        elapsed_time = datetime.now() - person_presence_start_time
        return initial_time, elapsed_time
    else:
        return person_presence_start_time, elapsed_time


def presence_threshold_flag(person_presence: datetime, wait_threshould: int, num_persons: int, initial_time: datetime) -> bool:
    if num_persons < 0:
        raise ValueError("num_persons must be non-negative")
    if num_persons != 0 and person_presence != initial_time:
        wait_time: timedelta = datetime.now() - person_presence
        if wait_time.total_seconds() >= wait_threshould:
            return True
    elif num_persons == 0 and person_presence == initial_time:
        return False
    else:
        return False

def post_sus_data(suspicious_state: int, previous_suspicious: int) -> int:
    if suspicious not in [suspicious, normal] or previous_suspicious not in [suspicious, normal]:
        raise ValueError("suspicious and previous_suspicious must be either 0 or 1")
    status = 0
    if suspicious_state == suspicious and previous_suspicious == normal:
        data = {
            'status': 'SUSPICIOUS' if suspicious == suspicious else 'NORMAL',
            'timestamp': formatted_timestamp  
        }
        #status = send_data_to_endpoint(data, SUSPICIOUS_TARGET_URL, JWT_TOKEN)
        #print("sent suspicious data")
        status = 200

    return status

def main() -> None:

    load_dotenv()

    VIDEO_NAME = "videos/atm_func.mp4"
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

    person_presence_start_time: datetime = datetime(1970, 1, 1, 0, 0, 0)
    elapsed_time: timedelta = timedelta(0)
    suspicious: int = 0  # 1 for SUSPECIOUS 0 for NORMAL
    previous_suspicious: int = 0
    initial_time: datetime = datetime(1970, 1, 1, 0, 0, 0)
    frame_count: int = 0

    atm_model: YOLO = load_model(ATM_MODEL)
    cap: cv2.VideoCapture = load_video(VIDEO_PATH)
    fps: int = int(cap.get(cv2.CAP_PROP_FPS))
    
    while True:
        ret: bool
        frame: np.ndarray

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % fps != 0: # for 1 fps
            logging.info(f"frame : {frame_count % fps}")
            continue

        results = process_frame(atm_model, frame, conf=0.8)[0]
        persons, elapsed_time, all_sus_flags, person_presence_start_time, suspicious_state = atm_suspecious(results
                                                    ,person_presence_start_time, elapsed_time, initial_time, wait_threshould=2)
        post_sus_data(suspicious_state, previous_suspicious)
        previous_suspicious = suspicious_state

        cv2.putText(frame, f"Helmet: {all_sus_flags['helmet_detected_flag']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(frame, f"No of Persons: {persons}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"No of Persons Flag: {all_sus_flags['num_persons_flag']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Presence elapsed time: {elapsed_time.total_seconds()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Start Time: {None if person_presence_start_time == datetime(1970, 1, 1, 0, 0, 0) else person_presence_start_time}",
                     (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Person Time Exceeded: {all_sus_flags['time_exceeded_flag']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"state: {'SUSPICIOUS' if suspicious_state == 1 else 'NORMAL'}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        r = results.plot()
        cv2.imshow('ATM', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

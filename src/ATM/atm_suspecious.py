import cv2
from ultralytics import YOLO
import logging
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Any, Dict, List
from utils import *
import numpy as np

def suspecious_cases(results: YOLO) -> dict[str: Any]:
    try:
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
    except Exception as e:
        logging.error(f"Error in suspecious_cases function: {e}")
        raise

def check_person_duration(num_persons: int, person_presence_start_time: datetime, elapsed_time: timedelta) -> tuple:
    try:
        if num_persons != 0 and person_presence_start_time is None:
            start_time: datetime = datetime.now()
            return start_time, None
        elif num_persons == 0 and person_presence_start_time is not None:
            elapsed_time = datetime.now() - person_presence_start_time
            return None, elapsed_time
        else:
            return person_presence_start_time, elapsed_time
    except Exception as e:
        logging.error(f"Error in check_person_duration function: {e}")
        raise

def presence_threshold_flag(person_presence: datetime, wait_threshould: int, num_persons: int) -> bool:
    try:
        if num_persons != 0 and person_presence is not None:
            wait_time = datetime.now() - person_presence
            if wait_time.total_seconds() >= wait_threshould:
                return True
        elif num_persons == 0 and person_presence is None:
            return False
        else:
            return False
    except Exception as e:
        logging.error(f"Error in presence_threshold_flag function: {e}")
        raise

def atm_suspecious(results: YOLO, person_presence_start_time: datetime, elapsed_time: timedelta, wait_threshould: int) -> tuple:
    try:
        time_exceeded_flag: bool = False
        sus_activity: dict = suspecious_cases(results)
        person_presence_start_time, elapsed_time = check_person_duration(sus_activity['num_persons'], person_presence_start_time, elapsed_time)
        time_exceeded_flag: bool = presence_threshold_flag(person_presence_start_time, wait_threshould, num_persons=sus_activity['num_persons'])
        persons: int = sus_activity['num_persons']
        flags: dict[str: bool] = {'num_persons_flag': sus_activity['num_persons_flag'],
                 'time_exceeded_flag': time_exceeded_flag,
                 'helmet_detected_flag': sus_activity['helmet_detected']}
        
        suspicious: bool = any(flags.values())   
        suspicious_label: str = 'SUSPICIOUS' if suspicious else 'NORMAL'
        
        return persons, elapsed_time, flags, person_presence_start_time, suspicious_label
    except Exception as e:
        logging.error(f"Error in atm_suspecious function: {e}")
        raise

def post_sus_data(suspicious: str, previous_suspicious: str) -> int:
    try:
        status = 0
        if suspicious == 'SUSPICIOUS' and previous_suspicious == 'NORMAL':
            data = {
                'status': suspicious,
                'timestamp': formatted_timestamp  
            }
            # send_data_to_endpoint(data, SUSPICIOUS_TARGET_URL, JWT_TOKEN)
            print("sent suspicious data")
            status = 200

        return status
    except Exception as e:
        logging.error(f"Error in post_sus_data function: {e}")
        raise

def main() -> None:
    try:
        load_dotenv()

        VIDEO_NAME = "videos/atm_func.mp4"
        ATM_MODEL = os.getenv('ATM_MODEL')
        VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

        person_presence_start_time = None
        elapsed_time = None 
        suspicious: bool = False
        previous_suspicious = None

        atm_model: YOLO = load_model(ATM_MODEL)
        cap: cv2.VideoCapture = load_video(VIDEO_PATH)

        while True:
            ret: bool
            frame: Any

            ret, frame = cap.read()
            if not ret:
                break

            results = process_frame(atm_model, frame, conf=0.8)[0]
            print(type(results))
            persons, elapsed_time, all_sus_flags, person_presence_start_time, suspicious = atm_suspecious(results, person_presence_start_time, elapsed_time, wait_threshould=2)
            post_sus_data(suspicious, previous_suspicious)
            previous_suspicious = suspicious

            if elapsed_time is not None:
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

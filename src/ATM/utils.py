import cv2
from ultralytics import YOLO
import logging
import time
from datetime import datetime, timezone, timedelta

def setup_logging(log_filename):
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def load_model(model_path):
    return YOLO(model_path)

def load_video(video_path):
    return cv2.VideoCapture(video_path)

def get_fps(video_capture):
    return video_capture.get(cv2.CAP_PROP_FPS)

def process_frame(model, frame, conf):
    return model(frame, imgsz=640, conf=conf)

def formatted_date():
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
    return formatted_date

def atm_overly(frame, atm_status, trash_count, mess_level,atm_functions, elapsed_time, person_presence, persons, time_exceeded_flag, persons_flag, helmet):
    if elapsed_time is not None:
        cv2.putText(frame, f"Presence Time: {elapsed_time.total_seconds()}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    status_text = f"Room Status: {atm_status}"
    count_text = f"Trash Count: {trash_count}"
    level_text = f"Mess Level: {mess_level}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, level_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"No of Persons: {persons}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv2.putText(frame, f"Helmet: {helmet}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv2.putText(frame, f"Person Start Time: {person_presence}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv2.putText(frame, f"Activity: {time_exceeded_flag}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv2.putText(frame, f"No of Persons Flag: {persons_flag}", (10, 3000), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
    cv2.putText(frame, f"Atm func: {atm_functions}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
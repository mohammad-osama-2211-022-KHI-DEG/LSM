import cv2
from ultralytics import YOLO
import logging
import time
import os
from datetime import datetime, timezone, timedelta
from utils import *
from dotenv import load_dotenv

def trash_count(frame, results):
    trash_count = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        threshold = 0.5
        if score > threshold and results.names[int(class_id)].lower() == 'trash':
            trash_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)   
    return trash_count

def calculate_mess_level(trash_count):
    return round((trash_count / 1000) * 100, 2)

def atm_cleanliness_status(trash_count, mess_level, prev_atm_status = "", prev_mess_level = 0.0):

    if trash_count == 0:
        atm_status = "CLEAN"
    else:
        atm_status = "MESSY"

    if atm_status != prev_atm_status or mess_level != prev_mess_level:
        print(f"Room Status Changed: {prev_atm_status} -> {atm_status}")
        print(f"Mess Level Changed: {prev_mess_level} -> {mess_level}")

    prev_atm_status = atm_status
    prev_mess_level = mess_level
    return atm_status

def atm_cleanliness(frame, results):
    atm_trash_count = trash_count(frame, results)
    mess_level = calculate_mess_level(atm_trash_count)
    atm_statuse = atm_cleanliness_status(atm_trash_count, mess_level)
    return atm_statuse, atm_trash_count, mess_level

def main():
    load_dotenv()

    VIDEO_NAME = "videos/cleanliness.mp4"
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

    atm_model = load_model(ATM_MODEL)

    cap = load_video(VIDEO_PATH)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = process_frame(atm_model, frame, conf=0.8)[0]
        atm_statuse, atm_trash_count, mess_level = atm_cleanliness(frame, results)

        cv2.putText(frame, f"ATM Status: {atm_statuse}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Trash Count: {atm_trash_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Mess Level: {mess_level}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        r = results.plot()

        cv2.imshow('atm', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


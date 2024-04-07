import cv2
from ultralytics import YOLO
from datetime import datetime
import time
import logging
from datetime import datetime, timezone, timedelta
import httpx
import requests
import os
from utils import *
from dotenv import load_dotenv
    
def get_atm_functions(results):
    complainBoxAvailable = False
    telephoneAvailable = False
    res={}
    class_name = []
    confidences = []
    data = dict()

    for result in results:
        class_name, confidences = result.boxes.cls.tolist(), result.boxes.conf.tolist()
        res = dict(zip(class_name, confidences))

        # Check for complain-box and telephone
        alert_cb_tel = complainbox_telephone(class_name)
        complainBoxAvailable = alert_cb_tel["complaintBoxAvailable"]
        telephoneAvailable = alert_cb_tel["telephoneAvailable"]

    return {'res': res, 'complainBoxAvailable': complainBoxAvailable, 'telephoneAvailable': telephoneAvailable}

def detect_atm_and_start_timer(res, atm_detected, start_time):
    
    if len(res) != 0 and 0.0 in res:
        atm_detected = True
        start_time = time.time()  # Start timer
    
    return atm_detected, start_time

def check_transaction_success(atm_detected, start_time, res, threshold):
    if atm_detected:
        elapsed_time = time.time() - start_time
        
        if elapsed_time <= threshold and 1.0 in res:
            return True, elapsed_time, "Successful transaction"
        else:
            return False, elapsed_time, "Unsuccessful transaction (Timeout or no cash detected)"
    else:
        return False, None, "No Transaction is being happened....."


def complainbox_telephone(class_name):
    # Complain-box & Telephone are present
    if 2.0 in class_name and 3.0 in class_name:
        alert= {"complaintBoxAvailable": True,
        "telephoneAvailable": True}
        return alert
    # Complain-box is present & Telephone is not present
    elif 2.0 in class_name and 3.0 not in class_name:
        alert= {"complaintBoxAvailable": True,
        "telephoneAvailable": False}
        return alert
    # Complain-box is not present & Telephone is present
    elif 2.0 not in class_name and 3.0 in class_name:
        alert= {"complaintBoxAvailable": False,
        "telephoneAvailable": True}
        return alert 
    # Complain-box & Telephone are not present
    elif 2.0 not in class_name and 3.0 not in class_name:
        alert= {"complaintBoxAvailable": False,
        "telephoneAvailable": False}
        return alert  

def main():
    load_dotenv()

    atm_detected = False 
    start_time = None
    transection_status = None 
    elapsed_time = None
    transection_message = ''
    start_reset = False
    atm_detected_reset = False

    VIDEO_NAME = "videos/ATM_working.mp4"
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)

    atm_model = load_model(ATM_MODEL)
    cap = load_video(VIDEO_PATH)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = process_frame(atm_model, frame, conf=0.6)[0]
        atm_functions = get_atm_functions(results)
        atm_detected, start_time = detect_atm_and_start_timer(atm_functions['res'], atm_detected, start_time, start_reset, atm_detected_reset)
        transection_status, elapsed_time, transection_message = check_transaction_success(atm_detected, start_time, atm_functions['res'], threshold=1000)
        if transection_status:
            atm_detected = False
            start_time = None

        cv2.putText(frame, f"Atm func: {atm_functions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"transection_status: {transection_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"elapsed_time: {elapsed_time}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"atm_detected: {atm_detected}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"transection_message: {transection_message}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        r = results.plot()

        cv2.imshow('atm', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
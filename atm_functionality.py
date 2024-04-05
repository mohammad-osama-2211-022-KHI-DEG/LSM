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

def atm_transaction(res, class_name):
    atm_detected = False 
    workingStatus = True 
    total_working_count = 0
    total_notworking_count = 0
    counter = 0
    data = dict()

    if len(res) != 0 and not atm_detected:
        if 0.0 in class_name and res[0.0] >= 0.4:
            atm_detected = True
            count = 0
    elif len(res) != 0 and atm_detected:
        if 1.0 in class_name and res[1.0] >= 0.2:
            workingStatus = True
            total_working_count += 1
            atm_detected = False
            return {'workingStatus': workingStatus, 'total_working_count': total_working_count, 'total_notworking_count': total_notworking_count}
        elif count >= 1000:
            counter += 1
            total_notworking_count += 1
            if counter >= 2:
                workingStatus = False
                counter = 0
                return {'workingStatus': workingStatus, 'total_working_count': total_working_count, 'total_notworking_count': total_notworking_count}
            atm_detected = False
        else:
            count += 1
    elif atm_detected:
        if count >= 1000:
            counter += 1
            total_notworking_count += 1
            if counter >= 2:
                workingStatus = False
                counter = 0
                return {'workingStatus': workingStatus, 'total_working_count': total_working_count, 'total_notworking_count': total_notworking_count}
            atm_detected = False
        else:
            count += 1
    else:
        pass

    
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

        data = atm_transaction(res, class_name)

    return {'data': data, 'complainBoxAvailable': complainBoxAvailable, 'telephoneAvailable': telephoneAvailable}

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

    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.getenv('ATM_FUNCTIONALITY_VIDEO_PATH')

    atm_model = load_model(ATM_MODEL)

    cap = load_video(VIDEO_PATH)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = process_frame(atm_model, frame, conf=0.8)[0]
        atm_functions = get_atm_functions(results)

        cv2.putText(frame, f"Atm func: {atm_functions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        r = results.plot()

        cv2.imshow('atm', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
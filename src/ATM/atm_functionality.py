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
    
def get_atm_functions(results, atm_detected, start_time, complainBoxAvailable,telephoneAvailable
                      ,workingstatus, counter, successfull_count, unsuccessfull_count, threshold, workingstatus_threshold):
    res={}
    class_name = []
    confidences = []

    for result in results:
        class_name, confidences = result.boxes.cls.tolist(), result.boxes.conf.tolist()
        res = dict(zip(class_name, confidences))

        # Check for complain-box and telephone
        alert_cb_tel = complainbox_telephone(class_name)
        complainBoxAvailable = alert_cb_tel["complaintBoxAvailable"]
        telephoneAvailable = alert_cb_tel["telephoneAvailable"]

    atm_detected, start_time = detect_atm_and_start_timer(res, atm_detected, start_time)
    transection_status, elapsed_time, transection_message = check_transaction_success(atm_detected, start_time, res, threshold)
    if elapsed_time is not None:
        if elapsed_time >= threshold:
            unsuccessfull_count += 1
            counter += 1
            atm_detected = False
            start_time = None
    if transection_status:
        successfull_count += 1
        atm_detected = False
        start_time = None

    if counter >= workingstatus_threshold:
        workingstatus = False
        counter = 0

    return atm_detected, start_time, {'res': res, 'complainBoxAvailable': complainBoxAvailable, 'telephoneAvailable': telephoneAvailable
            ,'transection_status': transection_status, 'elapsed_time': elapsed_time, 'class_name': class_name
            ,'transection_message': transection_message, 'workingstatus': workingstatus, 'workingstatus_counter': counter
            ,'unsuccessfull_count': unsuccessfull_count, 'successfull_count': successfull_count}

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
        if elapsed_time >= threshold and 1.0 not in res:
            return False, elapsed_time, "Timeout or Cash not detected"
        else:
            return False, elapsed_time, "Tracking transaction....."
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
    start_time = int()
    atm_functions = dict()

    complainBoxAvailable = False
    telephoneAvailable = False
    workingstatus = True
    counter = 0
    successfull_count = 0
    unsuccessfull_count = 0

    previous_unsuccessful_count = 0
    previous_successful_count = 0

    VIDEO_NAME = "videos/ATM_working.mp4"
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)
    FUNC_TARGET_URL = os.path.join(os.getenv('ENDPOINT'), "atm-functionality")
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')
    AUTH_URL = os.path.join(os.getenv('ENDPOINT'), "auth/user")

    #JWT_TOKEN = get_jwt_token(AUTH_URL, USERNAME, PASSWORD)

    atm_model = load_model(ATM_MODEL)
    cap = load_video(VIDEO_PATH)

    while True:

        ret, frame = cap.read()

        if not ret:
            break
        
        results = process_frame(atm_model, frame, conf=0.6)[0]
        atm_detected, start_time, atm_functions = get_atm_functions(results, atm_detected, start_time, complainBoxAvailable, telephoneAvailable,
                                                                    workingstatus, counter, successfull_count, unsuccessfull_count
                                                                    ,threshold=10, workingstatus_threshold = 2)
        unsuccessfull_count = atm_functions['unsuccessfull_count']
        successfull_count = atm_functions['successfull_count']
        workingstatus = atm_functions['workingstatus']
        counter = atm_functions['workingstatus_counter']
        transection_status = atm_functions['transection_status']

        if unsuccessfull_count != previous_unsuccessful_count or successfull_count != previous_successful_count:
            data = {
            "country": "pakistan",
            "branch": "clifton",
            "city": "karachi",
            "timestamp": formatted_timestamp,
            "workingStatus": workingstatus,
            "totalSuccessfulTransaction": successfull_count,
            "totalUnsuccessfulTransaction": unsuccessfull_count,
            "complaintBoxAvailable": atm_functions["complainBoxAvailable"],
            "telephoneAvailable": atm_functions["telephoneAvailable"]
            }
            #send_data_to_endpoint(data, SUSPECIOUS_TARGET_URL, JWT_TOKEN)
            print("Sent data:") 

        previous_unsuccessful_count = unsuccessfull_count
        previous_successful_count = successfull_count
        
        cv2.putText(frame, f"Atm complainBoxAvailable: {atm_functions['complainBoxAvailable']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"Atm telephoneAvailable: {atm_functions['telephoneAvailable']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"transection_status: {atm_functions['transection_status']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"elapsed_time: {atm_functions['elapsed_time']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"atm_detected: {atm_detected}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"transection_message: {atm_functions['transection_message']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"unsuccessfull_count: {unsuccessfull_count}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"successfull_count: {successfull_count}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"workingstatus: {workingstatus}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.putText(frame, f"workingstatus_counter: {counter}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)


        r = results.plot()

        cv2.imshow('atm', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import datetime
from ultralytics import YOLO
import logging
import os
from utils import *
from router import *
from atm_cleanliness import *
from atm_suspecious import *
from atm_functionality import *
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta


def main():
    load_dotenv()

    person_presence_start_time = None
    sus_elapsed_time = None 
    suspecious = False
    previous_suspecious = None
    previous_status = None
    previous_mess_level = None

    atm_detected = False 
    start_time = int()

    complainBoxAvailable = False
    telephoneAvailable = False
    workingstatus = True
    counter = 0
    successfull_count = 0
    unsuccessfull_count = 0

    previous_unsuccessful_count = 0
    previous_successful_count = 0

    VIDEO_NAME = "videos/ATM_working.mp4"
    BRANCH_ID = 1
    ATM_MODEL = os.getenv('ATM_MODEL')
    VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), VIDEO_NAME)
    LOG_FILENAME = os.getenv('LOG_FILENAME')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('PASSWORD')
    AUTH_URL = os.path.join(os.getenv('ENDPOINT'), "auth/user")
    SUSPECIOUS_TARGET_URL = os.path.join(os.getenv('ENDPOINT'), f"suspicious?branchId={BRANCH_ID}")

    setup_logging(LOG_FILENAME)

    atm_model = load_model(ATM_MODEL)
    cap = load_video(VIDEO_PATH)

    #JWT_TOKEN = get_jwt_token(AUTH_URL, USERNAME, PASSWORD)

    # Get current FPS
    fps = get_fps(cap)
    logging.info(f"frame_rate: {fps}")
    # print("Current FPS:", fps)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = process_frame(atm_model, frame, conf=0.8)[0]

        # wait_thresould is in sec
        persons, sus_elapsed_time, all_sus_flags, person_presence_start_time, suspecious = atm_suspecious(results,person_presence_start_time, sus_elapsed_time, wait_threshould = 2) 
        post_suspecious_status = post_sus_data(suspecious, previous_suspecious)
        previous_suspecious = suspecious
        
        atm_status, atm_trash_count, mess_level = atm_cleanliness(frame, results)

        # Check if status has changed from False to True or vice versa
        # if previous_status is not None and previous_status != atm_status:
        #     print("Status changed: ok")
            
        # # Check if mess level has increased or decreased by 10
        # if previous_mess_level is not None and abs(mess_level - previous_mess_level) >= 10:
        #     print("Mess level changed: ok")

        # previous_status = atm_status
        # previous_mess_level = mess_level
        
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

        atm_overly(frame, atm_status, atm_trash_count, mess_level, atm_functions, sus_elapsed_time, person_presence_start_time 
                ,persons, suspecious, post_suspecious_status
                ,all_sus_flags['time_exceeded_flag'], all_sus_flags['num_persons_flag'], all_sus_flags['helmet_detected_flag'])
        
        cv2.putText(frame, f"atm_detected: {atm_detected}", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        r = results.plot()

        cv2.imshow('ATM', r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
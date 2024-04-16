import cv2
import asyncio
import json
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from utils import *
from router import *
from atm_cleanliness import *
from atm_suspecious import *
from atm_functionality import *

logging.basicConfig(level=logging.INFO)
load_dotenv()

async def process_stream(rtsp_url, model):
    cap = cv2.VideoCapture(rtsp_url)

    fps = cap.get(cv2.CAP_PROP_FPS)
    FPS_INTERVAL = 10  # Adjusted variable name to match Python naming conventions
    frame_count = 0

    # Initialize variables as in main.py
    person_presence_start_time = None
    sus_elapsed_time = None
    suspecious = False
    previous_suspecious = None
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

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % FPS_INTERVAL != 0:
            continue

        # Process frame as in main.py
        results = process_frame(model, frame, conf=0.8)[0]

        persons, sus_elapsed_time, all_sus_flags, person_presence_start_time, suspecious = atm_suspecious(
            results, person_presence_start_time, sus_elapsed_time, wait_threshould=2)
        post_suspecious_status = post_sus_data(suspecious, previous_suspecious)
        previous_suspecious = suspecious

        atm_status, atm_trash_count, mess_level = atm_cleanliness(frame, results)

        atm_detected, start_time, atm_functions = get_atm_functions(
            results, atm_detected, start_time, complainBoxAvailable, telephoneAvailable,
            workingstatus, counter, successfull_count, unsuccessfull_count,
            threshold=10, workingstatus_threshold=2)
        unsuccessfull_count = atm_functions['unsuccessfull_count']
        successfull_count = atm_functions['successfull_count']
        workingstatus = atm_functions['workingstatus']
        counter = atm_functions['workingstatus_counter']

        # Overlay information on the frame
        cv2.putText(frame, f"ATM Detected: {atm_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Working Status: {workingstatus}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Successful Transactions: {successfull_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Unsuccessful Transactions: {unsuccessfull_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('ATM Surveillance', frame)

        # Log or process the results as needed
        logging.info(f"ATM Detected: {atm_detected}, Working Status: {workingstatus}, Successful Transactions: {successfull_count}, Unsuccessful Transactions: {unsuccessfull_count}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)

async def process_stream(rtsp_url, model):
# def process_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    FPS_INTERVEL = 10
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % int(fps) != 0:
            logging.info(f"frame : {frame_count % int(fps)}")
            continue

        # print(f'{frame_count} Number of frames successfully executed')
        # Run inference directly on the frame
        results = model(frame)

        # Process the results as needed
        for result in results:
            # Log or process the result here
            results_array = json.loads(result.tojson())
            logging.info(f"Result: {results_array}")

    cap.release()
    cv2.destroyAllWindows()
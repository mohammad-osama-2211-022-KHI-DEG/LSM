import logging
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import tempfile
import asyncio

app = FastAPI()

ATM_MODEL = os.getenv('ATM_MODEL')

# Load the YOLO model
model = YOLO(ATM_MODEL)

# Dictionary to store results for each video
results_dict = {}

async def process_video(video_path, temp_dir, video_id):
    cap = cv2.VideoCapture(video_path)
    total_trash_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as a temporary image file
        temp_image_path = f"{temp_dir}/frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Run inference on the temporary image file
        results = model(temp_image_path)
        for result in results:
            # Convert the result to JSON and load it into a dictionary
            results_array = json.loads(result.tojson())
            logging.info(f"Video ID: {video_id}, {results_array}")

        # Store results in the results_dict
        results_dict[video_id] = results

    cap.release()
    cv2.destroyAllWindows()

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Generate a unique video ID using the current timestamp
    video_id = str(time.time())

    # Save the uploaded video with a unique name
    video_path = f"{video_id}.mp4"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    temp_dir = tempfile.mkdtemp()

    await asyncio.gather(process_video(video_path, temp_dir, video_id))

    os.remove(video_path)  # Remove the temporary video file after processing

    return {"status": "Processing completed for video", "video_id": video_id}

@app.get("/results/{video_id}")
async def get_results(video_id: str):
    if video_id in results_dict:
        return {"results": results_dict[video_id]}
    else:
        return {"error": "Results not found for the given video_id"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

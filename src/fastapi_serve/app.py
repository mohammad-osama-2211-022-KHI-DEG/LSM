import logging
from fastapi import FastAPI
from ultralytics import YOLO
import cv2
import asyncio
import json
from pydantic import BaseModel

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Load the YOLO models
model_paths = {
    "cleanliness": '/home/xloop/LSM/src/models/ATM_models/atm_model_v3.pt',
    "suspicious": '/home/xloop/LSM/src/models/suspecious_activity_last_modified.pt'
}

models = {name: YOLO(path) for name, path in model_paths.items()}

class DetectRequest(BaseModel):
    rtsp_url: str

async def process_stream(rtsp_url, model):
    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference directly on the frame
        results = model(frame)

        # Process the results as needed
        for result in results:
            # Log or process the result here
            results_array = json.loads(result.tojson())
            logging.info(f"Result: {results_array}")

    cap.release()
    cv2.destroyAllWindows()

@app.post("/detect/{model_name}/")
async def detect_objects(model_name: str, request: DetectRequest):
    rtsp_url = request.rtsp_url
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found."}

    model = models[model_name]

    # No need to save the video, directly process the stream
    await asyncio.gather(process_stream(rtsp_url, model))

    return {"status": f"Processing completed for stream using model '{model_name}'"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
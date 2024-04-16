from fastapi import FastAPI
from ultralytics import YOLO
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
from stream_processor import process_stream

app = FastAPI()
load_dotenv()

CLEANLINESS = os.getenv('CLEANLINESS')
ATM_MODEL = os.getenv('ATM_MODEL')
GUARD_MODEL = os.getenv('GUARD_MODEL')

# Load the YOLO models
model_paths = {
    "cleanliness": CLEANLINESS,
    "atm": ATM_MODEL,
    "guard": GUARD_MODEL,
}

models = {name: YOLO(path) for name, path in model_paths.items()}

class DetectRequest(BaseModel):
    rtsp_url: str

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

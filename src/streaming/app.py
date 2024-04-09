from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import cv2
import os

video_app = FastAPI()
load_dotenv()

CLEANLINESS_VIDEO_NAME = "videos/cleanliness.mp4"
SUSPECIOUS_VIDEO_NAME = "videos/suspecious.mp4"
GUARD_ATTIRE_VIDEO_NAME = "videos/2_guards.mp4"

CLEANLINESS_VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), CLEANLINESS_VIDEO_NAME)
GUARD_VIDEO_PATH = os.path.join(os.getenv('GUARD_ATTIRE_VIDEO_PATHS'), GUARD_ATTIRE_VIDEO_NAME)
SUSPECIOUS_VIDEO_PATH = os.path.join(os.getenv('ATM_VIDEO_PATHS'), SUSPECIOUS_VIDEO_NAME)

# Dictionary mapping video names to file paths
video_files = {
    'cleanliness_1': CLEANLINESS_VIDEO_PATH,
    'suspicious_1': SUSPECIOUS_VIDEO_PATH,
    'guard_attire_1': GUARD_VIDEO_PATH,
}

@video_app.get("/video/{video_name}")
def video_feed(video_name: str):
    video_path = video_files.get(video_name)
    if not video_path:
        return Response("Video not found", status_code=404)

    cap = cv2.VideoCapture(video_path)

    def iter_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the image frame to .jpg format
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(iter_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(video_app, host="0.0.0.0", port=5000)
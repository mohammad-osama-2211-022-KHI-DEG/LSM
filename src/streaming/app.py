from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2

video_app = FastAPI()

# Dictionary mapping video names to file paths
video_files = {
    'cleanliness_1': '/home/xloop/LSM/data/ATM_data/videos/cleanliness.mp4',
    'cleanliness_2': '/home/xloop/LSM/data/ATM_data/videos/ATM_working.mp4',
    'suspicious_1': '/home/xloop/LSM/data/ATM_data/videos/suspecious.mp4',
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
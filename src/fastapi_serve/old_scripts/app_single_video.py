from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import tempfile

app = FastAPI()

# Load the YOLO model
model = YOLO('/home/shahzaibkhan/work/bafl_workflows/bafl_mlflow/01-03-2024---ATM_Cleanliness.pt')  # pretrained YOLOv8n model

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_trash_count = 0
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame as a temporary image file
            temp_image_path = f"{temp_dir}/frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg"
            cv2.imwrite(temp_image_path, frame)

            # Run inference on the temporary image file
            result = model(temp_image_path)
            results.append(result)


            # # Process the results
            # for result in results:
            #     for obj in result.boxes[0]:
            #         if obj[-1] == "Trash":
            #             total_trash_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return results

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Save the uploaded video temporarily
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process the video and count "Trash" objects
    results = process_video(video_path)

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import json
import cv2
import tempfile

app = FastAPI()

# Load the YOLO model
model = YOLO('/home/shahzaibkhan/work/bafl_workflows/bafl_mlflow/01-03-2024---ATM_Cleanliness.pt')  # pretrained YOLOv8n model

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    filename = file.filename
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())

    # Run inference on the uploaded image
    results = model([filename])

    # Process the results
    result = results[0]  # Assuming only one image is uploaded
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs

    # Save the result image
    result_image_filename = f"result_{filename}"
    result.save(filename=result_image_filename)

    # Perform additional processing to count "Trash" objects
    total_trash_count = 0
    for result in results:
        # Convert the result to JSON and load it into a dictionary
        results_array = json.loads(result.tojson())
        # Initialize count for trash in the current result
        count = 0
        # Iterate over each object detected in the current result
        for obj in results_array:
            # Check if the object is "Trash"
            if obj["name"] == "Trash":
                # Increment count if it's "Trash"
                count += 1
        # Add count to the total trash count
        total_trash_count += count

    # Print total trash count across all images
    print("Total Trash Count across all images: ", total_trash_count)

    return {"result_image_filename": result_image_filename, "total_trash_count": total_trash_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




# from fastapi import FastAPI, File, UploadFile
# from ultralytics import YOLO

# app = FastAPI()

# # Load your model (adjust path to your model file)
# model = YOLO('yolov8n.pt')

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Read the image file
#     image_data = await file.read()

#     # Perform prediction
#     results = model(image_data)

#     # Convert results to JSON
#     results_json = results.pandas().xyxy[0].to_json(orient="records")

#     return {"predictions": results_json}

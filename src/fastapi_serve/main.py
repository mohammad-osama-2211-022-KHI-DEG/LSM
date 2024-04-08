from ultralytics import YOLO
import json
import os

# Load a model
model = YOLO('/home/shahzaibkhan/work/bafl_workflows/bafl_mlflow/01-03-2024---ATM_Cleanliness.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['images/150_jpeg.jpg', 'images/605_jpg.jpg'])  # return a list of Results objects

# print(results)

for result in results:
    # Convert the result to JSON and load it into a dictionary
    results_array = json.loads(result.tojson())

    print(results_array)



# total_trash_count = 0

# # Iterate over each result in the results list
# for result in results:
#     # Convert the result to JSON and load it into a dictionary
#     results_array = json.loads(result.tojson())
#     # Initialize count for trash in the current result
#     count = 0
#     # Iterate over each object detected in the current result
#     for obj in results_array:
#         # Check if the object is "Trash"
#         if obj["name"] == "Trash":
#             # Increment count if it's "Trash"
#             count += 1
#     # Print trash count for the current image
#     print("Trash Count for image {}: {}".format(results.index(result) + 1, count))
#     # Add count to the total trash count
#     total_trash_count += count

# # Print total trash count across all images
# print("Total Trash Count across all images: ", total_trash_count)


output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    # result.show()  # display to screen
    result.save(filename=os.path.join(output_dir, f'result_{i}.jpg'))  # save to disk with unique filename
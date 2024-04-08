#!/bin/bash

cd /home/xloop/LSM/src/ATM

python3 main.py &

cd /home/xloop/LSM/src/fastapi_serve

uvicorn app:app --workers 4 &

sleep 5

video_path=/home/xloop/LSM/data/ATM_data/videos/cleanliness.mp4

curl -X POST -F "file=@$video_path" http://localhost:8000/detect/
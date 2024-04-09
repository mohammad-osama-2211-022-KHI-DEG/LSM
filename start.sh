#!/bin/bash

cd /home/xloop/LSM/src/ATM

python3 main.py &

cd /home/xloop/LSM/src/fastapi_serve

uvicorn app:app --workers 4 &

sleep 3

cd /home/xloop/LSM/src/streaming

uvicorn app:video_app --port 8001 &

sleep 5

video_path=/home/xloop/LSM/data/ATM_data/videos/cleanliness.mp4

curl -L -X POST http://127.0.0.1:8000/detect/cleanliness/ -H "Content-Type: application/json" -d "{\"rtsp_url\": \"http://127.0.0.1:8001/video/cleanliness_1\"}"
curl -L -X POST http://127.0.0.1:8000/detect/suspicious/ -H "Content-Type: application/json" -d "{\"rtsp_url\": \"http://127.0.0.1:8001/video/suspicious_1\"}"
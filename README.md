# Real-Time Object Detection Web App

A clean, professional Streamlit app for real-time object detection using YOLOv8. Detect objects from images, videos, and live webcam streams, then analyze results with Plotly dashboards.

## Features

- **Sidebar controls**
  - Model selector: yolov8n, yolov8s, yolov8m
  - Confidence threshold slider
  - Input mode selector: Image, Video, Webcam
- **Main area**
  - Annotated detections with bounding boxes and labels
  - Inference time and FPS metrics
- **Results tab**
  - Bar chart of object count per class
  - Average confidence per class
- **Analytics tab**
  - Detection history across uploads
  - Most frequently detected classes
  - Confidence trend over time

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Open the browser at http://localhost:8501 to use the app.

## Deployment

### Docker

Build and run the container:

```bash
docker build -t yolo-streamlit-app .
docker run -p 8501:8501 yolo-streamlit-app
```

### Docker Compose

```bash
docker compose up --build
```

### Heroku / Railway

Deploy using the included `Procfile` and `requirements.txt`.

## Notes

- YOLOv8 model weights are downloaded automatically on first use.
- For best performance, use yolov8n or yolov8s for real-time processing.
- Webcam detection captures a short sequence of frames for live inference.

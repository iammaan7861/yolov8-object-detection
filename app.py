import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import time
import os
from datetime import datetime

st.markdown("""
    <style>
        .stApp { background-color: #0f1117; color: #ffffff; }
        .stMainBlockContainer { background-color: #0f1117; }
        h1, h2, h3, p, label, .stMarkdown { color: #ffffff !important; }
        .stTabs [data-baseweb="tab"] { color: #ffffff !important; }
        [data-testid="stSidebar"] { background-color: #1a1c24; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_OPTIONS = {
    "yolov8n": "YOLOv8 Nano",
    "yolov8s": "YOLOv8 Small",
    "yolov8m": "YOLOv8 Medium",
}

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "model_name" not in st.session_state:
    st.session_state.model_name = None

if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = None

if "class_colors" not in st.session_state:
    st.session_state.class_colors = {}


def load_model(model_name: str) -> YOLO:
    if st.session_state.model_name != model_name or st.session_state.yolo_model is None:
        st.session_state.model_name = model_name
        st.session_state.yolo_model = YOLO(f"{model_name}.pt")
    return st.session_state.yolo_model


def get_color(class_name: str) -> tuple[int, int, int]:
    colors = st.session_state.class_colors
    if class_name not in colors:
        colors[class_name] = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return colors[class_name]


def annotate_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = get_color(det["class_name"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_detection(frame: np.ndarray, model: YOLO, confidence: float, iou: float, max_det: int) -> tuple[list[dict], float]:
    start_time = time.time()
    results = model(frame, conf=confidence, iou=iou, max_det=max_det, verbose=False)
    elapsed_ms = (time.time() - start_time) * 1000
    result = results[0]

    detections = []
    if len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_idx in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(
                {
                    "class_name": result.names[cls_idx],
                    "confidence": float(score),
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return detections, elapsed_ms


def record_detection(mode: str, detections: list[dict], inference_time_ms: float, fps: float) -> None:
    if detections:
        avg_confidence = float(np.mean([det["confidence"] for det in detections]))
    else:
        avg_confidence = 0.0

    st.session_state.detection_history.append(
        {
            "timestamp": datetime.now(),
            "mode": mode,
            "detections": detections,
            "count": len(detections),
            "average_confidence": avg_confidence,
            "inference_time_ms": inference_time_ms,
            "fps": fps,
        }
    )


def build_results():
    records = st.session_state.detection_history
    all_detections = [det for record in records for det in record["detections"]]

    counts = {}
    confidences = {}
    for det in all_detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1
        confidences.setdefault(det["class_name"], []).append(det["confidence"])

    average_confidence = {
        cls: float(np.mean(values)) for cls, values in confidences.items() if values
    }

    return {
        "history": records,
        "counts": counts,
        "average_confidence": average_confidence,
        "total_detections": len(all_detections),
    }


def format_timestamp(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")

# Sidebar Controls
st.sidebar.header("Model & Input")
model_choice = st.sidebar.selectbox(
    "YOLOv8 Model",
    options=list(MODEL_OPTIONS.keys()),
    format_func=lambda key: MODEL_OPTIONS[key],
    index=0,
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.35,
    step=0.05,
)
input_mode = st.sidebar.radio("Input Mode", ["Image", "Video", "Webcam"])

st.sidebar.markdown("---")
st.sidebar.write(
    "Use the sidebar to choose your model, adjust confidence filtering, and switch between image, video, or webcam detection modes."
)

st.title("Real-Time Object Detection")
st.markdown(
    "Detect objects with YOLOv8, visualize results instantly, and monitor analytics over every session."
)

tabs = st.tabs(["Detection", "Results", "Analytics"])

with tabs[0]:
    st.subheader("Detection")
    detection_card = st.empty()

    if input_mode == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            st.image(image, caption="Input Image", use_column_width=True)

            if st.button("Run Image Detection"):
                with st.spinner("Loading YOLO model and running image detection..."):
                    model = load_model(model_choice)
                    detections, inference_time = run_detection(
                        frame, model, confidence_threshold, iou=0.45, max_det=100
                    )
                    annotated = annotate_frame(frame, detections)
                    fps = 1000.0 / inference_time if inference_time > 0 else 0.0

                    st.success(f"Detected {len(detections)} object(s)")
                    st.image(annotated, caption="Detected Objects", use_column_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Inference Time", f"{inference_time:.1f} ms")
                    col2.metric("FPS", f"{fps:.1f}")
                    col3.metric("Detections", len(detections))

                    if detections:
                        details = pd.DataFrame(
                            [
                                {
                                    "Class": det["class_name"],
                                    "Confidence": f"{det['confidence']:.2f}",
                                    "BBox": det["bbox"],
                                }
                                for det in detections
                            ]
                        )
                        st.dataframe(details, use_container_width=True)

                    record_detection("image", detections, inference_time, fps)

    elif input_mode == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_file.read())
            temp_video.flush()
            temp_video.close()

            st.video(temp_video.name)

            if st.button("Run Video Detection"):
                with st.spinner("Processing video frames... this may take a while"):
                    cap = cv2.VideoCapture(temp_video.name)
                    if not cap.isOpened():
                        st.error("Unable to open the uploaded video.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps_input = cap.get(cv2.CAP_PROP_FPS) or 24.0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        writer = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

                        frame_index = 0
                        all_detections = []
                        frame_counts = []
                        timer_start = time.time()
                        progress_bar = st.progress(0)

                        model = load_model(model_choice)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            detections, _ = run_detection(
                                frame, model, confidence_threshold, iou=0.45, max_det=100
                            )
                            all_detections.extend(detections)
                            frame_counts.append(len(detections))
                            annotated = annotate_frame(frame, detections)
                            writer.write(annotated)

                            frame_index += 1
                            if total_frames > 0:
                                progress_bar.progress(min(frame_index / total_frames, 1.0))

                        cap.release()
                        writer.release()
                        elapsed_ms = (time.time() - timer_start) * 1000
                        avg_fps = frame_index / ((elapsed_ms / 1000) or 1)
                        total_detections = len(all_detections)

                        st.success("Video processing complete")
                        st.video(output_path)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Frames", frame_index)
                        col2.metric("Total Detections", total_detections)
                        col3.metric("Avg FPS", f"{avg_fps:.1f}")

                        if total_detections:
                            counts = {}
                            confidence_by_class = {}
                            for det in all_detections:
                                counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1
                                confidence_by_class.setdefault(det["class_name"], []).append(det["confidence"])

                            class_chart = px.bar(
                                x=list(counts.keys()),
                                y=list(counts.values()),
                                labels={"x": "Class", "y": "Count"},
                                title="Object Count by Class"
                            )
                            st.plotly_chart(class_chart, use_container_width=True)

                            avg_conf_chart = px.bar(
                                x=list(confidence_by_class.keys()),
                                y=[float(np.mean(v)) for v in confidence_by_class.values()],
                                labels={"x": "Class", "y": "Average Confidence"},
                                title="Average Confidence per Class"
                            )
                            st.plotly_chart(avg_conf_chart, use_container_width=True)

                        record_detection("video", all_detections, elapsed_ms / max(frame_index, 1), avg_fps)

            try:
                os.remove(temp_video.name)
            except OSError:
                pass

    else:
        st.write("Use your webcam to capture live frames with object detection.")
        if st.button("Start Webcam Detection"):
            with st.spinner("Opening webcam..."):
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Unable to access webcam. Make sure your browser has permission and a camera is connected.")
                else:
                    frame_placeholder = st.empty()
                    frame_count = 0
                    all_detections = []
                    frame_results = []
                    timer_start = time.time()
                    max_frames = 10
                    model = load_model(model_choice)

                    while frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        detections, inference_time = run_detection(
                            frame, model, confidence_threshold, iou=0.45, max_det=100
                        )
                        annotated = annotate_frame(frame, detections)
                        frame_placeholder.image(annotated, caption=f"Frame {frame_count + 1}", use_column_width=True)
                        all_detections.extend(detections)
                        frame_results.append(
                            {
                                "frame": frame_count + 1,
                                "detections": len(detections),
                                "average_confidence": float(np.mean([d["confidence"] for d in detections]))
                                if detections
                                else 0.0,
                                "inference_time_ms": inference_time,
                            }
                        )
                        frame_count += 1
                        time.sleep(0.1)

                    cap.release()
                    elapsed_ms = (time.time() - timer_start) * 1000
                    avg_fps = frame_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

                    st.success(f"Captured {frame_count} webcam frames")
                    st.metric("Webcam FPS", f"{avg_fps:.1f}")
                    st.metric("Total Detections", len(all_detections))

                    if all_detections:
                        counts = {}
                        for det in all_detections:
                            counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1

                        webcam_chart = px.bar(
                            x=list(counts.keys()),
                            y=list(counts.values()),
                            labels={"x": "Class", "y": "Count"},
                            title="Webcam Object Count"
                        )
                        st.plotly_chart(webcam_chart, use_container_width=True)

                    record_detection("webcam", all_detections, elapsed_ms / max(frame_count, 1), avg_fps)

with tabs[1]:
    st.subheader("Results")
    summary = build_results()
    if summary["total_detections"] == 0:
        st.info("No detection history yet. Run one detection to populate result charts.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Detections", summary["total_detections"])
        col2.metric("Classes Detected", len(summary["counts"]))
        col3.metric("Sessions", len(summary["history"]))

        if summary["counts"]:
            count_chart = px.bar(
                x=list(summary["counts"].keys()),
                y=list(summary["counts"].values()),
                labels={"x": "Class", "y": "Count"},
                title="Object Count per Class"
            )
            st.plotly_chart(count_chart, use_container_width=True)

        if summary["average_confidence"]:
            confidence_chart = px.bar(
                x=list(summary["average_confidence"].keys()),
                y=list(summary["average_confidence"].values()),
                labels={"x": "Class", "y": "Average Confidence"},
                title="Average Confidence per Class"
            )
            st.plotly_chart(confidence_chart, use_container_width=True)

with tabs[2]:
    st.subheader("Analytics")
    history = st.session_state.detection_history
    if not history:
        st.info("Analytics will appear after you run a detection.")
    else:
        history_df = pd.DataFrame(
            [
                {
                    "Timestamp": format_timestamp(record["timestamp"]),
                    "Mode": record["mode"].title(),
                    "Detections": record["count"],
                    "Avg Confidence": f"{record['average_confidence']:.3f}",
                    "Inference Time (ms)": f"{record['inference_time_ms']:.1f}",
                    "FPS": f"{record['fps']:.1f}",
                }
                for record in history
            ]
        )

        st.dataframe(history_df, use_container_width=True)

        top_classes = pd.DataFrame(
            {
                "Class": list(summary["counts"].keys()),
                "Count": list(summary["counts"].values()),
            }
        ).sort_values(by="Count", ascending=False)

        st.markdown("### Most Frequently Detected Classes")
        st.write(top_classes.head(10).reset_index(drop=True))

        confidence_over_time = pd.DataFrame(
            [
                {
                    "Timestamp": record["timestamp"],
                    "Avg Confidence": record["average_confidence"],
                    "Mode": record["mode"].title(),
                }
                for record in history
            ]
        )
        confidence_over_time = confidence_over_time.sort_values(by="Timestamp")
        confidence_line = px.line(
            confidence_over_time,
            x="Timestamp",
            y="Avg Confidence",
            color="Mode",
            title="Average Confidence Over Time",
            markers=True,
        )
        st.plotly_chart(confidence_line, use_container_width=True)

        if st.button("Clear Analytics History"):
            st.session_state.detection_history = []
            st.experimental_rerun()

st.markdown("---")
st.caption("Built with Streamlit, Ultralytics YOLOv8, OpenCV, and Plotly for a clean real-time detection experience.")

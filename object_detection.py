import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import time
from pathlib import Path

class YOLODetector:
    """
    YOLOv8 Object Detection wrapper for handling image, video, and webcam inputs.
    """
    
    def __init__(self, model_name: str = "yolov8n"):
        """
        Initialize the YOLO detector.
        
        Args:
            model_name: YOLOv8 model size ('yolov8n', 'yolov8s', 'yolov8m')
        """
        self.model_name = model_name
        self.model = YOLO(f"{model_name}.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def _draw_boxes(self, image, detections, confidence_threshold):
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            detections: List of detections with class, confidence, and bbox
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Annotated image with bounding boxes
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
        image_rgb = image.copy()
        
        for detection in detections:
            if detection['confidence'] >= confidence_threshold:
                x1, y1, x2, y2 = detection['bbox']
                
                # Draw rectangle
                color = self._get_color(detection['class_id'])
                cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                label_y = int(y1) - 5 if y1 > 25 else int(y2) + 20
                cv2.rectangle(image_rgb, 
                            (int(x1), label_y - label_size[1] - 4),
                            (int(x1) + label_size[0] + 4, label_y + 4),
                            color, -1)
                cv2.putText(image_rgb, label, (int(x1) + 2, label_y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    
    def _get_color(self, class_id: int):
        """Generate a consistent color for each class."""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 256, 3).tolist())
    
    def detect_image(self, image, confidence_threshold=0.5, iou_threshold=0.45, max_detections=100):
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image or path to image
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum number of detections
            
        Returns:
            Dictionary with annotated image, detections, inference time, and FPS
        """
        start_time = time.time()
        
        # Convert PIL to array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = cv2.imread(str(image))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model.predict(
            image_array,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        fps = 1000 / inference_time if inference_time > 0 else 0
        
        # Parse detections
        detections = []
        result = results[0]
        
        for i, box in enumerate(result.boxes):
            if i >= max_detections:
                break
            
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class and confidence
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            confidence = float(box.conf[0].cpu().numpy())
            
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        
        # Draw boxes
        annotated_image = self._draw_boxes(image_array, detections, confidence_threshold)
        annotated_image = Image.fromarray(annotated_image)
        
        return {
            'annotated_image': annotated_image,
            'detections': detections,
            'raw_image': image_array
        }, inference_time, fps
    
    def detect_video(self, video_path, confidence_threshold=0.5, iou_threshold=0.45, max_detections=100):
        """
        Detect objects in a video.
        
        Args:
            video_path: Path to video file
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum detections per frame
            
        Returns:
            Dictionary with video statistics and frame detections
        """
        cap = cv2.VideoCapture(str(video_path))
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path
        output_path = Path(f"detected_video_{time.time()}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        frame_detections = []
        frames_with_detections = 0
        total_detections = 0
        start_time = time.time()
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model.predict(
                frame_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # Parse detections
            detections = []
            result = results[0]
            
            for i, box in enumerate(result.boxes):
                if i >= max_detections:
                    break
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
            
            frame_detections.append(detections)
            
            if len(detections) > 0:
                frames_with_detections += 1
                total_detections += len(detections)
            
            # Draw boxes on frame
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                color = self._get_color(detection['class_id'])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = int(y1) - 5 if y1 > 25 else int(y2) + 20
                
                cv2.rectangle(frame, 
                            (int(x1), label_y - label_size[1] - 4),
                            (int(x1) + label_size[0] + 4, label_y + 4),
                            color, -1)
                cv2.putText(frame, label, (int(x1) + 2, label_y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        avg_fps = total_frames / processing_time if processing_time > 0 else 0
        
        return {
            'frame_detections': frame_detections,
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'output_video_path': str(output_path)
        }, processing_time, avg_fps
    
    def detect_webcam(self, confidence_threshold=0.5, iou_threshold=0.45, max_detections=100, num_frames=5):
        """
        Capture and detect objects from webcam.
        
        Args:
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum detections per frame
            num_frames: Number of frames to capture
            
        Returns:
            Dictionary with frame results
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return None
        
        frame_results = []
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_inference_start = time.time()
            
            # Run inference
            results = self.model.predict(
                frame_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            inference_time = (time.time() - frame_inference_start) * 1000
            
            # Parse detections
            detections = []
            result = results[0]
            
            for i, box in enumerate(result.boxes):
                if i >= max_detections:
                    break
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
            
            # Draw boxes
            annotated_frame = self._draw_boxes(frame_rgb, detections, confidence_threshold)
            
            frame_results.append({
                'annotated_image': annotated_frame,
                'detections': detections,
                'inference_time': inference_time
            })
        
        cap.release()
        
        return {
            'frame_results': frame_results
        }

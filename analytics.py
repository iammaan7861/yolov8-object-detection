import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

class AnalyticsManager:
    """
    Manages detection analytics and history.
    """
    
    def __init__(self, storage_file: str = "detection_history.json"):
        """
        Initialize the Analytics Manager.
        
        Args:
            storage_file: Path to JSON file for storing detection history
        """
        self.storage_file = Path(storage_file)
        self.data = self._load_data()
    
    def _load_data(self):
        """Load detection history from storage file."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._initialize_data()
        return self._initialize_data()
    
    def _initialize_data(self):
        """Initialize empty data structure."""
        return {
            'sessions': [],
            'detections': [],
            'class_stats': {}
        }
    
    def _save_data(self):
        """Save detection history to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except IOError as e:
            print(f"Error saving analytics data: {e}")
    
    def add_detection_record(self, mode: str, detections: list, inference_time: float):
        """
        Add a detection record to the history.
        
        Args:
            mode: Detection mode ('image', 'video', 'webcam')
            detections: List of detection dictionaries
            inference_time: Time taken for inference in milliseconds
        """
        session = {
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'detections_count': len(detections),
            'inference_time': inference_time,
            'detections': detections
        }
        
        self.data['sessions'].append(session)
        
        # Add individual detections
        for detection in detections:
            self.data['detections'].append(detection)
            
            # Update class statistics
            class_name = detection['class_name']
            if class_name not in self.data['class_stats']:
                self.data['class_stats'][class_name] = {
                    'count': 0,
                    'confidences': [],
                    'total_confidence': 0.0
                }
            
            self.data['class_stats'][class_name]['count'] += 1
            self.data['class_stats'][class_name]['confidences'].append(detection['confidence'])
            self.data['class_stats'][class_name]['total_confidence'] += detection['confidence']
        
        self._save_data()
    
    def get_analytics(self) -> dict:
        """
        Get comprehensive analytics based on detection history.
        
        Returns:
            Dictionary with analytics data
        """
        if not self.data['detections']:
            return {
                'total_detections': 0,
                'total_sessions': 0,
                'avg_confidence': 0.0,
                'avg_inference_time': 0.0,
                'class_frequencies': {},
                'avg_confidence_by_class': {},
                'mode_distribution': {},
                'class_statistics': {}
            }
        
        # Calculate basic metrics
        total_detections = len(self.data['detections'])
        total_sessions = len(self.data['sessions'])
        
        # Average confidence
        confidences = [det.get('confidence', 0) for det in self.data['detections']]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Average inference time
        inference_times = [session.get('inference_time', 0) for session in self.data['sessions']]
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        # Class frequencies and statistics
        class_frequencies = {}
        avg_confidence_by_class = {}
        class_stats = {}
        
        for detection in self.data['detections']:
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0)
            
            # Frequency
            class_frequencies[class_name] = class_frequencies.get(class_name, 0) + 1
            
            # Stats tracking
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'confidences': [],
                    'detections': 0
                }
            
            class_stats[class_name]['confidences'].append(confidence)
            class_stats[class_name]['detections'] += 1
        
        # Calculate average confidence per class
        for class_name, stats in class_stats.items():
            avg_confidence_by_class[class_name] = np.mean(stats['confidences']) if stats['confidences'] else 0.0
        
        # Mode distribution
        mode_distribution = {}
        for session in self.data['sessions']:
            mode = session.get('mode', 'unknown')
            mode_distribution[mode] = mode_distribution.get(mode, 0) + session.get('detections_count', 0)
        
        # Sort by frequency
        class_frequencies = dict(sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'total_detections': total_detections,
            'total_sessions': total_sessions,
            'avg_confidence': float(avg_confidence),
            'avg_inference_time': float(avg_inference_time),
            'class_frequencies': class_frequencies,
            'avg_confidence_by_class': avg_confidence_by_class,
            'mode_distribution': mode_distribution,
            'class_statistics': class_stats
        }
    
    def get_session_details(self, session_index: int = None) -> dict:
        """
        Get details about a specific session or all sessions.
        
        Args:
            session_index: Index of specific session, or None for all
            
        Returns:
            Session or sessions data
        """
        if session_index is not None:
            if 0 <= session_index < len(self.data['sessions']):
                return self.data['sessions'][session_index]
            return None
        return self.data['sessions']
    
    def get_class_history(self, class_name: str) -> list:
        """
        Get all detections for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of detections for that class
        """
        return [det for det in self.data['detections'] if det.get('class_name') == class_name]
    
    def get_top_classes(self, n: int = 10) -> list:
        """
        Get top N detected classes by frequency.
        
        Args:
            n: Number of classes to return
            
        Returns:
            List of (class_name, frequency) tuples
        """
        analytics = self.get_analytics()
        frequencies = analytics['class_frequencies']
        return sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def export_analytics(self, filepath: str = None) -> str:
        """
        Export analytics to a JSON file.
        
        Args:
            filepath: Path to export to, or None for default
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analytics = self.get_analytics()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analytics': analytics,
            'raw_sessions': self.data['sessions'][:100]  # Limit to last 100 sessions
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
    
    def clear(self):
        """Clear all analytics data."""
        self.data = self._initialize_data()
        self._save_data()
    
    def get_statistics_summary(self) -> str:
        """
        Get a text summary of statistics.
        
        Returns:
            Formatted string with statistics
        """
        analytics = self.get_analytics()
        
        summary = f"""
=== Detection Analytics Summary ===
Total Detections: {analytics['total_detections']}
Total Sessions: {analytics['total_sessions']}
Average Confidence: {analytics['avg_confidence']:.3f}
Average Inference Time: {analytics['avg_inference_time']:.2f}ms

Top Detected Classes:
"""
        
        for class_name, count in list(analytics['class_frequencies'].items())[:5]:
            avg_conf = analytics['avg_confidence_by_class'].get(class_name, 0)
            summary += f"  - {class_name}: {count} detections (avg confidence: {avg_conf:.3f})\n"
        
        summary += f"\nDetection by Mode:\n"
        for mode, count in analytics['mode_distribution'].items():
            summary += f"  - {mode}: {count} detections\n"
        
        return summary

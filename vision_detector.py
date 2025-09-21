"""
Camera and YOLO object detection module
Provides real-time object detection and directional guidance
"""

import time
import threading
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from queue import Queue, Empty
import logging

try:
    from picamera2 import Picamera2
    from picamera2.outputs import FileOutput
    from picamera2.encoders import H264Encoder
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logging.warning("picamera2 not available, using OpenCV camera")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.error("ultralytics YOLO not available")

from config import CameraConfig
from utils import Direction, angle_to_clock_direction, RateLimiter

@dataclass
class DetectedObject:
    """Data structure for detected object"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center_x, center_y
    direction: Direction
    estimated_distance: Optional[float] = None
    timestamp: float = 0.0

@dataclass
class VisionFrame:
    """Data structure for processed vision frame"""
    frame: np.ndarray
    objects: List[DetectedObject]
    timestamp: float
    frame_id: int

class CameraInterface:
    """Camera interface supporting both picamera2 and OpenCV"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = logging.getLogger('camera')
        self.camera = None
        self.using_picamera = False

        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize camera interface"""
        if PICAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()
                # Configure camera
                camera_config = self.camera.create_preview_configuration(
                    main={"size": self.config.resolution}
                )
                self.camera.configure(camera_config)
                self.using_picamera = True
                self.logger.info("Initialized Picamera2")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize Picamera2: {e}")

        # Fallback to OpenCV
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config.framerate)
            self.using_picamera = False
            self.logger.info("Initialized OpenCV camera")
        except Exception as e:
            self.logger.error(f"Failed to initialize any camera: {e}")
            raise

    def start(self):
        """Start camera capture"""
        if self.using_picamera:
            self.camera.start()
        else:
            # OpenCV camera is ready immediately
            pass

    def stop(self):
        """Stop camera capture"""
        if self.using_picamera:
            self.camera.stop()
        else:
            self.camera.release()

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        try:
            if self.using_picamera:
                frame = self.camera.capture_array()
                # Convert from RGB to BGR for OpenCV compatibility
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            else:
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class YOLODetector:
    """YOLO object detection wrapper"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = logging.getLogger('yolo_detector')
        self.model = None

        # Common object classes we care about for mobility
        self.important_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
            11: 'stop sign',
            15: 'cat',
            16: 'dog',
            39: 'bottle',
            41: 'cup',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone'
        }

        # Typical object sizes for distance estimation (height in meters)
        self.object_sizes = {
            'person': 1.7,
            'car': 1.5,
            'bicycle': 1.0,
            'motorcycle': 1.2,
            'bus': 3.0,
            'truck': 3.5,
            'traffic light': 0.8,
            'stop sign': 0.8,
            'chair': 0.9,
            'bottle': 0.25
        }

        self._load_model()

    def _load_model(self):
        """Load YOLO model"""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available")
            return

        try:
            self.model = YOLO(self.config.model_path)
            self.logger.info(f"Loaded YOLO model: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")

    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect objects in frame"""
        if self.model is None:
            return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                verbose=False
            )

            objects = []
            frame_height, frame_width = frame.shape[:2]

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection data
                        class_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        bbox = box.xyxy.cpu().numpy()[0].astype(int)

                        # Skip if not an important class
                        if class_id not in self.important_classes:
                            continue

                        class_name = self.important_classes[class_id]

                        # Calculate center point
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)

                        # Convert to direction based on horizontal position
                        # Divide frame into directional zones
                        frame_center = frame_width / 2
                        relative_x = (center_x - frame_center) / frame_center

                        # Map to clock directions (approximate)
                        if relative_x < -0.6:
                            direction = Direction.TEN
                        elif relative_x < -0.3:
                            direction = Direction.ELEVEN
                        elif relative_x < -0.1:
                            direction = Direction.TWELVE  # Slight left
                        elif relative_x < 0.1:
                            direction = Direction.TWELVE  # Center
                        elif relative_x < 0.3:
                            direction = Direction.ONE
                        elif relative_x < 0.6:
                            direction = Direction.TWO
                        else:
                            direction = Direction.THREE

                        # Estimate distance based on object size
                        estimated_distance = self._estimate_distance(
                            class_name, bbox, frame_height
                        )

                        obj = DetectedObject(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=tuple(bbox),
                            center=(center_x, center_y),
                            direction=direction,
                            estimated_distance=estimated_distance,
                            timestamp=time.time()
                        )

                        objects.append(obj)

            return objects

        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return []

    def _estimate_distance(self, class_name: str, bbox: Tuple[int, int, int, int],
                          frame_height: int) -> Optional[float]:
        """Estimate distance to object based on its size in pixels"""
        if class_name not in self.object_sizes:
            return None

        try:
            # Calculate object height in pixels
            object_height_pixels = bbox[3] - bbox[1]

            # Estimate distance using similar triangles
            # Assume camera focal length equivalent to 35mm camera (~700 pixels for our resolution)
            focal_length = 700  # Approximate focal length in pixels
            real_height = self.object_sizes[class_name]

            # Distance = (real_height * focal_length) / pixel_height
            distance = (real_height * focal_length) / object_height_pixels

            # Clamp to reasonable values
            distance = max(0.5, min(50.0, distance))

            return distance

        except Exception as e:
            self.logger.debug(f"Error estimating distance: {e}")
            return None

class VisionDetector:
    """Main vision detection system"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = logging.getLogger('vision_detector')

        self.camera = CameraInterface(config)
        self.detector = YOLODetector(config)

        self.frame_queue = Queue(maxsize=5)
        self.results_queue = Queue(maxsize=20)

        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        self.frame_counter = 0

        self.rate_limiter = RateLimiter(1.0 / config.processing_interval)

    def _capture_thread_func(self):
        """Thread function for camera capture"""
        with self.camera as cam:
            while self.running:
                try:
                    frame = cam.capture_frame()
                    if frame is not None:
                        # Add frame to queue (non-blocking)
                        try:
                            self.frame_queue.put_nowait((frame, time.time(), self.frame_counter))
                            self.frame_counter += 1
                        except:
                            # Queue full, skip this frame
                            pass

                    # Control frame rate
                    time.sleep(1.0 / self.config.framerate)

                except Exception as e:
                    self.logger.error(f"Error in capture thread: {e}")
                    time.sleep(0.1)

    def _detection_thread_func(self):
        """Thread function for object detection"""
        while self.running:
            try:
                if self.rate_limiter.should_execute():
                    # Get latest frame
                    try:
                        frame, timestamp, frame_id = self.frame_queue.get(timeout=0.1)

                        # Detect objects
                        objects = self.detector.detect_objects(frame)

                        # Create result
                        result = VisionFrame(
                            frame=frame,
                            objects=objects,
                            timestamp=timestamp,
                            frame_id=frame_id
                        )

                        # Add to results queue
                        try:
                            self.results_queue.put_nowait(result)
                        except:
                            # Queue full, skip this result
                            pass

                    except Empty:
                        # No frame available
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in detection thread: {e}")
                time.sleep(0.1)

    def start(self):
        """Start vision detection system"""
        if self.running:
            self.logger.warning("Vision detector already running")
            return

        self.running = True

        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_thread_func,
            name="vision_capture",
            daemon=True
        )
        self.capture_thread.start()

        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self._detection_thread_func,
            name="vision_detection",
            daemon=True
        )
        self.detection_thread.start()

        self.logger.info("Vision detector started")

    def stop(self):
        """Stop vision detection system"""
        self.logger.info("Stopping vision detector...")
        self.running = False

        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        while not self.results_queue.empty():
            try:
                self.results_queue.get_nowait()
            except Empty:
                break

        self.logger.info("Vision detector stopped")

    def get_latest_detections(self, max_results: int = 5) -> List[VisionFrame]:
        """Get latest detection results"""
        results = []
        count = 0

        while count < max_results and not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                results.append(result)
                count += 1
            except Empty:
                break

        return results

    def get_objects_by_direction(self) -> Dict[Direction, List[DetectedObject]]:
        """Get detected objects organized by direction"""
        latest_results = self.get_latest_detections(max_results=1)
        if not latest_results:
            return {}

        objects_by_direction = {}
        for obj in latest_results[0].objects:
            if obj.direction not in objects_by_direction:
                objects_by_direction[obj.direction] = []
            objects_by_direction[obj.direction].append(obj)

        return objects_by_direction

    def get_closest_objects(self, max_objects: int = 5) -> List[DetectedObject]:
        """Get closest detected objects"""
        latest_results = self.get_latest_detections(max_results=1)
        if not latest_results:
            return []

        objects_with_distance = [
            obj for obj in latest_results[0].objects
            if obj.estimated_distance is not None
        ]

        # Sort by distance
        objects_with_distance.sort(key=lambda x: x.estimated_distance)

        return objects_with_distance[:max_objects]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Test function
def test_vision_detector():
    """Test function for vision detector"""
    from config import CAMERA_CONFIG

    logging.basicConfig(level=logging.INFO)

    with VisionDetector(CAMERA_CONFIG) as detector:
        print("Testing vision detector for 30 seconds...")

        for i in range(300):
            time.sleep(0.1)

            objects = detector.get_closest_objects()
            if objects:
                for obj in objects[:3]:  # Show top 3
                    print(f"Detected {obj.class_name} at {obj.direction.name}, "
                          f"confidence: {obj.confidence:.2f}, "
                          f"distance: {obj.estimated_distance:.1f}m" if obj.estimated_distance else "distance: unknown")

if __name__ == "__main__":
    test_vision_detector()
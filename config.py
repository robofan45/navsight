"""
Configuration settings for the Mobility Aid System
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class UltrasonicConfig:
    """Configuration for ultrasonic sensors"""
    sensors: Dict[str, Dict[str, int]]  # sensor_name -> {trigger_pin, echo_pin}
    max_distance: float = 2.5  # meters
    min_distance: float = 0.02  # meters
    measurement_timeout: float = 0.5  # seconds
    sample_rate: float = 10.0  # Hz

@dataclass
class CameraConfig:
    """Configuration for camera and vision detection"""
    resolution: Tuple[int, int] = (640, 480)
    framerate: int = 30
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    processing_interval: float = 0.2  # seconds

@dataclass
class GPSConfig:
    """Configuration for GPS navigation"""
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout: float = 1.0
    update_interval: float = 1.0  # seconds

@dataclass
class AudioConfig:
    """Configuration for audio guidance"""
    engine: str = "pyttsx3"  # or "espeak"
    voice_rate: int = 150  # words per minute
    voice_volume: float = 0.9
    priority_queue_size: int = 10

@dataclass
class SystemConfig:
    """Main system configuration"""
    log_level: int = logging.INFO
    log_file: str = "/var/log/mobility_aid.log"
    status_led_pin: int = 18
    emergency_button_pin: int = 16
    main_loop_interval: float = 0.1  # seconds

    # Distance thresholds
    immediate_danger_distance: float = 0.3  # meters
    warning_distance: float = 1.0  # meters
    caution_distance: float = 2.0  # meters

# Default configuration instance
CONFIG = SystemConfig()
ULTRASONIC_CONFIG = UltrasonicConfig(
    sensors={
        "front": {"trigger_pin": 23, "echo_pin": 24},
        "front_left": {"trigger_pin": 25, "echo_pin": 8},
        "front_right": {"trigger_pin": 7, "echo_pin": 1},
        "left": {"trigger_pin": 12, "echo_pin": 16},
        "right": {"trigger_pin": 20, "echo_pin": 21}
    }
)
CAMERA_CONFIG = CameraConfig()
GPS_CONFIG = GPSConfig()
AUDIO_CONFIG = AudioConfig()
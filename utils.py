"""
Utility functions for the Mobility Aid System
"""

import time
import math
import logging
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum

class Direction(Enum):
    """Clock-face directions for intuitive guidance"""
    TWELVE = 0      # Straight ahead
    ONE = 30        # 30 degrees right
    TWO = 60        # 60 degrees right
    THREE = 90      # 90 degrees right
    FOUR = 120      # 120 degrees right
    FIVE = 150      # 150 degrees right
    SIX = 180       # Behind
    SEVEN = 210     # 150 degrees left
    EIGHT = 240     # 120 degrees left
    NINE = 270      # 90 degrees left
    TEN = 300       # 60 degrees left
    ELEVEN = 330    # 30 degrees left

def angle_to_clock_direction(angle_degrees: float) -> Direction:
    """Convert angle in degrees to clock direction"""
    # Normalize angle to 0-360
    angle_degrees = angle_degrees % 360

    # Find closest clock direction
    directions = list(Direction)
    closest_dir = min(directions, key=lambda d: min(
        abs(angle_degrees - d.value),
        abs(angle_degrees - d.value + 360),
        abs(angle_degrees - d.value - 360)
    ))

    return closest_dir

def clock_direction_to_text(direction: Direction) -> str:
    """Convert clock direction to spoken text"""
    direction_map = {
        Direction.TWELVE: "straight ahead",
        Direction.ONE: "one o'clock",
        Direction.TWO: "two o'clock",
        Direction.THREE: "three o'clock",
        Direction.FOUR: "four o'clock",
        Direction.FIVE: "five o'clock",
        Direction.SIX: "behind you",
        Direction.SEVEN: "seven o'clock",
        Direction.EIGHT: "eight o'clock",
        Direction.NINE: "nine o'clock",
        Direction.TEN: "ten o'clock",
        Direction.ELEVEN: "eleven o'clock"
    }
    return direction_map[direction]

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing between two GPS coordinates"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)

    return (bearing_deg + 360) % 360

def setup_logging(log_level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('mobility_aid')
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")

    return logger

class MovingAverage:
    """Simple moving average filter for sensor data"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.values = []

    def add_value(self, value: float) -> float:
        """Add new value and return current average"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

    def reset(self):
        """Reset the filter"""
        self.values.clear()

class RateLimiter:
    """Rate limiter for controlling operation frequency"""

    def __init__(self, max_rate: float):
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate
        self.last_call = 0

    def should_execute(self) -> bool:
        """Check if enough time has passed for next execution"""
        current_time = time.time()
        if current_time - self.last_call >= self.min_interval:
            self.last_call = current_time
            return True
        return False

    def wait_if_needed(self):
        """Wait if needed to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_call
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_call = time.time()

def validate_sensor_data(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validate that sensor data contains required fields"""
    return all(field in data and data[field] is not None for field in required_fields)

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default
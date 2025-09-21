"""
Ultrasonic sensor interface for obstacle detection
Supports multiple HC-SR04 sensors for 360-degree coverage
"""

import time
import threading
import RPi.GPIO as GPIO
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
import logging

from config import UltrasonicConfig
from utils import MovingAverage, RateLimiter, Direction, angle_to_clock_direction

@dataclass
class UltrasonicReading:
    """Data structure for ultrasonic sensor reading"""
    sensor_name: str
    distance: float  # meters
    timestamp: float
    direction: Direction
    valid: bool = True

class UltrasonicSensor:
    """Individual ultrasonic sensor controller"""

    def __init__(self, name: str, trigger_pin: int, echo_pin: int,
                 max_distance: float = 2.5, timeout: float = 0.5):
        self.name = name
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.max_distance = max_distance
        self.timeout = timeout
        self.logger = logging.getLogger(f'ultrasonic.{name}')

        # Setup GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trigger_pin, False)

        # Allow sensor to settle
        time.sleep(0.1)

        self.logger.info(f"Ultrasonic sensor {name} initialized on pins {trigger_pin}/{echo_pin}")

    def measure_distance(self) -> Optional[float]:
        """Measure distance using ultrasonic sensor"""
        try:
            # Send trigger pulse
            GPIO.output(self.trigger_pin, True)
            time.sleep(0.00001)  # 10μs pulse
            GPIO.output(self.trigger_pin, False)

            # Wait for echo start
            pulse_start = time.time()
            timeout_start = pulse_start
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > self.timeout:
                    self.logger.warning(f"Echo start timeout for sensor {self.name}")
                    return None

            # Wait for echo end
            pulse_end = time.time()
            timeout_end = pulse_end
            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end - timeout_end > self.timeout:
                    self.logger.warning(f"Echo end timeout for sensor {self.name}")
                    return None

            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150  # Speed of sound = 343 m/s

            # Validate reading
            if 0.02 <= distance <= self.max_distance:
                return distance
            else:
                self.logger.debug(f"Invalid distance reading: {distance:.3f}m")
                return None

        except Exception as e:
            self.logger.error(f"Error measuring distance: {e}")
            return None

    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            GPIO.cleanup([self.trigger_pin, self.echo_pin])
        except Exception as e:
            self.logger.error(f"Error cleaning up GPIO: {e}")

class UltrasonicArray:
    """Manager for multiple ultrasonic sensors"""

    def __init__(self, config: UltrasonicConfig):
        self.config = config
        self.sensors: Dict[str, UltrasonicSensor] = {}
        self.readings_queue = Queue(maxsize=100)
        self.running = False
        self.threads: List[threading.Thread] = []
        self.logger = logging.getLogger('ultrasonic_array')

        # Moving average filters for each sensor
        self.filters: Dict[str, MovingAverage] = {}

        # Rate limiter for sensor readings
        self.rate_limiter = RateLimiter(config.sample_rate)

        # Sensor direction mapping (degrees from forward)
        self.sensor_directions = {
            "front": 0,
            "front_left": 330,
            "front_right": 30,
            "left": 270,
            "right": 90,
            "back_left": 210,
            "back_right": 150,
            "back": 180
        }

        self._initialize_sensors()

    def _initialize_sensors(self):
        """Initialize all configured sensors"""
        for sensor_name, pins in self.config.sensors.items():
            try:
                sensor = UltrasonicSensor(
                    name=sensor_name,
                    trigger_pin=pins["trigger_pin"],
                    echo_pin=pins["echo_pin"],
                    max_distance=self.config.max_distance,
                    timeout=self.config.measurement_timeout
                )
                self.sensors[sensor_name] = sensor
                self.filters[sensor_name] = MovingAverage(window_size=3)
                self.logger.info(f"Initialized sensor: {sensor_name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize sensor {sensor_name}: {e}")

    def _sensor_thread(self, sensor_name: str):
        """Thread function for continuous sensor reading"""
        sensor = self.sensors[sensor_name]
        filter_obj = self.filters[sensor_name]

        while self.running:
            try:
                if self.rate_limiter.should_execute():
                    distance = sensor.measure_distance()

                    if distance is not None:
                        # Apply moving average filter
                        filtered_distance = filter_obj.add_value(distance)

                        # Get direction for this sensor
                        angle = self.sensor_directions.get(sensor_name, 0)
                        direction = angle_to_clock_direction(angle)

                        # Create reading object
                        reading = UltrasonicReading(
                            sensor_name=sensor_name,
                            distance=filtered_distance,
                            timestamp=time.time(),
                            direction=direction,
                            valid=True
                        )

                        # Add to queue (non-blocking)
                        try:
                            self.readings_queue.put_nowait(reading)
                        except:
                            # Queue full, skip this reading
                            pass

                    else:
                        # Invalid reading
                        reading = UltrasonicReading(
                            sensor_name=sensor_name,
                            distance=0.0,
                            timestamp=time.time(),
                            direction=Direction.TWELVE,
                            valid=False
                        )

                        try:
                            self.readings_queue.put_nowait(reading)
                        except:
                            pass

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in sensor thread {sensor_name}: {e}")
                time.sleep(0.1)

    def start(self):
        """Start continuous sensor monitoring"""
        if self.running:
            self.logger.warning("Ultrasonic array already running")
            return

        self.running = True

        # Start a thread for each sensor
        for sensor_name in self.sensors:
            thread = threading.Thread(
                target=self._sensor_thread,
                args=(sensor_name,),
                name=f"ultrasonic_{sensor_name}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)

        self.logger.info(f"Started {len(self.threads)} ultrasonic sensor threads")

    def stop(self):
        """Stop sensor monitoring and cleanup"""
        self.logger.info("Stopping ultrasonic array...")
        self.running = False

        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        # Clear readings queue
        while not self.readings_queue.empty():
            try:
                self.readings_queue.get_nowait()
            except Empty:
                break

        # Cleanup sensors
        for sensor in self.sensors.values():
            sensor.cleanup()

        self.threads.clear()
        self.logger.info("Ultrasonic array stopped")

    def get_latest_readings(self, max_readings: int = 10) -> List[UltrasonicReading]:
        """Get the latest sensor readings"""
        readings = []
        count = 0

        while count < max_readings and not self.readings_queue.empty():
            try:
                reading = self.readings_queue.get_nowait()
                readings.append(reading)
                count += 1
            except Empty:
                break

        return readings

    def get_closest_obstacle(self) -> Optional[UltrasonicReading]:
        """Get the closest valid obstacle reading"""
        readings = self.get_latest_readings()
        valid_readings = [r for r in readings if r.valid and r.distance > 0]

        if not valid_readings:
            return None

        return min(valid_readings, key=lambda r: r.distance)

    def get_obstacles_by_direction(self) -> Dict[Direction, float]:
        """Get obstacle distances organized by direction"""
        readings = self.get_latest_readings()
        obstacles = {}

        for reading in readings:
            if reading.valid and reading.distance > 0:
                current_distance = obstacles.get(reading.direction)
                if current_distance is None or reading.distance < current_distance:
                    obstacles[reading.direction] = reading.distance

        return obstacles

    def is_path_clear(self, direction: Direction, min_distance: float = 1.0) -> bool:
        """Check if path in given direction is clear"""
        obstacles = self.get_obstacles_by_direction()
        distance = obstacles.get(direction)
        return distance is None or distance >= min_distance

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

# Test function
def test_ultrasonic_array():
    """Test function for ultrasonic sensor array"""
    from config import ULTRASONIC_CONFIG

    logging.basicConfig(level=logging.INFO)

    with UltrasonicArray(ULTRASONIC_CONFIG) as array:
        print("Testing ultrasonic array for 10 seconds...")

        for i in range(100):
            time.sleep(0.1)

            closest = array.get_closest_obstacle()
            if closest:
                print(f"Closest obstacle: {closest.distance:.2f}m at {closest.direction.name}")

            obstacles = array.get_obstacles_by_direction()
            if obstacles:
                obstacle_str = ", ".join([f"{d.name}: {dist:.2f}m" for d, dist in obstacles.items()])
                print(f"All obstacles: {obstacle_str}")

if __name__ == "__main__":
    test_ultrasonic_array()
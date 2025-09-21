"""
GPS navigation module for positioning and waypoint navigation
Supports NMEA GPS modules via serial interface
"""

import time
import threading
import serial
import math
from typing import Optional, List, Dict, Tuple, NamedTuple
from dataclasses import dataclass
from queue import Queue, Empty
import logging
import re

from config import GPSConfig
from utils import calculate_distance, calculate_bearing, Direction, angle_to_clock_direction, RateLimiter

class GPSCoordinate(NamedTuple):
    """GPS coordinate structure"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None

@dataclass
class GPSFix:
    """GPS fix information"""
    coordinate: GPSCoordinate
    timestamp: float
    quality: int  # 0=invalid, 1=GPS, 2=DGPS
    satellites: int
    hdop: float  # Horizontal dilution of precision
    speed: Optional[float] = None  # Speed in m/s
    course: Optional[float] = None  # Course in degrees
    valid: bool = True

@dataclass
class Waypoint:
    """Navigation waypoint"""
    name: str
    coordinate: GPSCoordinate
    description: str = ""
    radius: float = 5.0  # Arrival radius in meters

@dataclass
class NavigationStatus:
    """Navigation status information"""
    current_position: Optional[GPSCoordinate]
    target_waypoint: Optional[Waypoint]
    distance_to_target: Optional[float]
    bearing_to_target: Optional[float]
    direction_to_target: Optional[Direction]
    navigation_active: bool = False

class NMEAParser:
    """NMEA sentence parser for GPS data"""

    def __init__(self):
        self.logger = logging.getLogger('nmea_parser')

    def parse_gga(self, sentence: str) -> Optional[GPSFix]:
        """Parse GPGGA sentence for position fix"""
        try:
            parts = sentence.split(',')
            if len(parts) < 15 or parts[0] != '$GPGGA':
                return None

            # Check if we have valid data
            if not parts[2] or not parts[4] or parts[6] == '0':
                return None

            # Parse timestamp
            time_str = parts[1]
            if len(time_str) >= 6:
                hours = int(time_str[:2])
                minutes = int(time_str[2:4])
                seconds = float(time_str[4:])
                timestamp = hours * 3600 + minutes * 60 + seconds
            else:
                timestamp = time.time()

            # Parse latitude
            lat_raw = float(parts[2])
            lat_deg = int(lat_raw / 100)
            lat_min = lat_raw - (lat_deg * 100)
            latitude = lat_deg + lat_min / 60.0
            if parts[3] == 'S':
                latitude = -latitude

            # Parse longitude
            lon_raw = float(parts[4])
            lon_deg = int(lon_raw / 100)
            lon_min = lon_raw - (lon_deg * 100)
            longitude = lon_deg + lon_min / 60.0
            if parts[5] == 'W':
                longitude = -longitude

            # Parse other fields
            quality = int(parts[6]) if parts[6] else 0
            satellites = int(parts[7]) if parts[7] else 0
            hdop = float(parts[8]) if parts[8] else 99.9
            altitude = float(parts[9]) if parts[9] else None

            coordinate = GPSCoordinate(latitude, longitude, altitude)

            return GPSFix(
                coordinate=coordinate,
                timestamp=timestamp,
                quality=quality,
                satellites=satellites,
                hdop=hdop,
                valid=True
            )

        except Exception as e:
            self.logger.debug(f"Error parsing GGA sentence: {e}")
            return None

    def parse_rmc(self, sentence: str) -> Optional[Dict]:
        """Parse GPRMC sentence for speed and course"""
        try:
            parts = sentence.split(',')
            if len(parts) < 12 or parts[0] != '$GPRMC':
                return None

            # Check if data is valid
            if parts[2] != 'A':
                return None

            # Parse speed (knots to m/s)
            speed = float(parts[7]) * 0.514444 if parts[7] else None

            # Parse course (degrees)
            course = float(parts[8]) if parts[8] else None

            return {
                'speed': speed,
                'course': course
            }

        except Exception as e:
            self.logger.debug(f"Error parsing RMC sentence: {e}")
            return None

class GPSReceiver:
    """GPS receiver interface"""

    def __init__(self, config: GPSConfig):
        self.config = config
        self.logger = logging.getLogger('gps_receiver')
        self.serial_port = None
        self.parser = NMEAParser()

        self.running = False
        self.thread = None
        self.fix_queue = Queue(maxsize=10)

        self.rate_limiter = RateLimiter(1.0 / config.update_interval)

        self._connect()

    def _connect(self):
        """Connect to GPS receiver"""
        try:
            self.serial_port = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            self.logger.info(f"Connected to GPS on {self.config.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to GPS: {e}")
            self.serial_port = None

    def _read_thread(self):
        """Thread function for reading GPS data"""
        while self.running and self.serial_port:
            try:
                if self.rate_limiter.should_execute():
                    line = self.serial_port.readline().decode('ascii', errors='ignore').strip()

                    if line.startswith('$GPGGA'):
                        fix = self.parser.parse_gga(line)
                        if fix:
                            try:
                                self.fix_queue.put_nowait(fix)
                            except:
                                # Queue full, skip
                                pass

                    elif line.startswith('$GPRMC'):
                        rmc_data = self.parser.parse_rmc(line)
                        if rmc_data:
                            # Could be used to update speed/course in latest fix
                            pass

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error reading GPS data: {e}")
                time.sleep(1.0)

    def start(self):
        """Start GPS data collection"""
        if self.running or not self.serial_port:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._read_thread,
            name="gps_reader",
            daemon=True
        )
        self.thread.start()
        self.logger.info("GPS receiver started")

    def stop(self):
        """Stop GPS data collection"""
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.serial_port:
            self.serial_port.close()

        # Clear queue
        while not self.fix_queue.empty():
            try:
                self.fix_queue.get_nowait()
            except Empty:
                break

        self.logger.info("GPS receiver stopped")

    def get_latest_fix(self) -> Optional[GPSFix]:
        """Get the latest GPS fix"""
        latest_fix = None

        # Get the most recent fix from queue
        while not self.fix_queue.empty():
            try:
                latest_fix = self.fix_queue.get_nowait()
            except Empty:
                break

        return latest_fix

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class GPSNavigator:
    """GPS navigation system with waypoint support"""

    def __init__(self, config: GPSConfig):
        self.config = config
        self.logger = logging.getLogger('gps_navigator')

        self.gps_receiver = GPSReceiver(config)

        # Navigation state
        self.current_position: Optional[GPSCoordinate] = None
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index: int = 0
        self.navigation_active: bool = False

        # GPS health tracking
        self.last_fix_time: float = 0
        self.fix_timeout: float = 10.0  # seconds

        self.running = False
        self.thread = None

    def start(self):
        """Start GPS navigation system"""
        if self.running:
            return

        self.gps_receiver.start()
        self.running = True

        # Start navigation thread
        self.thread = threading.Thread(
            target=self._navigation_thread,
            name="gps_navigation",
            daemon=True
        )
        self.thread.start()

        self.logger.info("GPS navigator started")

    def stop(self):
        """Stop GPS navigation system"""
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.gps_receiver.stop()
        self.logger.info("GPS navigator stopped")

    def _navigation_thread(self):
        """Main navigation processing thread"""
        while self.running:
            try:
                # Get latest GPS fix
                fix = self.gps_receiver.get_latest_fix()

                if fix and fix.valid:
                    self.current_position = fix.coordinate
                    self.last_fix_time = time.time()

                    # Process navigation if active
                    if self.navigation_active and self.waypoints:
                        self._process_navigation()

                time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error in navigation thread: {e}")
                time.sleep(1.0)

    def _process_navigation(self):
        """Process navigation to current waypoint"""
        if not self.current_position or not self.waypoints:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            self.navigation_active = False
            self.logger.info("Navigation completed - reached final waypoint")
            return

        current_waypoint = self.waypoints[self.current_waypoint_index]

        # Calculate distance to waypoint
        distance = self._calculate_distance_to_waypoint(current_waypoint)

        # Check if we've reached the waypoint
        if distance <= current_waypoint.radius:
            self.logger.info(f"Reached waypoint: {current_waypoint.name}")
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.waypoints):
                self.navigation_active = False
                self.logger.info("Navigation completed")

    def _calculate_distance_to_waypoint(self, waypoint: Waypoint) -> float:
        """Calculate distance to waypoint in meters"""
        if not self.current_position:
            return float('inf')

        # Use Haversine formula for accurate distance
        lat1, lon1 = math.radians(self.current_position.latitude), math.radians(self.current_position.longitude)
        lat2, lon2 = math.radians(waypoint.coordinate.latitude), math.radians(waypoint.coordinate.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in meters
        R = 6371000
        distance = R * c

        return distance

    def add_waypoint(self, waypoint: Waypoint):
        """Add a waypoint to the navigation route"""
        self.waypoints.append(waypoint)
        self.logger.info(f"Added waypoint: {waypoint.name}")

    def clear_waypoints(self):
        """Clear all waypoints"""
        self.waypoints.clear()
        self.current_waypoint_index = 0
        self.navigation_active = False
        self.logger.info("Cleared all waypoints")

    def start_navigation(self):
        """Start navigation to waypoints"""
        if not self.waypoints:
            self.logger.warning("No waypoints set for navigation")
            return False

        self.current_waypoint_index = 0
        self.navigation_active = True
        self.logger.info("Started navigation")
        return True

    def stop_navigation(self):
        """Stop active navigation"""
        self.navigation_active = False
        self.logger.info("Stopped navigation")

    def get_navigation_status(self) -> NavigationStatus:
        """Get current navigation status"""
        status = NavigationStatus(
            current_position=self.current_position,
            navigation_active=self.navigation_active
        )

        # Add target waypoint info if navigating
        if self.navigation_active and self.waypoints and self.current_waypoint_index < len(self.waypoints):
            target_waypoint = self.waypoints[self.current_waypoint_index]
            status.target_waypoint = target_waypoint

            if self.current_position:
                # Calculate distance and bearing
                status.distance_to_target = self._calculate_distance_to_waypoint(target_waypoint)

                bearing = calculate_bearing(
                    self.current_position.latitude,
                    self.current_position.longitude,
                    target_waypoint.coordinate.latitude,
                    target_waypoint.coordinate.longitude
                )

                status.bearing_to_target = bearing
                status.direction_to_target = angle_to_clock_direction(bearing)

        return status

    def is_gps_healthy(self) -> bool:
        """Check if GPS is providing recent fixes"""
        return (time.time() - self.last_fix_time) < self.fix_timeout

    def get_current_position(self) -> Optional[GPSCoordinate]:
        """Get current GPS position"""
        return self.current_position

    def create_waypoint_from_current_position(self, name: str, description: str = "") -> Optional[Waypoint]:
        """Create waypoint from current position"""
        if not self.current_position:
            self.logger.warning("No current position available for waypoint creation")
            return None

        waypoint = Waypoint(
            name=name,
            coordinate=self.current_position,
            description=description
        )

        self.logger.info(f"Created waypoint '{name}' at current position")
        return waypoint

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Test function
def test_gps_navigator():
    """Test function for GPS navigator"""
    from config import GPS_CONFIG

    logging.basicConfig(level=logging.INFO)

    with GPSNavigator(GPS_CONFIG) as navigator:
        print("Testing GPS navigator for 60 seconds...")

        # Wait for GPS fix
        print("Waiting for GPS fix...")
        for i in range(30):
            time.sleep(1)
            status = navigator.get_navigation_status()
            if status.current_position:
                print(f"GPS fix acquired: {status.current_position.latitude:.6f}, {status.current_position.longitude:.6f}")
                break
        else:
            print("No GPS fix acquired in 30 seconds")
            return

        # Create a test waypoint 50 meters north
        current_pos = navigator.get_current_position()
        if current_pos:
            # Calculate position ~50m north (approximate)
            target_lat = current_pos.latitude + 0.00045  # ~50m north
            target_waypoint = Waypoint(
                name="Test Target",
                coordinate=GPSCoordinate(target_lat, current_pos.longitude),
                description="Test navigation target"
            )

            navigator.add_waypoint(target_waypoint)
            navigator.start_navigation()

            print("Navigation started to test waypoint")

            # Monitor navigation for 30 seconds
            for i in range(30):
                time.sleep(1)
                status = navigator.get_navigation_status()

                if status.distance_to_target is not None:
                    print(f"Distance to target: {status.distance_to_target:.1f}m, "
                          f"Direction: {status.direction_to_target.name if status.direction_to_target else 'Unknown'}")

                if not status.navigation_active:
                    print("Navigation completed!")
                    break

if __name__ == "__main__":
    test_gps_navigator()
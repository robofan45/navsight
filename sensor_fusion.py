"""
Sensor fusion and decision logic for the mobility aid system
Combines data from ultrasonic, vision, and GPS sensors to make intelligent navigation decisions
"""

import time
import threading
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from config import SystemConfig
from utils import Direction, clock_direction_to_text, MovingAverage
from ultrasonic_sensor import UltrasonicReading, UltrasonicArray
from vision_detector import DetectedObject, VisionDetector
from gps_navigator import GPSNavigator, NavigationStatus
from audio_guide import AudioGuide, MessagePriority, MessageType

class ThreatLevel(Enum):
    """Threat assessment levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class NavigationMode(Enum):
    """Navigation modes"""
    IDLE = "idle"
    EXPLORATION = "exploration"  # Free roaming with obstacle avoidance
    WAYPOINT_NAVIGATION = "waypoint_navigation"  # Following GPS waypoints
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class SensorData:
    """Combined sensor data structure"""
    ultrasonic_readings: List[UltrasonicReading] = field(default_factory=list)
    detected_objects: List[DetectedObject] = field(default_factory=list)
    navigation_status: Optional[NavigationStatus] = None
    timestamp: float = 0.0

@dataclass
class ThreatAssessment:
    """Threat assessment for a direction"""
    direction: Direction
    threat_level: ThreatLevel
    distance: Optional[float] = None
    source: str = ""  # Source of threat (ultrasonic, vision, etc.)
    description: str = ""

@dataclass
class GuidanceCommand:
    """Guidance command for the user"""
    primary_direction: Direction
    message: str
    priority: MessagePriority
    backup_directions: List[Direction] = field(default_factory=list)
    stop_movement: bool = False

class SensorFusion:
    """Main sensor fusion and decision logic system"""

    def __init__(self, config: SystemConfig, ultrasonic_array: UltrasonicArray,
                 vision_detector: VisionDetector, gps_navigator: GPSNavigator,
                 audio_guide: AudioGuide):
        self.config = config
        self.ultrasonic_array = ultrasonic_array
        self.vision_detector = vision_detector
        self.gps_navigator = gps_navigator
        self.audio_guide = audio_guide

        self.logger = logging.getLogger('sensor_fusion')

        # Current system state
        self.navigation_mode = NavigationMode.IDLE
        self.current_sensor_data = SensorData()
        self.threat_assessments: Dict[Direction, ThreatAssessment] = {}

        # Decision making state
        self.last_guidance_time = 0.0
        self.last_guidance_direction: Optional[Direction] = None
        self.guidance_cooldown = 2.0  # Minimum seconds between guidance updates

        # Threat level tracking
        self.threat_history: Dict[Direction, MovingAverage] = {
            direction: MovingAverage(window_size=5) for direction in Direction
        }

        # Emergency state
        self.emergency_active = False
        self.last_emergency_time = 0.0

        # Processing control
        self.running = False
        self.thread = None

        # Statistics and monitoring
        self.processing_stats = {
            'total_cycles': 0,
            'emergency_stops': 0,
            'guidance_commands': 0,
            'threat_detections': 0
        }

    def start(self):
        """Start sensor fusion processing"""
        if self.running:
            self.logger.warning("Sensor fusion already running")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._processing_loop,
            name="sensor_fusion",
            daemon=True
        )
        self.thread.start()

        self.logger.info("Sensor fusion started")

        # Announce system ready
        self.audio_guide.speak_system_status("All sensors active, system ready")

    def stop(self):
        """Stop sensor fusion processing"""
        self.logger.info("Stopping sensor fusion...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.logger.info("Sensor fusion stopped")

    def _processing_loop(self):
        """Main sensor fusion processing loop"""
        while self.running:
            try:
                start_time = time.time()

                # Collect sensor data
                sensor_data = self._collect_sensor_data()

                # Assess threats from all sensors
                threat_assessments = self._assess_threats(sensor_data)

                # Make navigation decision
                guidance_command = self._make_navigation_decision(threat_assessments, sensor_data)

                # Execute guidance command
                if guidance_command:
                    self._execute_guidance(guidance_command)

                # Update statistics
                self.processing_stats['total_cycles'] += 1

                # Control processing rate
                processing_time = time.time() - start_time
                sleep_time = max(0, self.config.main_loop_interval - processing_time)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in sensor fusion loop: {e}")
                time.sleep(0.5)

    def _collect_sensor_data(self) -> SensorData:
        """Collect data from all sensors"""
        sensor_data = SensorData(timestamp=time.time())

        # Collect ultrasonic data
        sensor_data.ultrasonic_readings = self.ultrasonic_array.get_latest_readings()

        # Collect vision data
        vision_results = self.vision_detector.get_latest_detections(max_results=1)
        if vision_results:
            sensor_data.detected_objects = vision_results[0].objects

        # Collect GPS/navigation data
        sensor_data.navigation_status = self.gps_navigator.get_navigation_status()

        self.current_sensor_data = sensor_data
        return sensor_data

    def _assess_threats(self, sensor_data: SensorData) -> Dict[Direction, ThreatAssessment]:
        """Assess threats from all sensor sources"""
        threats = {}

        # Assess ultrasonic threats
        ultrasonic_threats = self._assess_ultrasonic_threats(sensor_data.ultrasonic_readings)
        threats.update(ultrasonic_threats)

        # Assess vision threats
        vision_threats = self._assess_vision_threats(sensor_data.detected_objects)
        for direction, threat in vision_threats.items():
            if direction in threats:
                # Combine with existing threat - take higher threat level
                if threat.threat_level.value > threats[direction].threat_level.value:
                    threats[direction] = threat
            else:
                threats[direction] = threat

        # Update threat history for smoothing
        for direction in Direction:
            threat_level = threats.get(direction, ThreatAssessment(direction, ThreatLevel.NONE)).threat_level
            self.threat_history[direction].add_value(threat_level.value)

        self.threat_assessments = threats
        return threats

    def _assess_ultrasonic_threats(self, readings: List[UltrasonicReading]) -> Dict[Direction, ThreatAssessment]:
        """Assess threats from ultrasonic sensor data"""
        threats = {}

        for reading in readings:
            if not reading.valid:
                continue

            # Determine threat level based on distance
            if reading.distance < self.config.immediate_danger_distance:
                threat_level = ThreatLevel.CRITICAL
                description = f"Immediate obstacle at {reading.distance:.1f}m"
            elif reading.distance < self.config.warning_distance:
                threat_level = ThreatLevel.HIGH
                description = f"Close obstacle at {reading.distance:.1f}m"
            elif reading.distance < self.config.caution_distance:
                threat_level = ThreatLevel.MEDIUM
                description = f"Obstacle at {reading.distance:.1f}m"
            else:
                threat_level = ThreatLevel.LOW
                description = f"Distant obstacle at {reading.distance:.1f}m"

            threat = ThreatAssessment(
                direction=reading.direction,
                threat_level=threat_level,
                distance=reading.distance,
                source="ultrasonic",
                description=description
            )

            # Keep highest threat per direction
            if reading.direction not in threats or threat_level.value > threats[reading.direction].threat_level.value:
                threats[reading.direction] = threat

        return threats

    def _assess_vision_threats(self, objects: List[DetectedObject]) -> Dict[Direction, ThreatAssessment]:
        """Assess threats from vision detection data"""
        threats = {}

        for obj in objects:
            # Determine threat level based on object type and distance
            threat_level = self._get_object_threat_level(obj)

            if threat_level == ThreatLevel.NONE:
                continue

            distance_str = f" at {obj.estimated_distance:.1f}m" if obj.estimated_distance else ""
            description = f"{obj.class_name}{distance_str}"

            threat = ThreatAssessment(
                direction=obj.direction,
                threat_level=threat_level,
                distance=obj.estimated_distance,
                source="vision",
                description=description
            )

            # Keep highest threat per direction
            if obj.direction not in threats or threat_level.value > threats[obj.direction].threat_level.value:
                threats[obj.direction] = threat

        return threats

    def _get_object_threat_level(self, obj: DetectedObject) -> ThreatLevel:
        """Determine threat level for detected object"""
        # Moving objects are higher threat
        moving_objects = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'}

        # Stationary obstacles
        stationary_objects = {'chair', 'table', 'potted plant', 'couch'}

        # Traffic infrastructure
        traffic_objects = {'traffic light', 'stop sign'}

        if obj.class_name in moving_objects:
            if obj.estimated_distance and obj.estimated_distance < 2.0:
                return ThreatLevel.HIGH
            elif obj.estimated_distance and obj.estimated_distance < 5.0:
                return ThreatLevel.MEDIUM
            else:
                return ThreatLevel.LOW

        elif obj.class_name in stationary_objects:
            if obj.estimated_distance and obj.estimated_distance < 1.0:
                return ThreatLevel.MEDIUM
            else:
                return ThreatLevel.LOW

        elif obj.class_name in traffic_objects:
            return ThreatLevel.LOW

        # Unknown objects - treat with caution
        if obj.estimated_distance and obj.estimated_distance < 1.5:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    def _make_navigation_decision(self, threats: Dict[Direction, ThreatAssessment],
                                 sensor_data: SensorData) -> Optional[GuidanceCommand]:
        """Make navigation decision based on threat assessment and current mode"""

        # Check for emergency situations
        if self._check_emergency_conditions(threats):
            return self._handle_emergency(threats)

        # Clear emergency state if no longer needed
        if self.emergency_active and not self._check_emergency_conditions(threats):
            self.emergency_active = False
            self.audio_guide.speak_system_status("Clear to proceed")

        # Determine navigation mode
        self._update_navigation_mode(sensor_data)

        # Generate guidance based on mode
        if self.navigation_mode == NavigationMode.EMERGENCY_STOP:
            return None  # Already handled in emergency

        elif self.navigation_mode == NavigationMode.WAYPOINT_NAVIGATION:
            return self._generate_waypoint_guidance(threats, sensor_data)

        elif self.navigation_mode == NavigationMode.EXPLORATION:
            return self._generate_exploration_guidance(threats)

        else:  # IDLE
            return self._generate_idle_guidance(threats)

    def _check_emergency_conditions(self, threats: Dict[Direction, ThreatAssessment]) -> bool:
        """Check if emergency stop is needed"""
        # Critical threat in forward direction
        forward_threats = [Direction.ELEVEN, Direction.TWELVE, Direction.ONE]
        for direction in forward_threats:
            threat = threats.get(direction)
            if threat and threat.threat_level == ThreatLevel.CRITICAL:
                return True

        # Multiple high threats surrounding user
        high_threat_count = sum(1 for threat in threats.values()
                               if threat.threat_level.value >= ThreatLevel.HIGH.value)
        if high_threat_count >= 3:
            return True

        return False

    def _handle_emergency(self, threats: Dict[Direction, ThreatAssessment]) -> GuidanceCommand:
        """Handle emergency stop situation"""
        if not self.emergency_active:
            self.emergency_active = True
            self.last_emergency_time = time.time()
            self.processing_stats['emergency_stops'] += 1

            # Find the most critical threat
            critical_threats = [t for t in threats.values() if t.threat_level == ThreatLevel.CRITICAL]
            if critical_threats:
                threat = min(critical_threats, key=lambda t: t.distance or float('inf'))
                message = f"Stop! {threat.description} {clock_direction_to_text(threat.direction)}"
            else:
                message = "Stop! Multiple obstacles detected"

            self.audio_guide.emergency_stop()

            return GuidanceCommand(
                primary_direction=Direction.TWELVE,  # Stay facing forward
                message=message,
                priority=MessagePriority.EMERGENCY,
                stop_movement=True
            )

        return None  # Emergency already active

    def _update_navigation_mode(self, sensor_data: SensorData):
        """Update current navigation mode"""
        if self.emergency_active:
            self.navigation_mode = NavigationMode.EMERGENCY_STOP
        elif sensor_data.navigation_status and sensor_data.navigation_status.navigation_active:
            self.navigation_mode = NavigationMode.WAYPOINT_NAVIGATION
        else:
            self.navigation_mode = NavigationMode.EXPLORATION

    def _generate_waypoint_guidance(self, threats: Dict[Direction, ThreatAssessment],
                                   sensor_data: SensorData) -> Optional[GuidanceCommand]:
        """Generate guidance for waypoint navigation"""
        nav_status = sensor_data.navigation_status
        if not nav_status or not nav_status.direction_to_target:
            return None

        target_direction = nav_status.direction_to_target

        # Check if target direction is safe
        target_threat = threats.get(target_direction)
        if target_threat and target_threat.threat_level.value >= ThreatLevel.HIGH.value:
            # Find alternative safe direction
            safe_directions = self._find_safe_directions(threats)
            if safe_directions:
                alt_direction = self._choose_best_alternative(safe_directions, target_direction)
                message = f"Obstacle ahead, navigate {clock_direction_to_text(alt_direction)} to avoid"
                return GuidanceCommand(
                    primary_direction=alt_direction,
                    message=message,
                    priority=MessagePriority.WARNING,
                    backup_directions=[target_direction]
                )
            else:
                # No safe directions available
                return GuidanceCommand(
                    primary_direction=Direction.TWELVE,
                    message="Path blocked, stop and reassess",
                    priority=MessagePriority.CRITICAL,
                    stop_movement=True
                )

        # Target direction is safe
        if self._should_provide_guidance():
            distance_str = f", {nav_status.distance_to_target:.0f} meters" if nav_status.distance_to_target else ""
            message = f"Navigate {clock_direction_to_text(target_direction)}{distance_str}"

            return GuidanceCommand(
                primary_direction=target_direction,
                message=message,
                priority=MessagePriority.NAVIGATION
            )

        return None

    def _generate_exploration_guidance(self, threats: Dict[Direction, ThreatAssessment]) -> Optional[GuidanceCommand]:
        """Generate guidance for free exploration mode"""
        # Find safe directions
        safe_directions = self._find_safe_directions(threats)

        if not safe_directions:
            return GuidanceCommand(
                primary_direction=Direction.TWELVE,
                message="No clear path detected",
                priority=MessagePriority.WARNING,
                stop_movement=True
            )

        # Prefer forward directions
        forward_preferences = [Direction.TWELVE, Direction.ELEVEN, Direction.ONE]
        for direction in forward_preferences:
            if direction in safe_directions:
                if self._should_provide_guidance():
                    return GuidanceCommand(
                        primary_direction=direction,
                        message=f"Path clear {clock_direction_to_text(direction)}",
                        priority=MessagePriority.INFO
                    )
                return None

        # Use any safe direction
        best_direction = safe_directions[0]
        if self._should_provide_guidance():
            return GuidanceCommand(
                primary_direction=best_direction,
                message=f"Clear path {clock_direction_to_text(best_direction)}",
                priority=MessagePriority.INFO
            )

        return None

    def _generate_idle_guidance(self, threats: Dict[Direction, ThreatAssessment]) -> Optional[GuidanceCommand]:
        """Generate guidance for idle mode (mainly obstacle alerts)"""
        # Only provide obstacle warnings in idle mode
        immediate_threats = [t for t in threats.values()
                           if t.threat_level.value >= ThreatLevel.HIGH.value]

        if immediate_threats and self._should_provide_guidance():
            closest_threat = min(immediate_threats, key=lambda t: t.distance or float('inf'))
            self.audio_guide.speak_obstacle_alert(closest_threat.direction, closest_threat.distance or 0)

        return None

    def _find_safe_directions(self, threats: Dict[Direction, ThreatAssessment]) -> List[Direction]:
        """Find directions that are safe for movement"""
        safe_directions = []

        for direction in Direction:
            threat = threats.get(direction)
            if not threat or threat.threat_level.value <= ThreatLevel.MEDIUM.value:
                safe_directions.append(direction)

        return safe_directions

    def _choose_best_alternative(self, safe_directions: List[Direction],
                                target_direction: Direction) -> Direction:
        """Choose best alternative direction from safe options"""
        if not safe_directions:
            return Direction.TWELVE

        # Find direction closest to target
        target_angle = target_direction.value
        best_direction = safe_directions[0]
        min_angle_diff = float('inf')

        for direction in safe_directions:
            angle_diff = min(
                abs(direction.value - target_angle),
                abs(direction.value - target_angle + 360),
                abs(direction.value - target_angle - 360)
            )

            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_direction = direction

        return best_direction

    def _should_provide_guidance(self) -> bool:
        """Check if enough time has passed to provide new guidance"""
        current_time = time.time()
        return (current_time - self.last_guidance_time) >= self.guidance_cooldown

    def _execute_guidance(self, command: GuidanceCommand):
        """Execute guidance command"""
        current_time = time.time()

        # Avoid duplicate guidance
        if (command.primary_direction == self.last_guidance_direction and
            (current_time - self.last_guidance_time) < self.guidance_cooldown):
            return

        # Speak guidance
        self.audio_guide.speak(
            command.message,
            command.priority,
            MessageType.NAVIGATION_INSTRUCTION
        )

        # Update state
        self.last_guidance_time = current_time
        self.last_guidance_direction = command.primary_direction
        self.processing_stats['guidance_commands'] += 1

        self.logger.info(f"Guidance: {command.message}")

    def get_current_threats(self) -> Dict[Direction, ThreatAssessment]:
        """Get current threat assessments"""
        return self.threat_assessments.copy()

    def get_navigation_mode(self) -> NavigationMode:
        """Get current navigation mode"""
        return self.navigation_mode

    def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return self.processing_stats.copy()

    def force_emergency_stop(self):
        """Force emergency stop (e.g., from external button)"""
        self.emergency_active = True
        self.navigation_mode = NavigationMode.EMERGENCY_STOP
        self.audio_guide.emergency_stop()
        self.logger.warning("Emergency stop activated manually")

    def clear_emergency(self):
        """Clear emergency state"""
        self.emergency_active = False
        self.navigation_mode = NavigationMode.IDLE
        self.audio_guide.speak_system_status("Emergency cleared, system ready")
        self.logger.info("Emergency state cleared")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Test function
def test_sensor_fusion():
    """Test function for sensor fusion (simulation)"""
    from config import ULTRASONIC_CONFIG, CAMERA_CONFIG, GPS_CONFIG, AUDIO_CONFIG, CONFIG

    logging.basicConfig(level=logging.INFO)

    # This would normally use real sensors
    print("Sensor fusion test would require real hardware sensors")
    print("See mobility_aid.py for complete system integration")

if __name__ == "__main__":
    test_sensor_fusion()
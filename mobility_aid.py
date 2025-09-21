"""
Main application controller for the Mobility Aid System
Coordinates all subsystems and provides the primary user interface
"""

import time
import signal
import sys
import argparse
import json
from typing import Optional, Dict, Any
import logging
from pathlib import Path

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available - running in simulation mode")

from config import (CONFIG, ULTRASONIC_CONFIG, CAMERA_CONFIG,
                   GPS_CONFIG, AUDIO_CONFIG, SystemConfig)
from utils import setup_logging
from ultrasonic_sensor import UltrasonicArray
from vision_detector import VisionDetector
from gps_navigator import GPSNavigator, Waypoint, GPSCoordinate
from audio_guide import AudioGuide
from sensor_fusion import SensorFusion, NavigationMode

class MobilityAid:
    """Main mobility aid system controller"""

    def __init__(self, config: SystemConfig = None):
        """Initialize the mobility aid system"""
        self.config = config or CONFIG

        # Setup logging first
        self.logger = setup_logging(self.config.log_level, self.config.log_file)
        self.logger.info("Initializing Mobility Aid System")

        # System state
        self.running = False
        self.initialization_complete = False

        # Subsystem instances
        self.ultrasonic_array: Optional[UltrasonicArray] = None
        self.vision_detector: Optional[VisionDetector] = None
        self.gps_navigator: Optional[GPSNavigator] = None
        self.audio_guide: Optional[AudioGuide] = None
        self.sensor_fusion: Optional[SensorFusion] = None

        # GPIO setup for status indicators
        self.status_led_pin = self.config.status_led_pin
        self.emergency_button_pin = self.config.emergency_button_pin

        # System monitoring
        self.start_time = time.time()
        self.last_health_check = 0.0
        self.health_check_interval = 30.0  # seconds

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._setup_gpio()

    def _setup_gpio(self):
        """Setup GPIO pins for status LED and emergency button"""
        if not GPIO_AVAILABLE:
            self.logger.warning("GPIO not available - skipping GPIO setup")
            return

        try:
            GPIO.setmode(GPIO.BCM)

            # Status LED
            GPIO.setup(self.status_led_pin, GPIO.OUT)
            GPIO.output(self.status_led_pin, GPIO.LOW)

            # Emergency button
            GPIO.setup(self.emergency_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                self.emergency_button_pin,
                GPIO.FALLING,
                callback=self._emergency_button_callback,
                bouncetime=300
            )

            self.logger.info("GPIO setup complete")

        except Exception as e:
            self.logger.error(f"Failed to setup GPIO: {e}")

    def _emergency_button_callback(self, channel):
        """Handle emergency button press"""
        self.logger.warning("Emergency button pressed!")
        if self.sensor_fusion:
            self.sensor_fusion.force_emergency_stop()

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

    def initialize(self) -> bool:
        """Initialize all subsystems"""
        self.logger.info("Starting system initialization...")

        try:
            # Set status LED to indicate initialization
            self._set_status_led(True)

            # Initialize audio guide first for user feedback
            self.logger.info("Initializing audio guide...")
            self.audio_guide = AudioGuide(AUDIO_CONFIG)
            self.audio_guide.start()
            self.audio_guide.speak_system_status("System starting up")

            # Initialize ultrasonic sensors
            self.logger.info("Initializing ultrasonic sensors...")
            self.ultrasonic_array = UltrasonicArray(ULTRASONIC_CONFIG)
            self.ultrasonic_array.start()

            # Initialize vision detection
            self.logger.info("Initializing vision system...")
            self.vision_detector = VisionDetector(CAMERA_CONFIG)
            self.vision_detector.start()

            # Initialize GPS navigation
            self.logger.info("Initializing GPS navigation...")
            self.gps_navigator = GPSNavigator(GPS_CONFIG)
            self.gps_navigator.start()

            # Wait a moment for sensors to stabilize
            time.sleep(2)

            # Initialize sensor fusion (requires all other systems)
            self.logger.info("Initializing sensor fusion...")
            self.sensor_fusion = SensorFusion(
                self.config,
                self.ultrasonic_array,
                self.vision_detector,
                self.gps_navigator,
                self.audio_guide
            )
            self.sensor_fusion.start()

            self.initialization_complete = True
            self.logger.info("System initialization complete")

            # Flash status LED to indicate ready state
            self._flash_status_led(3)

            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.audio_guide.speak_error("System initialization failed")
            return False

    def start(self) -> bool:
        """Start the mobility aid system"""
        if not self.initialize():
            return False

        self.running = True
        self.logger.info("Mobility Aid System started successfully")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()

        return True

    def _main_loop(self):
        """Main system monitoring loop"""
        while self.running:
            try:
                current_time = time.time()

                # Periodic health checks
                if current_time - self.last_health_check > self.health_check_interval:
                    self._perform_health_check()
                    self.last_health_check = current_time

                # Monitor system status
                self._update_status_indicators()

                # Sleep for a short interval
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1.0)

    def _perform_health_check(self):
        """Perform system health check"""
        self.logger.debug("Performing system health check...")

        health_status = {
            'ultrasonic': False,
            'vision': False,
            'gps': False,
            'audio': False,
            'fusion': False
        }

        # Check ultrasonic sensors
        if self.ultrasonic_array and self.ultrasonic_array.running:
            recent_readings = self.ultrasonic_array.get_latest_readings(max_readings=5)
            health_status['ultrasonic'] = len(recent_readings) > 0

        # Check vision system
        if self.vision_detector and self.vision_detector.running:
            recent_detections = self.vision_detector.get_latest_detections(max_results=1)
            health_status['vision'] = len(recent_detections) > 0

        # Check GPS
        if self.gps_navigator:
            health_status['gps'] = self.gps_navigator.is_gps_healthy()

        # Check audio
        if self.audio_guide and self.audio_guide.running:
            health_status['audio'] = not self.audio_guide.is_speaking() or self.audio_guide.get_queue_size() < 5

        # Check sensor fusion
        if self.sensor_fusion and self.sensor_fusion.running:
            health_status['fusion'] = True

        # Log health status
        failed_systems = [name for name, status in health_status.items() if not status]
        if failed_systems:
            self.logger.warning(f"Health check failed for: {', '.join(failed_systems)}")
            if self.audio_guide:
                self.audio_guide.speak_system_status(f"Warning: {failed_systems[0]} system issue")
        else:
            self.logger.debug("All systems healthy")

    def _update_status_indicators(self):
        """Update visual status indicators"""
        if not GPIO_AVAILABLE:
            return

        try:
            # Solid LED when running normally
            if self.running and self.initialization_complete:
                if self.sensor_fusion and self.sensor_fusion.emergency_active:
                    # Flash rapidly during emergency
                    self._flash_status_led_async(0.2)
                else:
                    # Steady on when operational
                    GPIO.output(self.status_led_pin, GPIO.HIGH)
            else:
                # Slow flash during startup
                self._flash_status_led_async(1.0)

        except Exception as e:
            self.logger.debug(f"Error updating status indicators: {e}")

    def _set_status_led(self, state: bool):
        """Set status LED state"""
        if GPIO_AVAILABLE:
            try:
                GPIO.output(self.status_led_pin, GPIO.HIGH if state else GPIO.LOW)
            except Exception:
                pass

    def _flash_status_led(self, count: int, interval: float = 0.5):
        """Flash status LED a specific number of times"""
        if not GPIO_AVAILABLE:
            return

        for _ in range(count):
            self._set_status_led(True)
            time.sleep(interval)
            self._set_status_led(False)
            time.sleep(interval)

    def _flash_status_led_async(self, interval: float):
        """Flash status LED based on current time"""
        if not GPIO_AVAILABLE:
            return

        flash_state = int(time.time() / interval) % 2 == 0
        self._set_status_led(flash_state)

    def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("Shutting down Mobility Aid System...")
        self.running = False

        # Announce shutdown
        if self.audio_guide and self.audio_guide.running:
            self.audio_guide.speak_system_status("System shutting down")
            time.sleep(2)  # Give time for audio to complete

        # Stop all subsystems in reverse order
        if self.sensor_fusion:
            self.sensor_fusion.stop()

        if self.gps_navigator:
            self.gps_navigator.stop()

        if self.vision_detector:
            self.vision_detector.stop()

        if self.ultrasonic_array:
            self.ultrasonic_array.stop()

        if self.audio_guide:
            self.audio_guide.stop()

        # Cleanup GPIO
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except Exception as e:
                self.logger.debug(f"GPIO cleanup error: {e}")

        # Calculate uptime
        uptime = time.time() - self.start_time
        self.logger.info(f"System shutdown complete. Uptime: {uptime:.1f} seconds")

    def add_waypoint(self, name: str, lat: float, lon: float, description: str = "") -> bool:
        """Add a navigation waypoint"""
        if not self.gps_navigator:
            self.logger.error("GPS navigator not available")
            return False

        try:
            waypoint = Waypoint(
                name=name,
                coordinate=GPSCoordinate(lat, lon),
                description=description
            )
            self.gps_navigator.add_waypoint(waypoint)

            if self.audio_guide:
                self.audio_guide.speak_system_status(f"Waypoint {name} added")

            self.logger.info(f"Added waypoint: {name} at {lat:.6f}, {lon:.6f}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add waypoint: {e}")
            return False

    def start_navigation(self) -> bool:
        """Start GPS navigation to waypoints"""
        if not self.gps_navigator:
            self.logger.error("GPS navigator not available")
            return False

        success = self.gps_navigator.start_navigation()
        if success and self.audio_guide:
            self.audio_guide.speak_system_status("Navigation started")

        return success

    def stop_navigation(self):
        """Stop GPS navigation"""
        if self.gps_navigator:
            self.gps_navigator.stop_navigation()
            if self.audio_guide:
                self.audio_guide.speak_system_status("Navigation stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.running,
            'initialized': self.initialization_complete,
            'uptime': time.time() - self.start_time,
            'subsystems': {}
        }

        if self.ultrasonic_array:
            status['subsystems']['ultrasonic'] = {
                'running': self.ultrasonic_array.running,
                'sensor_count': len(self.ultrasonic_array.sensors)
            }

        if self.vision_detector:
            status['subsystems']['vision'] = {
                'running': self.vision_detector.running,
                'frame_counter': self.vision_detector.frame_counter
            }

        if self.gps_navigator:
            nav_status = self.gps_navigator.get_navigation_status()
            status['subsystems']['gps'] = {
                'healthy': self.gps_navigator.is_gps_healthy(),
                'navigation_active': nav_status.navigation_active,
                'current_position': nav_status.current_position
            }

        if self.audio_guide:
            status['subsystems']['audio'] = {
                'running': self.audio_guide.running,
                'speaking': self.audio_guide.is_speaking(),
                'queue_size': self.audio_guide.get_queue_size()
            }

        if self.sensor_fusion:
            fusion_stats = self.sensor_fusion.get_processing_stats()
            status['subsystems']['fusion'] = {
                'running': self.sensor_fusion.running,
                'mode': self.sensor_fusion.get_navigation_mode().value,
                'emergency_active': self.sensor_fusion.emergency_active,
                'stats': fusion_stats
            }

        return status

def load_config_file(config_path: str) -> SystemConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Create config object with loaded data
        config = SystemConfig()
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        logging.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logging.error(f"Failed to load config file {config_path}: {e}")
        logging.info("Using default configuration")
        return SystemConfig()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Mobility Aid System")
    parser.add_argument('--config', '-c', type=str, help="Configuration file path")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Log level")
    parser.add_argument('--simulate', action='store_true',
                       help="Run in simulation mode (no hardware required)")
    parser.add_argument('--test', action='store_true',
                       help="Run system tests")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        config = SystemConfig()

    # Set log level
    config.log_level = getattr(logging, args.log_level)

    if args.simulate:
        print("Running in simulation mode - hardware sensors will be mocked")

    if args.test:
        print("Running system tests...")
        # Add test routines here
        return

    # Create and start mobility aid system
    mobility_aid = MobilityAid(config)

    try:
        print("Starting Mobility Aid System...")
        print("Press Ctrl+C to shutdown")

        success = mobility_aid.start()

        if not success:
            print("Failed to start system")
            return 1

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
        return 1
    finally:
        mobility_aid.shutdown()

    print("System shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
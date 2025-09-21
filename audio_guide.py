"""
Audio guidance system for spoken directions and alerts
Supports multiple TTS engines with priority-based messaging
"""

import time
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from queue import PriorityQueue, Queue, Empty
from enum import Enum, IntEnum
import logging

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available")

try:
    import subprocess
    ESPEAK_AVAILABLE = True
except ImportError:
    ESPEAK_AVAILABLE = False

from config import AudioConfig
from utils import Direction, clock_direction_to_text

class MessagePriority(IntEnum):
    """Priority levels for audio messages (lower number = higher priority)"""
    EMERGENCY = 1      # Immediate danger
    CRITICAL = 2       # Important safety alerts
    WARNING = 3        # Caution messages
    NAVIGATION = 4     # Turn-by-turn directions
    INFO = 5          # General information
    STATUS = 6        # System status updates

class MessageType(Enum):
    """Types of audio messages"""
    OBSTACLE_ALERT = "obstacle_alert"
    NAVIGATION_INSTRUCTION = "navigation_instruction"
    SYSTEM_STATUS = "system_status"
    OBJECT_DETECTION = "object_detection"
    GPS_STATUS = "gps_status"
    BATTERY_STATUS = "battery_status"
    ERROR_ALERT = "error_alert"

@dataclass
class AudioMessage:
    """Audio message data structure"""
    text: str
    priority: MessagePriority
    message_type: MessageType
    timestamp: float
    interrupt_current: bool = False
    repeat_count: int = 1

    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

class TTSEngine:
    """Base class for TTS engines"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = logging.getLogger('tts_engine')

    def speak(self, text: str) -> bool:
        """Speak text. Return True if successful."""
        raise NotImplementedError

    def stop(self):
        """Stop current speech"""
        raise NotImplementedError

    def set_rate(self, rate: int):
        """Set speech rate"""
        pass

    def set_volume(self, volume: float):
        """Set speech volume"""
        pass

class Pyttsx3Engine(TTSEngine):
    """pyttsx3 TTS engine implementation"""

    def __init__(self, config: AudioConfig):
        super().__init__(config)

        if not PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 not available")

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', config.voice_rate)
            self.engine.setProperty('volume', config.voice_volume)

            # Try to set a clear voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available, as they're often clearer
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)

            self.logger.info("Initialized pyttsx3 TTS engine")

        except Exception as e:
            self.logger.error(f"Failed to initialize pyttsx3: {e}")
            raise

    def speak(self, text: str) -> bool:
        """Speak text using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"Error speaking text: {e}")
            return False

    def stop(self):
        """Stop current speech"""
        try:
            self.engine.stop()
        except Exception as e:
            self.logger.debug(f"Error stopping speech: {e}")

    def set_rate(self, rate: int):
        """Set speech rate"""
        try:
            self.engine.setProperty('rate', rate)
        except Exception as e:
            self.logger.debug(f"Error setting rate: {e}")

    def set_volume(self, volume: float):
        """Set speech volume"""
        try:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
        except Exception as e:
            self.logger.debug(f"Error setting volume: {e}")

class EspeakEngine(TTSEngine):
    """espeak TTS engine implementation"""

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self.current_process = None
        self.logger.info("Initialized espeak TTS engine")

    def speak(self, text: str) -> bool:
        """Speak text using espeak"""
        try:
            # Stop any current speech
            self.stop()

            # Build espeak command
            cmd = [
                'espeak',
                '-s', str(self.config.voice_rate),
                '-a', str(int(self.config.voice_volume * 200)),  # espeak uses 0-200
                '-v', 'en',  # English voice
                text
            ]

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Wait for completion
            self.current_process.wait()
            self.current_process = None

            return True

        except Exception as e:
            self.logger.error(f"Error speaking with espeak: {e}")
            return False

    def stop(self):
        """Stop current speech"""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=1.0)
            except Exception:
                try:
                    self.current_process.kill()
                except Exception:
                    pass
            finally:
                self.current_process = None

class AudioGuide:
    """Main audio guidance system"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = logging.getLogger('audio_guide')

        # Initialize TTS engine
        self.tts_engine = self._initialize_tts_engine()

        # Message queues
        self.message_queue = PriorityQueue(maxsize=config.priority_queue_size)
        self.speaking = False
        self.running = False

        # Threading
        self.thread = None
        self.current_message: Optional[AudioMessage] = None

        # Message filtering to prevent spam
        self.last_messages: Dict[MessageType, float] = {}
        self.message_cooldown: Dict[MessageType, float] = {
            MessageType.OBSTACLE_ALERT: 2.0,  # 2 second cooldown
            MessageType.NAVIGATION_INSTRUCTION: 1.0,
            MessageType.SYSTEM_STATUS: 5.0,
            MessageType.OBJECT_DETECTION: 3.0,
            MessageType.GPS_STATUS: 10.0,
            MessageType.BATTERY_STATUS: 30.0,
            MessageType.ERROR_ALERT: 1.0
        }

    def _initialize_tts_engine(self) -> TTSEngine:
        """Initialize the appropriate TTS engine"""
        if self.config.engine == "pyttsx3" and PYTTSX3_AVAILABLE:
            try:
                return Pyttsx3Engine(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize pyttsx3: {e}")

        # Fallback to espeak
        if ESPEAK_AVAILABLE:
            try:
                return EspeakEngine(self.config)
            except Exception as e:
                self.logger.error(f"Failed to initialize espeak: {e}")

        raise RuntimeError("No TTS engine available")

    def _audio_thread(self):
        """Main audio processing thread"""
        while self.running:
            try:
                # Get next message from queue
                try:
                    message = self.message_queue.get(timeout=0.5)
                except Empty:
                    continue

                # Check if we should interrupt current speech
                if self.speaking and message.interrupt_current:
                    self.tts_engine.stop()
                    self.speaking = False

                # Wait for current speech to finish if not interrupting
                while self.speaking and not message.interrupt_current:
                    time.sleep(0.1)

                # Speak the message
                self.current_message = message
                self.speaking = True

                for _ in range(message.repeat_count):
                    if not self.running:
                        break

                    success = self.tts_engine.speak(message.text)
                    if not success:
                        self.logger.warning(f"Failed to speak message: {message.text}")
                        break

                    # Small pause between repeats
                    if message.repeat_count > 1:
                        time.sleep(0.5)

                self.speaking = False
                self.current_message = None

                # Mark task as done
                self.message_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in audio thread: {e}")
                self.speaking = False
                time.sleep(0.1)

    def start(self):
        """Start the audio guidance system"""
        if self.running:
            self.logger.warning("Audio guide already running")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._audio_thread,
            name="audio_guide",
            daemon=True
        )
        self.thread.start()

        self.logger.info("Audio guide started")

        # Welcome message
        self.speak(
            "Mobility aid system ready",
            MessagePriority.INFO,
            MessageType.SYSTEM_STATUS
        )

    def stop(self):
        """Stop the audio guidance system"""
        self.logger.info("Stopping audio guide...")
        self.running = False

        # Stop current speech
        if self.tts_engine:
            self.tts_engine.stop()

        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Clear message queue
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except Empty:
                break

        self.logger.info("Audio guide stopped")

    def speak(self, text: str, priority: MessagePriority, message_type: MessageType,
             interrupt_current: bool = False, repeat_count: int = 1) -> bool:
        """Add message to speech queue"""

        # Check message cooldown to prevent spam
        current_time = time.time()
        last_time = self.last_messages.get(message_type, 0)
        cooldown = self.message_cooldown.get(message_type, 0)

        if current_time - last_time < cooldown and priority > MessagePriority.CRITICAL:
            self.logger.debug(f"Message filtered due to cooldown: {text}")
            return False

        # Update last message time
        self.last_messages[message_type] = current_time

        # Create message
        message = AudioMessage(
            text=text,
            priority=priority,
            message_type=message_type,
            timestamp=current_time,
            interrupt_current=interrupt_current,
            repeat_count=repeat_count
        )

        # Add to queue
        try:
            self.message_queue.put_nowait(message)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to queue message: {e}")
            return False

    def speak_obstacle_alert(self, direction: Direction, distance: float):
        """Speak obstacle alert with direction and distance"""
        direction_text = clock_direction_to_text(direction)

        if distance < 0.5:
            text = f"Immediate obstacle {direction_text}, {distance:.1f} meters"
            priority = MessagePriority.EMERGENCY
            interrupt = True
        elif distance < 1.0:
            text = f"Obstacle {direction_text}, {distance:.1f} meters"
            priority = MessagePriority.CRITICAL
            interrupt = True
        else:
            text = f"Caution, obstacle {direction_text}, {distance:.1f} meters"
            priority = MessagePriority.WARNING
            interrupt = False

        self.speak(text, priority, MessageType.OBSTACLE_ALERT, interrupt)

    def speak_navigation_instruction(self, direction: Direction, distance: Optional[float] = None):
        """Speak navigation instruction"""
        direction_text = clock_direction_to_text(direction)

        if distance:
            text = f"Navigate {direction_text}, {distance:.0f} meters"
        else:
            text = f"Navigate {direction_text}"

        self.speak(text, MessagePriority.NAVIGATION, MessageType.NAVIGATION_INSTRUCTION)

    def speak_object_detection(self, object_name: str, direction: Direction, distance: Optional[float] = None):
        """Speak object detection information"""
        direction_text = clock_direction_to_text(direction)

        if distance:
            text = f"{object_name.title()} detected {direction_text}, {distance:.1f} meters"
        else:
            text = f"{object_name.title()} detected {direction_text}"

        self.speak(text, MessagePriority.INFO, MessageType.OBJECT_DETECTION)

    def speak_gps_status(self, status: str):
        """Speak GPS status information"""
        self.speak(f"GPS {status}", MessagePriority.STATUS, MessageType.GPS_STATUS)

    def speak_system_status(self, status: str):
        """Speak system status information"""
        self.speak(status, MessagePriority.STATUS, MessageType.SYSTEM_STATUS)

    def speak_error(self, error: str):
        """Speak error message"""
        self.speak(f"Error: {error}", MessagePriority.CRITICAL, MessageType.ERROR_ALERT, interrupt_current=True)

    def emergency_stop(self):
        """Emergency stop - immediate halt with warning"""
        self.speak(
            "Emergency stop",
            MessagePriority.EMERGENCY,
            MessageType.ERROR_ALERT,
            interrupt_current=True,
            repeat_count=2
        )

    def clear_queue(self):
        """Clear all pending messages"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except Empty:
                break

        self.logger.info("Audio message queue cleared")

    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.speaking

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.message_queue.qsize()

    def set_volume(self, volume: float):
        """Set audio volume (0.0 to 1.0)"""
        self.config.voice_volume = max(0.0, min(1.0, volume))
        if self.tts_engine:
            self.tts_engine.set_volume(volume)

    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.config.voice_rate = max(50, min(300, rate))
        if self.tts_engine:
            self.tts_engine.set_rate(rate)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Test function
def test_audio_guide():
    """Test function for audio guide"""
    from config import AUDIO_CONFIG
    from utils import Direction

    logging.basicConfig(level=logging.INFO)

    with AudioGuide(AUDIO_CONFIG) as guide:
        print("Testing audio guide...")
        time.sleep(2)  # Wait for welcome message

        # Test various message types
        guide.speak_obstacle_alert(Direction.TWELVE, 0.8)
        time.sleep(3)

        guide.speak_navigation_instruction(Direction.TWO, 15)
        time.sleep(3)

        guide.speak_object_detection("person", Direction.ELEVEN, 3.2)
        time.sleep(3)

        guide.speak_gps_status("signal acquired")
        time.sleep(3)

        # Test emergency override
        guide.speak_system_status("This is a long status message that should be interrupted")
        time.sleep(1)
        guide.emergency_stop()

        time.sleep(5)
        print("Audio guide test completed")

if __name__ == "__main__":
    test_audio_guide()
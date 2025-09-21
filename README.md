# Mobility Aid System

A comprehensive multi-sensor mobility aid system for Raspberry Pi 5, designed to assist users with navigation and obstacle avoidance using ultrasonic sensors, computer vision, GPS, and audio guidance.

## Features

### Core Capabilities
- **Obstacle Detection**: Multiple ultrasonic sensors provide near-field obstacle detection (0-2.5m range)
- **Computer Vision**: YOLO-based object detection and recognition using camera module
- **GPS Navigation**: Waypoint-based navigation with turn-by-turn directions
- **Audio Guidance**: Spoken directions using clock-face directional system (12 o'clock = straight ahead)
- **Sensor Fusion**: Intelligent decision-making combining all sensor inputs
- **Safety-First Design**: Emergency stop capabilities and threat assessment

### Technical Features
- Multi-threaded real-time processing
- Priority-based audio messaging
- Configurable sensor parameters
- System health monitoring
- Graceful degradation when sensors fail
- Comprehensive logging and diagnostics

## Hardware Requirements

### Raspberry Pi 5 Setup
- Raspberry Pi 5 (4GB+ RAM recommended)
- MicroSD card (32GB+ Class 10)
- Camera Module 3 or compatible
- GPS module (USB or UART)
- Speaker or headphones
- 5V power supply (official Pi 5 power supply recommended)

### Sensors
- **Ultrasonic Sensors**: 5x HC-SR04 modules
  - Front: GPIO 23 (trigger), GPIO 24 (echo)
  - Front-left: GPIO 25 (trigger), GPIO 8 (echo)
  - Front-right: GPIO 7 (trigger), GPIO 1 (echo)
  - Left: GPIO 12 (trigger), GPIO 16 (echo)
  - Right: GPIO 20 (trigger), GPIO 21 (echo)

### Status Indicators
- Status LED: GPIO 18
- Emergency button: GPIO 16 (with pull-up resistor)

### Optional Components
- External GPS antenna for better reception
- Haptic feedback motors
- Battery pack for portable operation
- Waterproof enclosure

## Software Installation

### Automatic Setup (Recommended)
```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/your-repo/mobility-aid/main/setup.sh | bash
```

### Manual Installation

1. **Prepare Raspberry Pi OS**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo raspi-config  # Enable camera, SPI, I2C
   ```

2. **Install System Dependencies**
   ```bash
   sudo apt install -y python3-pip python3-venv python3-dev build-essential \
                       cmake pkg-config libjpeg-dev libavcodec-dev espeak \
                       gpsd gpsd-clients python3-gps portaudio19-dev
   ```

3. **Create Project Environment**
   ```bash
   mkdir -p ~/mobility_aid
   cd ~/mobility_aid
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download YOLO Model**
   ```bash
   # YOLOv8 nano model will be downloaded automatically on first run
   ```

## Configuration

### Basic Configuration
Copy and modify the example configuration:
```bash
cp example_config.json config.json
nano config.json
```

### Key Configuration Options

#### Sensor Settings
- `immediate_danger_distance`: Critical threat threshold (default: 0.3m)
- `warning_distance`: Warning threshold (default: 1.0m)
- `caution_distance`: Caution threshold (default: 2.0m)

#### Audio Settings
- `voice_rate`: Speech rate in words per minute (default: 150)
- `voice_volume`: Volume level 0.0-1.0 (default: 0.9)
- `engine`: TTS engine "pyttsx3" or "espeak" (default: pyttsx3)

#### Camera Settings
- `resolution`: Camera resolution [width, height] (default: [640, 480])
- `confidence_threshold`: YOLO detection confidence (default: 0.5)
- `processing_interval`: Vision processing rate in seconds (default: 0.2)

## Usage

### Basic Operation
```bash
# Start the system
cd ~/mobility_aid
source venv/bin/activate
python mobility_aid.py

# With custom config
python mobility_aid.py --config my_config.json

# Set log level
python mobility_aid.py --log-level DEBUG

# Run in simulation mode (no hardware)
python mobility_aid.py --simulate
```

### System Service
```bash
# Start as system service
sudo systemctl start mobility-aid

# Enable auto-start on boot
sudo systemctl enable mobility-aid

# View logs
journalctl -u mobility-aid -f

# Stop service
sudo systemctl stop mobility-aid
```

### Navigation Commands

#### Adding Waypoints
```python
# In Python script or interactive mode
from mobility_aid import MobilityAid
from gps_navigator import Waypoint, GPSCoordinate

aid = MobilityAid()
aid.initialize()

# Add waypoint by coordinates
aid.add_waypoint("Home", 40.7128, -74.0060, "Starting point")
aid.add_waypoint("Store", 40.7589, -73.9851, "Grocery store")

# Start navigation
aid.start_navigation()
```

#### Voice Commands (Future Enhancement)
- "Stop navigation"
- "Repeat last instruction"
- "What's ahead?"
- "Emergency stop"

## Safety Features

### Emergency Procedures
- **Emergency Button**: Physical button for immediate stop
- **Voice Command**: "Emergency stop" command
- **Automatic Stop**: System stops on critical obstacles
- **Fallback Modes**: Operates with reduced functionality if sensors fail

### System Monitoring
- Continuous health checks of all subsystems
- Audio alerts for system status changes
- Visual status indicators via LED
- Detailed logging for troubleshooting

## Troubleshooting

### Common Issues

#### GPS Not Working
```bash
# Check GPS device
ls /dev/ttyUSB* /dev/ttyAMA*

# Test GPS communication
sudo gpsd /dev/ttyUSB0
cgps -s

# Check service status
systemctl status gpsd
```

#### Camera Issues
```bash
# Test camera
libcamera-hello --display 0

# Check camera detection
vcgencmd get_camera

# Ensure camera is enabled
sudo raspi-config nonint do_camera 0
```

#### Audio Problems
```bash
# Test audio output
speaker-test -t wav

# Check audio devices
aplay -l

# Test TTS
espeak "Testing audio output"
```

#### Sensor Issues
```bash
# Check GPIO permissions
groups $USER  # Should include 'gpio'

# Test individual sensors
python -c "
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)  # Test trigger pin
GPIO.output(23, True)
time.sleep(0.1)
GPIO.output(23, False)
GPIO.cleanup()
"
```

### Log Analysis
```bash
# View system logs
tail -f /var/log/mobility_aid/system.log

# Filter for errors
grep -i error /var/log/mobility_aid/system.log

# Monitor in real-time
journalctl -u mobility-aid -f
```

## Development

### Code Structure
```
mobility_aid/
├── config.py              # Configuration management
├── utils.py               # Utility functions and classes
├── ultrasonic_sensor.py   # Ultrasonic sensor interface
├── vision_detector.py     # Camera and YOLO detection
├── gps_navigator.py       # GPS navigation system
├── audio_guide.py         # Audio guidance system
├── sensor_fusion.py       # Decision logic and sensor fusion
├── mobility_aid.py        # Main application controller
├── requirements.txt       # Python dependencies
├── setup.sh              # Installation script
└── README.md             # This file
```

### Testing
```bash
# Run individual component tests
python ultrasonic_sensor.py
python vision_detector.py
python gps_navigator.py
python audio_guide.py

# System integration test
python mobility_aid.py --test
```

### Adding New Features
1. Modify configuration in `config.py`
2. Implement new sensor/component modules
3. Update sensor fusion logic in `sensor_fusion.py`
4. Add audio messages in `audio_guide.py`
5. Update main controller in `mobility_aid.py`

## Performance Optimization

### Raspberry Pi 5 Optimizations
- Use 64-bit OS for better performance
- Enable GPU memory split: `gpu_mem=128`
- Use fast microSD card (Application Class 2)
- Consider USB 3.0 SSD for storage
- Monitor CPU temperature and ensure cooling

### System Tuning
```bash
# Increase GPU memory for camera/vision
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Set CPU governor to performance mode
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-powersave
```

## Support and Contributing

### Getting Help
- Check logs first: `/var/log/mobility_aid/system.log`
- Review common issues in troubleshooting section
- Test individual components to isolate problems

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with detailed description

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Disclaimer

This system is designed as an assistive technology aid. Users should:
- Always use proper mobility training and techniques
- Not rely solely on this system for navigation
- Maintain awareness of surroundings
- Follow local accessibility guidelines
- Test thoroughly in safe environments before daily use

The system is not a replacement for professional mobility training or traditional mobility aids.

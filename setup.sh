#!/bin/bash
# Setup script for Mobility Aid System on Raspberry Pi

set -e

echo "=== Mobility Aid System Setup ==="
echo "Setting up on Raspberry Pi..."

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    portaudio19-dev \
    python3-pyaudio \
    gpsd \
    gpsd-clients \
    python3-gps

# Enable camera and SPI/I2C
echo "Enabling camera and interfaces..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0

# Create project directory
PROJECT_DIR="/opt/mobility_aid"
echo "Creating project directory: $PROJECT_DIR"
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

# Copy files if running from source directory
if [ -f "mobility_aid.py" ]; then
    echo "Copying project files..."
    cp *.py $PROJECT_DIR/
    cp requirements.txt $PROJECT_DIR/
    cp example_config.json $PROJECT_DIR/config.json
fi

cd $PROJECT_DIR

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create log directory
sudo mkdir -p /var/log/mobility_aid
sudo chown $USER:$USER /var/log/mobility_aid

# Install systemd service
echo "Installing systemd service..."
sudo tee /etc/systemd/system/mobility-aid.service > /dev/null <<EOF
[Unit]
Description=Mobility Aid System
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python mobility_aid.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable mobility-aid.service

# Create udev rules for GPIO access
echo "Setting up GPIO permissions..."
sudo tee /etc/udev/rules.d/99-gpio.rules > /dev/null <<EOF
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", ACTION=="add", PROGRAM="/bin/sh -c 'chown root:gpio /sys/class/gpio/export /sys/class/gpio/unexport ; chmod 220 /sys/class/gpio/export /sys/class/gpio/unexport'"
SUBSYSTEM=="gpio", KERNEL=="gpio*", ACTION=="add", PROGRAM="/bin/sh -c 'chown root:gpio /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value ; chmod 660 /sys%p/active_low /sys%p/direction /sys%p/edge /sys%p/value'"
EOF

# Add user to required groups
sudo usermod -a -G gpio,audio,video,i2c,spi,dialout $USER

echo "=== Setup Complete ==="
echo "1. Reboot the system: sudo reboot"
echo "2. After reboot, test the system: cd $PROJECT_DIR && python mobility_aid.py --test"
echo "3. Start the service: sudo systemctl start mobility-aid"
echo "4. View logs: journalctl -u mobility-aid -f"
echo ""
echo "Hardware connections needed:"
echo "- Ultrasonic sensors on GPIO pins (see config.json)"
echo "- Camera module connected"
echo "- GPS module on /dev/ttyUSB0 (or update config)"
echo "- Speaker/headphones for audio output"
echo "- Status LED on GPIO 18"
echo "- Emergency button on GPIO 16"
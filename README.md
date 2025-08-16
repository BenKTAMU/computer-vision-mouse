# Hand Tracking Mouse Control

Control your computer's mouse using hand gestures captured by your webcam. This project uses computer vision to track your hand movements and translate them into mouse cursor movements, clicks, and drag operations.

## What It Does

- **Mouse Movement**: Point your index finger to move the cursor around the screen
- **Clicking**: Pinch your index finger and thumb together to click
- **Dragging**: Hold the pinch gesture to drag and drop items
- **Gesture Recognition**: Recognizes different hand positions for various actions
- **Real-time Control**: Provides immediate, responsive mouse control

## Features

- **Smooth Tracking**: Advanced algorithms reduce jitter and provide smooth cursor movement
- **Customizable Sensitivity**: Adjust how responsive the mouse is to your hand movements
- **Multiple Profiles**: Choose from different settings (precise, responsive, stable)
- **Hotkeys**: Use keyboard shortcuts to adjust settings on the fly
- **Performance Monitoring**: Built-in tools to optimize performance for your system

## Quick Start

### 1. Install Dependencies

Make sure you have Python installed, then install the required packages:

```bash
pip install opencv-python
pip install mediapipe
pip install pyautogui
pip install numpy
pip install psutil
```

### 2. Run the Program

Simply run the main program:

```bash
python HandMouseControl.py
```

A camera window will open showing your webcam feed with hand tracking overlays.

### 3. Basic Usage

- **Move Mouse**: Point your index finger where you want the cursor
- **Click**: Pinch index finger and thumb together, then release
- **Drag**: Hold the pinch gesture for a moment, then move your hand
- **Exit**: Press 'Q' in the camera window

## Available Tools

### Main Program
- **`HandMouseControl.py`** - The main hand tracking mouse control program

### Setup and Optimization
- **`calibration_tool.py`** - Find the best settings for your setup
- **`config.py`** - Configuration file to adjust behavior

### Advanced Features
- **`profile_manager.py`** - Manage different control profiles
- **`system_monitor.py`** - Monitor system performance
- **`gesture_trainer.py`** - Train custom hand gestures

## How to Use

### First Time Setup

1. **Run the calibration tool** to find optimal settings:
   ```bash
   python calibration_tool.py
   ```

2. **Test different values** using SPACE to cycle through options
3. **Select the best setting** with ENTER
4. **Copy the recommended values** to `config.py`

### Adjusting Settings

You can change settings in real-time using hotkeys:

- **S** - Toggle between sensitivity levels
- **M** - Turn mouse movement on/off
- **C** - Cycle through calibration presets
- **P** - Switch between different profiles
- **R** - Reset to default settings

### Creating Custom Profiles

1. **Run the profile manager**:
   ```bash
   python profile_manager.py
   ```

2. **Choose from built-in profiles**:
   - **Default**: Balanced settings for general use
   - **Precise**: High precision for detailed work
   - **Responsive**: Fast response for gaming
   - **Stable**: Reduced jitter for presentations

## Troubleshooting

### Common Issues

**Cursor too jumpy?**
- Increase smoothing in the config file
- Increase the dead zone setting
- Reduce sensitivity

**Cursor too slow?**
- Decrease smoothing
- Increase sensitivity
- Reduce the frame region

**Clicks not working?**
- Make sure your hand is well-lit
- Check that your hand is within the camera view
- Adjust click thresholds in config

**Poor performance?**
- Reduce camera resolution in config
- Close other applications using the camera
- Use a lighter detection model

### Performance Tips

- **Good lighting** - Ensure your hand is well-lit
- **Stable camera** - Keep your webcam steady
- **Hand position** - Keep your hand within the magenta control region
- **Smooth movements** - Move your hand smoothly for better tracking

## System Requirements

- **Python 3.7+**
- **Webcam** (built-in or external)
- **Windows/Mac/Linux** (tested on Windows 10)
- **Decent CPU** (for real-time processing)

## File Structure

```
HandTrackingProject/
├── HandMouseControl.py      # Main program
├── HandTrackingModule.py    # Hand detection engine
├── config.py               # Settings configuration
├── calibration_tool.py     # Setup optimization tool
├── profile_manager.py      # Profile management
├── system_monitor.py       # Performance monitoring
├── gesture_trainer.py      # Custom gesture training
└── README.md              # This file
```

## Getting Help

If you encounter issues:

1. **Check the logs** - The program creates log files in a `logs/` folder
2. **Try different profiles** - Use the profile manager to test preset configurations
3. **Adjust settings** - Modify values in `config.py` based on your needs
4. **Run calibration** - Use the calibration tool to find optimal settings

## Examples

### Basic Mouse Control
```bash
# Start the program
python HandMouseControl.py

# Use your hand to control the mouse
# Point index finger to move cursor
# Pinch to click
# Hold pinch to drag
```

### Optimize Your Setup
```bash
# Find best settings for your environment
python calibration_tool.py

# Test different profiles
python profile_manager.py

# Monitor performance
python system_monitor.py
```

## What's New

This version includes:
- **Improved accuracy** with advanced smoothing algorithms
- **Better responsiveness** through optimized processing
- **Multiple profiles** for different use cases
- **Real-time adjustments** with hotkeys
- **Performance monitoring** and optimization tools
- **Custom gesture training** capabilities

---

**Start controlling your computer with hand gestures today! 🖐️🖱️**

For questions or issues, check the configuration files or run the calibration tools to optimize for your setup.

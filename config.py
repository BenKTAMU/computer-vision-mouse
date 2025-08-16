# Hand Tracking Configuration File
# Adjust these values to fine-tune the hand tracking behavior

# Camera Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# Hand Detection Settings
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE = 0.8
MODEL_COMPLEXITY = 1  # 0=Light, 1=Full, 2=Heavy

# Mouse Control Settings
FRAME_REGION = 80  # Control area padding (smaller = more precise)
BASE_SENSITIVITY = 1.2  # Base mouse sensitivity multiplier

# Smoothing Settings
PRIMARY_SMOOTHING = 0.3  # Main smoothing factor (0-1, lower = smoother)
SECONDARY_SMOOTHING = 0.1  # Fine control smoothing
POSITION_HISTORY_LENGTH = 5  # Number of frames to keep for smoothing
DEAD_ZONE = 3  # Pixels of movement to ignore (prevents jitter)

# Click Detection Settings
CLICK_THRESHOLD = 40  # Distance to trigger click
RELEASE_THRESHOLD = 40  # Distance to release click (hysteresis)
CLICK_COOLDOWN = 15  # Frames to wait between clicks
DRAG_THRESHOLD = 0.8  # Seconds to hold for drag mode

# Visual Feedback Settings
SHOW_FPS = True
SHOW_SENSITIVITY = True
SHOW_PINCH_DISTANCE = True
SHOW_STATUS_INDICATORS = True

# Performance Settings
TARGET_FPS = 60
ENABLE_VSYNC = False

# Advanced Settings
ENABLE_DYNAMIC_SENSITIVITY = True  # Adjust sensitivity based on hand distance
ENABLE_LANDMARK_SMOOTHING = True  # Smooth hand landmarks
ENABLE_HAND_STABILITY_CHECK = True  # Check if hand is stable
ENABLE_ORIENTATION_DETECTION = True  # Detect hand orientation

# Color Settings (BGR format)
COLORS = {
    'index_finger': (255, 0, 255),    # Magenta
    'thumb': (0, 255, 255),           # Yellow
    'control_region': (255, 0, 255),  # Magenta
    'status_good': (0, 255, 0),       # Green
    'status_bad': (0, 0, 255),        # Red
    'fps': (255, 0, 0),               # Blue
    'info': (255, 255, 0)             # Cyan
}

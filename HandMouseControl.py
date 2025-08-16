import cv2
import numpy as np
import time
import HandTrackingModule as htm
import pyautogui
from collections import deque
import config
import logging
import json
import os
from datetime import datetime

class HandMouseController:
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        # Camera setup with error handling
        self.cap = self.setup_camera()
        if self.cap is None:
            raise RuntimeError("Failed to initialize camera")
        
        # Hand detector with fallback options
        self.detector = self.setup_detector()
        
        # Screen dimensions
        self.wScr, self.hScr = pyautogui.size()
        
        # Control parameters from config
        self.frameR = config.FRAME_REGION
        self.sensitivity = config.BASE_SENSITIVITY
        
        # Advanced smoothing with exponential moving average
        self.alpha = config.PRIMARY_SMOOTHING
        self.beta = config.SECONDARY_SMOOTHING
        
        # Position tracking
        self.currentX, self.currentY = 0, 0
        self.smoothedX, self.smoothedY = 0, 0
        
        # Dead zone to prevent jitter
        self.deadZone = config.DEAD_ZONE
        self.lastStableX, self.lastStableY = 0, 0
        
        # Click detection with hysteresis
        self.clickThreshold = config.CLICK_THRESHOLD
        self.releaseThreshold = config.RELEASE_THRESHOLD
        self.clickCooldown = 0
        self.clicking = False
        self.dragActive = False
        
        # Drag functionality
        self.dragStartTime = 0
        self.dragThreshold = config.DRAG_THRESHOLD
        self.dragStartX, self.dragStartY = 0, 0
        
        # Movement control
        self.movementEnabled = True
        self.toggleCooldown = 0
        
        # Performance tracking
        self.fps = 0
        self.frameCount = 0
        self.lastFpsTime = time.time()
        self.performance_metrics = {
            'avg_fps': 0,
            'detection_time': 0,
            'processing_time': 0,
            'hand_detection_rate': 0
        }
        
        # Multi-hand support
        self.hand_priority = 0  # Which hand to prioritize
        self.hand_count = 0
        
        # Gesture recognition
        self.gesture_history = deque(maxlen=10)
        self.current_gesture = "none"
        self.gesture_cooldown = 0
        
        # Error recovery
        self.consecutive_failures = 0
        self.max_failures = 5
        self.fallback_mode = False
        
        # Profile management
        self.current_profile = "default"
        self.profiles = self.load_profiles()
        
        # Hotkey system
        self.hotkeys = {
            's': self.toggle_sensitivity,
            'm': self.toggle_movement,
            'c': self.cycle_calibration,
            'p': self.cycle_profile,
            'r': self.reset_settings
        }
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        
        # Initialize position history for advanced smoothing
        self.positionHistory = deque(maxlen=config.POSITION_HISTORY_LENGTH)
        
        # Log initialization
        self.logger.info("HandMouseController initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/hand_tracking_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_camera(self):
        """Setup camera with error handling and fallback options"""
        try:
            cap = cv2.VideoCapture(0)
            cap.set(3, config.CAMERA_WIDTH)
            cap.set(4, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            if not cap.isOpened():
                self.logger.error("Failed to open camera 0, trying camera 1")
                cap = cv2.VideoCapture(1)
                cap.set(3, config.CAMERA_WIDTH)
                cap.set(4, config.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            if not cap.isOpened():
                self.logger.error("All camera attempts failed")
                return None
            
            self.logger.info("Camera initialized successfully")
            return cap
            
        except Exception as e:
            self.logger.error(f"Camera setup failed: {e}")
            return None
    
    def setup_detector(self):
        """Setup hand detector with fallback options"""
        try:
            detector = htm.handDetector(
                maxHands=config.MAX_HANDS,
                detectionCon=config.DETECTION_CONFIDENCE,
                trackCon=config.TRACKING_CONFIDENCE
            )
            self.logger.info("Hand detector initialized successfully")
            return detector
        except Exception as e:
            self.logger.error(f"Detector setup failed: {e}")
            # Fallback to basic settings
            return htm.handDetector(maxHands=1, detectionCon=0.5, trackCon=0.5)
    
    def load_profiles(self):
        """Load saved configuration profiles"""
        profiles = {
            "default": {
                "sensitivity": config.BASE_SENSITIVITY,
                "smoothing": config.PRIMARY_SMOOTHING,
                "deadzone": config.DEAD_ZONE,
                "frame_region": config.FRAME_REGION
            },
            "precise": {
                "sensitivity": 0.8,
                "smoothing": 0.5,
                "deadzone": 5,
                "frame_region": 60
            },
            "responsive": {
                "sensitivity": 1.8,
                "smoothing": 0.2,
                "deadzone": 2,
                "frame_region": 100
            }
        }
        
        # Try to load custom profiles
        try:
            if os.path.exists("profiles.json"):
                with open("profiles.json", "r") as f:
                    custom_profiles = json.load(f)
                    profiles.update(custom_profiles)
        except Exception as e:
            self.logger.warning(f"Failed to load custom profiles: {e}")
        
        return profiles
    
    def save_profile(self, name, settings):
        """Save current settings as a profile"""
        try:
            self.profiles[name] = settings
            with open("profiles.json", "w") as f:
                json.dump(self.profiles, f, indent=2)
            self.logger.info(f"Profile '{name}' saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save profile: {e}")
    
    def apply_profile(self, profile_name):
        """Apply a saved profile"""
        if profile_name in self.profiles:
            profile = self.profiles[profile_name]
            self.sensitivity = profile["sensitivity"]
            self.alpha = profile["smoothing"]
            self.deadZone = profile["deadzone"]
            self.frameR = profile["frame_region"]
            self.current_profile = profile_name
            self.logger.info(f"Applied profile: {profile_name}")
        else:
            self.logger.warning(f"Profile '{profile_name}' not found")
    
    def detect_gestures(self, lmList):
        """Advanced gesture recognition system"""
        if len(lmList) < 21:
            return "none"
        
        # Extract key points
        thumb_tip = lmList[4][1:3]
        index_tip = lmList[8][1:3]
        middle_tip = lmList[12][1:3]
        ring_tip = lmList[16][1:3]
        pinky_tip = lmList[20][1:3]
        
        # Calculate distances
        distances = {
            "pinch": np.linalg.norm(np.array(thumb_tip) - np.array(index_tip)),
            "fist": np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip)),
            "peace": np.linalg.norm(np.array(index_tip) - np.array(middle_tip)),
            "open_palm": np.linalg.norm(np.array(thumb_tip) - np.array(pinky_tip))
        }
        
        # Gesture classification
        gestures = {
            "pinch": distances["pinch"] < 30,
            "fist": distances["fist"] < 25 and distances["pinch"] > 40,
            "peace": distances["peace"] < 20 and distances["pinch"] > 40,
            "open_palm": distances["open_palm"] > 80 and distances["pinch"] > 50
        }
        
        # Determine primary gesture
        for gesture, detected in gestures.items():
            if detected:
                return gesture
        
        return "none"
    
    def handle_advanced_gestures(self, gesture):
        """Handle advanced gestures for different actions"""
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return
        
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            
            if gesture == "fist":
                # Fist gesture - toggle movement
                self.toggle_movement()
                self.gesture_cooldown = 30
            elif gesture == "peace":
                # Peace gesture - cycle sensitivity
                self.cycle_sensitivity()
                self.gesture_cooldown = 30
            elif gesture == "open_palm":
                # Open palm - reset cursor to center
                pyautogui.moveTo(self.wScr // 2, self.hScr // 2)
                self.gesture_cooldown = 60
    
    def toggle_sensitivity(self):
        """Toggle between different sensitivity levels"""
        sensitivities = [0.8, 1.2, 1.6, 2.0]
        current_index = sensitivities.index(self.sensitivity) if self.sensitivity in sensitivities else 1
        next_index = (current_index + 1) % len(sensitivities)
        self.sensitivity = sensitivities[next_index]
        self.logger.info(f"Sensitivity changed to: {self.sensitivity}")
    
    def toggle_movement(self):
        """Toggle mouse movement on/off"""
        self.movementEnabled = not self.movementEnabled
        status = "enabled" if self.movementEnabled else "disabled"
        self.logger.info(f"Movement {status}")
    
    def cycle_calibration(self):
        """Cycle through calibration presets"""
        presets = ["default", "precise", "responsive"]
        current_index = presets.index(self.current_profile) if self.current_profile in presets else 0
        next_preset = presets[(current_index + 1) % len(presets)]
        self.apply_profile(next_preset)
    
    def cycle_profile(self):
        """Cycle through available profiles"""
        profile_names = list(self.profiles.keys())
        current_index = profile_names.index(self.current_profile) if self.current_profile in profile_names else 0
        next_profile = profile_names[(current_index + 1) % len(profile_names)]
        self.apply_profile(next_profile)
    
    def reset_settings(self):
        """Reset to default settings"""
        self.apply_profile("default")
        self.logger.info("Settings reset to default")
    
    def calculate_dynamic_sensitivity(self, hand_depth):
        """Adjust sensitivity based on hand distance from camera"""
        if not config.ENABLE_DYNAMIC_SENSITIVITY:
            return self.sensitivity
            
        # Estimate depth from hand size (simplified approach)
        base_sensitivity = self.sensitivity
        if hand_depth < 0.3:  # Hand close to camera
            return base_sensitivity * 0.7  # More precise
        elif hand_depth > 0.7:  # Hand far from camera
            return base_sensitivity * 1.5  # More responsive
        return base_sensitivity
    
    def apply_advanced_smoothing(self, x, y):
        """Apply exponential moving average with momentum"""
        if not config.ENABLE_LANDMARK_SMOOTHING:
            return x, y
            
        # Add current position to history
        self.positionHistory.append((x, y))
        
        if len(self.positionHistory) < 3:
            return x, y
        
        # Primary smoothing
        smoothed_x = self.alpha * x + (1 - self.alpha) * self.smoothedX
        smoothed_y = self.alpha * y + (1 - self.alpha) * self.smoothedY
        
        # Secondary smoothing for fine control
        if len(self.positionHistory) >= 3:
            recent_avg_x = sum(pos[0] for pos in list(self.positionHistory)[-3:]) / 3
            recent_avg_y = sum(pos[1] for pos in list(self.positionHistory)[-3:]) / 3
            
            smoothed_x = self.beta * recent_avg_x + (1 - self.beta) * smoothed_x
            smoothed_y = self.beta * recent_avg_y + (1 - self.beta) * smoothed_y
        
        # Apply dead zone
        if abs(smoothed_x - self.lastStableX) < self.deadZone:
            smoothed_x = self.lastStableX
        if abs(smoothed_y - self.lastStableY) < self.deadZone:
            smoothed_y = self.lastStableY
        
        self.smoothedX, self.smoothedY = smoothed_x, smoothed_y
        self.lastStableX, self.lastStableY = smoothed_x, smoothed_y
        
        return smoothed_x, smoothed_y
    
    def map_hand_to_screen(self, hand_x, hand_y, hand_width, hand_height):
        """Map hand position to screen coordinates with improved precision"""
        # Calculate dynamic sensitivity based on hand size
        hand_depth = (hand_width * hand_height) / (config.CAMERA_WIDTH * config.CAMERA_HEIGHT)
        dynamic_sensitivity = self.calculate_dynamic_sensitivity(hand_depth)
        
        # Map coordinates with padding
        padX = self.frameR * dynamic_sensitivity
        padY = self.frameR * dynamic_sensitivity
        
        # Interpolate to screen coordinates
        screen_x = np.interp(hand_x, (self.frameR, config.CAMERA_WIDTH - padX), (0, self.wScr))
        screen_y = np.interp(hand_y, (self.frameR, config.CAMERA_HEIGHT - padY), (0, self.hScr))
        
        # Clamp to screen bounds
        screen_x = max(0, min(self.wScr, screen_x))
        screen_y = max(0, min(self.hScr, screen_y))
        
        return screen_x, screen_y
    
    def handle_click_gesture(self, index_thumb_distance):
        """Improved click detection with hysteresis"""
        if self.clickCooldown > 0:
            self.clickCooldown -= 1
        
        # Detect pinch with hysteresis
        if index_thumb_distance < self.clickThreshold and not self.clicking:
            self.clicking = True
            self.dragStartTime = time.time()
            return "click_start"
        
        elif index_thumb_distance > self.releaseThreshold and self.clicking:
            self.clicking = False
            if not self.dragActive and self.clickCooldown == 0:
                self.clickCooldown = config.CLICK_COOLDOWN
                return "click"
            elif self.dragActive:
                self.dragActive = False
                return "drag_end"
        
        # Check for drag transition
        if (self.clicking and not self.dragActive and 
            time.time() - self.dragStartTime > self.dragThreshold):
            self.dragActive = True
            self.dragStartX, self.dragStartY = self.smoothedX, self.smoothedY
            return "drag_start"
        
        return "none"
    
    def execute_mouse_action(self, action):
        """Execute mouse actions with error handling"""
        try:
            if action == "click":
                pyautogui.click()
                self.logger.debug("Mouse click executed")
            elif action == "drag_start":
                pyautogui.mouseDown()
                self.logger.debug("Mouse drag started")
            elif action == "drag_end":
                pyautogui.mouseUp()
                self.logger.debug("Mouse drag ended")
        except Exception as e:
            self.logger.error(f"Mouse action error: {e}")
    
    def update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        self.frameCount += 1
        current_time = time.time()
        
        if current_time - self.lastFpsTime >= 1.0:
            self.fps = self.frameCount
            self.performance_metrics['avg_fps'] = (
                self.performance_metrics['avg_fps'] * 0.9 + self.fps * 0.1
            )
            self.frameCount = 0
            self.lastFpsTime = current_time
    
    def draw_advanced_interface(self, img, hand_detected, action_state, gesture):
        """Draw comprehensive visual interface"""
        h, w, c = img.shape
        
        # Draw control region
        cv2.rectangle(img, (self.frameR, self.frameR), 
                     (w - self.frameR, h - self.frameR), config.COLORS['control_region'], 2)
        
        # Status indicators
        if config.SHOW_STATUS_INDICATORS:
            status_color = config.COLORS['status_good'] if hand_detected else config.COLORS['status_bad']
            cv2.circle(img, (w - 50, 50), 20, status_color, -1)
        
        # Action state display
        if action_state == "clicking":
            cv2.putText(img, "CLICKING", (w - 200, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['status_good'], 2)
        elif action_state == "dragging":
            cv2.putText(img, "DRAGGING", (w - 200, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['status_bad'], 2)
        
        # Gesture display
        if gesture != "none":
            cv2.putText(img, f"Gesture: {gesture.upper()}", (w - 200, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        
        # FPS and performance display
        if config.SHOW_FPS:
            cv2.putText(img, f"FPS: {self.fps}", (20, 50), 
                       cv2.FONT_HERSHEY_PLAIN, 2, config.COLORS['fps'], 2)
            cv2.putText(img, f"Avg FPS: {self.performance_metrics['avg_fps']:.1f}", (20, 80), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['fps'], 2)
        
        # Sensitivity and profile display
        if config.SHOW_SENSITIVITY:
            cv2.putText(img, f"Sens: {self.sensitivity:.1f}", (20, 110), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
            cv2.putText(img, f"Profile: {self.current_profile}", (20, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        
        # Hotkey help
        cv2.putText(img, "S:Sens M:Move C:Cal P:Profile R:Reset", (20, h - 20), 
                   cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
    
    def handle_hotkeys(self, key):
        """Handle keyboard hotkeys"""
        key_char = chr(key).lower()
        if key_char in self.hotkeys:
            self.hotkeys[key_char]()
    
    def run(self):
        """Main control loop with enhanced error handling"""
        self.logger.info("Starting hand tracking control loop")
        
        try:
            while True:
                loop_start = time.time()
                
                # Read camera frame
                success, img = self.cap.read()
                if not success:
                    self.consecutive_failures += 1
                    self.logger.warning(f"Failed to read camera frame: {self.consecutive_failures}")
                    
                    if self.consecutive_failures > self.max_failures:
                        self.logger.error("Too many consecutive failures, entering fallback mode")
                        self.fallback_mode = True
                    
                    continue
                
                # Reset failure counter on success
                self.consecutive_failures = 0
                
                # Process hand detection
                try:
                    img = self.detector.findHands(img)
                    lmList = self.detector.findPosition(img, draw=False)
                    
                    # Get image dimensions
                    h, w, c = img.shape
                    
                    hand_detected = len(lmList) != 0
                    action_state = "none"
                    gesture = "none"
                    
                    if hand_detected:
                        # Extract key landmarks (format: [id, x, y, confidence])
                        index_x, index_y = lmList[8][1], lmList[8][2]  # Index finger
                        thumb_x, thumb_y = lmList[4][1], lmList[4][2]  # Thumb
                        middle_x, middle_y = lmList[12][1], lmList[12][2]  # Middle finger
                        
                        # Calculate distances
                        index_thumb_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
                        middle_thumb_distance = ((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2) ** 0.5
                        
                        # Detect gestures
                        gesture = self.detect_gestures(lmList)
                        self.handle_advanced_gestures(gesture)
                        
                        # Handle mouse movement
                        if self.movementEnabled:
                            # Map hand to screen coordinates
                            screen_x, screen_y = self.map_hand_to_screen(
                                index_x, index_y, 
                                abs(index_x - middle_x), 
                                abs(index_y - middle_y)
                            )
                            
                            # Apply advanced smoothing
                            smoothed_x, smoothed_y = self.apply_advanced_smoothing(screen_x, screen_y)
                            
                            # Move mouse (inverted X for more intuitive control)
                            try:
                                pyautogui.moveTo(self.wScr - smoothed_x, smoothed_y, _pause=False)
                                self.currentX, self.currentY = smoothed_x, smoothed_y
                            except Exception as e:
                                self.logger.error(f"Mouse movement error: {e}")
                        
                        # Handle click gestures
                        action = self.handle_click_gesture(index_thumb_distance)
                        if action != "none":
                            self.execute_mouse_action(action)
                            action_state = "clicking" if action == "click_start" else "dragging"
                        
                        # Visual feedback for hand position
                        cv2.circle(img, (index_x, index_y), 15, config.COLORS['index_finger'], -1)
                        cv2.circle(img, (thumb_x, thumb_y), 10, config.COLORS['thumb'], -1)
                        
                        # Show pinch distance
                        if config.SHOW_PINCH_DISTANCE:
                            cv2.putText(img, f"Pinch: {int(index_thumb_distance)}", 
                                       (w - 200, 150), cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
                        
                        # Show additional hand information if enabled
                        if config.ENABLE_HAND_STABILITY_CHECK:
                            is_stable = self.detector.is_hand_stable(lmList)
                            cv2.putText(img, f"Stable: {is_stable}", (w - 200, 170), 
                                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
                        
                        if config.ENABLE_ORIENTATION_DETECTION:
                            orientation = self.detector.get_hand_orientation(lmList)
                            if orientation is not None:
                                cv2.putText(img, f"Angle: {orientation:.1f}°", (w - 200, 190), 
                                           cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
                    
                    else:
                        # Ensure drag is released when hand is lost
                        if self.dragActive:
                            self.execute_mouse_action("drag_end")
                            self.dragActive = False
                            self.clicking = False
                    
                    # Update performance metrics
                    self.update_performance_metrics()
                    
                    # Draw advanced interface
                    self.draw_advanced_interface(img, hand_detected, action_state, gesture)
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    continue
                
                # Display and handle input
                cv2.imshow("Hand Mouse Control", img)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key != 255:  # Handle hotkeys
                    self.handle_hotkeys(key)
                
                # Performance monitoring
                loop_time = time.time() - loop_start
                self.performance_metrics['processing_time'] = loop_time
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    try:
        controller = HandMouseController()
        controller.run()
    except Exception as e:
        print(f"Failed to start HandMouseController: {e}")
        logging.error(f"Startup failed: {e}")
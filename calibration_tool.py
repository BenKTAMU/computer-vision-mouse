import cv2
import numpy as np
import time
import HandTrackingModule as htm
import config

class HandTrackingCalibrator:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, config.CAMERA_WIDTH)
        self.cap.set(4, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        self.detector = htm.handDetector(
            maxHands=config.MAX_HANDS,
            detectionCon=config.DETECTION_CONFIDENCE,
            trackCon=config.TRACKING_CONFIDENCE
        )
        
        # Calibration parameters
        self.sensitivity = config.BASE_SENSITIVITY
        self.frame_region = config.FRAME_REGION
        self.primary_smoothing = config.PRIMARY_SMOOTHING
        self.dead_zone = config.DEAD_ZONE
        
        # Test mode
        self.test_mode = "sensitivity"  # sensitivity, smoothing, deadzone, frame_region
        self.test_values = {
            "sensitivity": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            "smoothing": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
            "deadzone": [1, 2, 3, 5, 8, 10],
            "frame_region": [40, 60, 80, 100, 120, 150]
        }
        self.current_test_index = 0
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def draw_calibration_interface(self, img, hand_detected, lmList):
        """Draw calibration interface"""
        h, w, c = img.shape
        
        # Draw control region
        cv2.rectangle(img, (self.frame_region, self.frame_region), 
                     (w - self.frame_region, h - self.frame_region), 
                     config.COLORS['control_region'], 2)
        
        # Current test mode and value
        current_value = self.test_values[self.test_mode][self.current_test_index]
        cv2.putText(img, f"Testing: {self.test_mode.upper()}", (20, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, config.COLORS['fps'], 2)
        cv2.putText(img, f"Value: {current_value}", (20, 80), 
                   cv2.FONT_HERSHEY_PLAIN, 2, config.COLORS['info'], 2)
        
        # Instructions
        cv2.putText(img, "Press SPACE to test next value", (20, 120), 
                   cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        cv2.putText(img, "Press ENTER to select current value", (20, 140), 
                   cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        cv2.putText(img, "Press TAB to change test mode", (20, 160), 
                   cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        cv2.putText(img, "Press Q to quit", (20, 180), 
                   cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        
        # FPS
        cv2.putText(img, f"FPS: {self.fps}", (w - 150, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, config.COLORS['fps'], 2)
        
        # Hand status
        if hand_detected:
            cv2.circle(img, (w - 50, 100), 20, config.COLORS['status_good'], -1)
            cv2.putText(img, "HAND DETECTED", (w - 200, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['status_good'], 2)
            
            # Show hand metrics
            if len(lmList) >= 21:
                hand_size = self.detector.get_hand_size(lmList)
                hand_orientation = self.detector.get_hand_orientation(lmList)
                is_stable = self.detector.is_hand_stable(lmList)
                
                cv2.putText(img, f"Size: {hand_size:.3f}", (w - 200, 160), 
                           cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
                if hand_orientation is not None:
                    cv2.putText(img, f"Angle: {hand_orientation:.1f}°", (w - 200, 180), 
                               cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
                cv2.putText(img, f"Stable: {is_stable}", (w - 200, 200), 
                           cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
        else:
            cv2.circle(img, (w - 50, 100), 20, config.COLORS['status_bad'], -1)
            cv2.putText(img, "NO HAND", (w - 200, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['status_bad'], 2)
        
        # Test progress
        progress = (self.current_test_index + 1) / len(self.test_values[self.test_mode])
        cv2.putText(img, f"Progress: {progress*100:.0f}%", (20, 220), 
                   cv2.FONT_HERSHEY_PLAIN, 1, config.COLORS['info'], 2)
    
    def apply_test_settings(self):
        """Apply current test settings"""
        current_value = self.test_values[self.test_mode][self.current_test_index]
        
        if self.test_mode == "sensitivity":
            self.sensitivity = current_value
        elif self.test_mode == "smoothing":
            self.primary_smoothing = current_value
        elif self.test_mode == "deadzone":
            self.dead_zone = current_value
        elif self.test_mode == "frame_region":
            self.frame_region = current_value
    
    def next_test_value(self):
        """Move to next test value"""
        self.current_test_index = (self.current_test_index + 1) % len(self.test_values[self.test_mode])
        self.apply_test_settings()
    
    def change_test_mode(self):
        """Change to next test mode"""
        modes = list(self.test_values.keys())
        current_index = modes.index(self.test_mode)
        self.test_mode = modes[(current_index + 1) % len(modes)]
        self.current_test_index = 0
        self.apply_test_settings()
    
    def save_current_settings(self):
        """Save current settings to config"""
        current_value = self.test_values[self.test_mode][self.current_test_index]
        print(f"\nSelected {self.test_mode}: {current_value}")
        print("Update your config.py file with these values:")
        print(f"BASE_SENSITIVITY = {self.sensitivity}")
        print(f"PRIMARY_SMOOTHING = {self.primary_smoothing}")
        print(f"DEAD_ZONE = {self.dead_zone}")
        print(f"FRAME_REGION = {self.frame_region}")
    
    def run(self):
        """Main calibration loop"""
        print("Hand Tracking Calibration Tool")
        print("Use this tool to find optimal settings for your setup")
        print("Move your hand around and test different values")
        
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            # Process hand detection
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)
            
            hand_detected = len(lmList) != 0
            
            # Draw calibration interface
            self.draw_calibration_interface(img, hand_detected, lmList)
            
            # Update FPS
            self.update_fps()
            
            # Display and handle input
            cv2.imshow("Hand Tracking Calibration", img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                self.next_test_value()
            elif key == 13:  # Enter key
                self.save_current_settings()
            elif key == 9:  # Tab key
                self.change_test_mode()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nCalibration complete!")

if __name__ == "__main__":
    calibrator = HandTrackingCalibrator()
    calibrator.run()

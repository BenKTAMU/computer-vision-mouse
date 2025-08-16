import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
import HandTrackingModule as htm
import config

class GestureTrainer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, config.CAMERA_WIDTH)
        self.cap.set(4, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        self.detector = htm.handDetector(
            maxHands=1,
            detectionCon=config.DETECTION_CONFIDENCE,
            trackCon=config.TRACKING_CONFIDENCE
        )
        
        # Gesture training data
        self.gestures_dir = "gestures"
        if not os.path.exists(self.gestures_dir):
            os.makedirs(self.gestures_dir)
        
        self.gestures = self.load_gestures()
        self.current_gesture = None
        self.recording = False
        self.samples = []
        self.sample_count = 0
        self.target_samples = 50
        
        # Training mode
        self.mode = "view"  # view, record, test
        self.gesture_name = ""
        
        # Recognition settings
        self.recognition_threshold = 0.8
        self.min_landmarks = 21
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def load_gestures(self):
        """Load existing gesture definitions"""
        gestures = {}
        
        try:
            for filename in os.listdir(self.gestures_dir):
                if filename.endswith('.json'):
                    gesture_path = os.path.join(self.gestures_dir, filename)
                    with open(gesture_path, 'r') as f:
                        gesture_data = json.load(f)
                        gesture_name = os.path.splitext(filename)[0]
                        gestures[gesture_name] = gesture_data
        except Exception as e:
            print(f"Failed to load gestures: {e}")
        
        return gestures
    
    def save_gesture(self, name, samples, description=""):
        """Save a trained gesture"""
        try:
            # Calculate gesture signature (average landmark positions)
            signature = self.calculate_gesture_signature(samples)
            
            gesture_data = {
                "name": name,
                "description": description,
                "created": datetime.now().isoformat(),
                "samples_count": len(samples),
                "signature": signature,
                "landmarks_count": len(signature),
                "recognition_threshold": self.recognition_threshold
            }
            
            # Save to file
            gesture_path = os.path.join(self.gestures_dir, f"{name}.json")
            with open(gesture_path, 'w') as f:
                json.dump(gesture_data, f, indent=2)
            
            # Update in memory
            self.gestures[name] = gesture_data
            print(f"Gesture '{name}' saved successfully with {len(samples)} samples")
            return True
            
        except Exception as e:
            print(f"Failed to save gesture: {e}")
            return False
    
    def calculate_gesture_signature(self, samples):
        """Calculate the signature (average) of a gesture from multiple samples"""
        if not samples:
            return []
        
        # Initialize signature with zeros
        signature = np.zeros(len(samples[0]))
        
        # Average all samples
        for sample in samples:
            signature += np.array(sample)
        
        signature /= len(samples)
        
        return signature.tolist()
    
    def extract_landmark_features(self, lmList):
        """Extract normalized landmark features for gesture recognition"""
        if len(lmList) < self.min_landmarks:
            return None
        
        features = []
        
        # Normalize by wrist position (landmark 0)
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]
        
        for landmark in lmList:
            # Normalize coordinates relative to wrist
            norm_x = (landmark[1] - wrist_x) / config.CAMERA_WIDTH
            norm_y = (landmark[2] - wrist_y) / config.CAMERA_HEIGHT
            
            features.extend([norm_x, norm_y])
        
        return features
    
    def recognize_gesture(self, lmList):
        """Recognize the current hand position as a known gesture"""
        features = self.extract_landmark_features(lmList)
        if features is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for gesture_name, gesture_data in self.gestures.items():
            signature = gesture_data["signature"]
            
            if len(features) == len(signature):
                # Calculate similarity score (inverse of Euclidean distance)
                distance = np.linalg.norm(np.array(features) - np.array(signature))
                score = 1.0 / (1.0 + distance)
                
                if score > best_score and score > self.recognition_threshold:
                    best_score = score
                    best_match = gesture_name
        
        return best_match, best_score
    
    def start_recording(self, gesture_name):
        """Start recording samples for a new gesture"""
        self.gesture_name = gesture_name
        self.recording = True
        self.samples = []
        self.sample_count = 0
        print(f"Started recording gesture: {gesture_name}")
    
    def stop_recording(self):
        """Stop recording and save the gesture"""
        if self.recording and self.samples:
            self.recording = False
            
            # Save the gesture
            if self.save_gesture(self.gesture_name, self.samples):
                print(f"Gesture '{self.gesture_name}' recorded successfully!")
                self.samples = []
                self.sample_count = 0
            else:
                print("Failed to save gesture")
    
    def record_sample(self, lmList):
        """Record a sample of the current hand position"""
        if not self.recording:
            return
        
        features = self.extract_landmark_features(lmList)
        if features:
            self.samples.append(features)
            self.sample_count += 1
            
            if self.sample_count >= self.target_samples:
                self.stop_recording()
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def draw_interface(self, img, lmList, recognized_gesture=None, confidence=0.0):
        """Draw the training interface"""
        h, w, c = img.shape
        
        # Draw mode indicator
        mode_color = (0, 255, 0) if self.mode == "view" else (0, 0, 255)
        cv2.putText(img, f"Mode: {self.mode.upper()}", (20, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, mode_color, 2)
        
        # Draw FPS
        cv2.putText(img, f"FPS: {self.fps}", (w - 150, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        # Draw instructions based on mode
        if self.mode == "view":
            cv2.putText(img, "Press R to record new gesture", (20, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press T to test recognition", (20, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press Q to quit", (20, 160), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            
            # Show recognized gesture
            if recognized_gesture:
                cv2.putText(img, f"Gesture: {recognized_gesture}", (20, 200), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {confidence:.2f}", (20, 230), 
                           cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        
        elif self.mode == "record":
            cv2.putText(img, f"Recording: {self.gesture_name}", (20, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, f"Samples: {self.sample_count}/{self.target_samples}", (20, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.putText(img, "Hold hand steady and press SPACE to record sample", (20, 160), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press ESC to cancel recording", (20, 190), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        
        elif self.mode == "test":
            cv2.putText(img, "Test Mode - Show different gestures", (20, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
            cv2.putText(img, "Press ESC to return to view mode", (20, 130), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        
        # Draw hand landmarks if available
        if lmList:
            # Draw landmarks
            for lm in lmList:
                x, y = lm[1], lm[2]
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            
            # Draw hand bounding box
            if len(lmList) >= 21:
                x_coords = [lm[1] for lm in lmList]
                y_coords = [lm[2] for lm in lmList]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                cv2.rectangle(img, (x_min-10, y_min-10), (x_max+10, y_max+10), 
                             (255, 0, 255), 2)
        
        # Draw available gestures
        gesture_y = h - 100
        cv2.putText(img, "Available gestures:", (20, gesture_y), 
                   cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        
        gesture_list = list(self.gestures.keys())
        for i, gesture in enumerate(gesture_list[:5]):  # Show first 5
            y_pos = gesture_y + 20 + (i * 20)
            cv2.putText(img, f"  {gesture}", (20, y_pos), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    
    def run(self):
        """Main training loop"""
        print("Gesture Trainer Started")
        print("Available commands:")
        print("  R - Record new gesture")
        print("  T - Test recognition")
        print("  SPACE - Record sample (in record mode)")
        print("  ESC - Cancel/return to view mode")
        print("  Q - Quit")
        
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            # Process hand detection
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)
            
            recognized_gesture = None
            confidence = 0.0
            
            if lmList:
                if self.mode == "record":
                    # Auto-record samples
                    if self.recording and len(lmList) >= self.min_landmarks:
                        self.record_sample(lmList)
                
                elif self.mode == "test":
                    # Test gesture recognition
                    recognized_gesture, confidence = self.recognize_gesture(lmList)
                
                elif self.mode == "view":
                    # Show real-time recognition
                    recognized_gesture, confidence = self.recognize_gesture(lmList)
            
            # Update FPS
            self.update_fps()
            
            # Draw interface
            self.draw_interface(img, lmList, recognized_gesture, confidence)
            
            # Display and handle input
            cv2.imshow("Gesture Trainer", img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r') and self.mode == "view":
                # Start recording
                gesture_name = input("Enter gesture name: ").strip()
                if gesture_name:
                    self.start_recording(gesture_name)
                    self.mode = "record"
            elif key == ord('t') and self.mode == "view":
                # Enter test mode
                self.mode = "test"
            elif key == ord(' ') and self.mode == "record":
                # Manual sample recording
                if lmList and len(lmList) >= self.min_landmarks:
                    self.record_sample(lmList)
            elif key == 27:  # ESC key
                if self.mode == "record":
                    # Cancel recording
                    self.recording = False
                    self.samples = []
                    self.sample_count = 0
                    print("Recording cancelled")
                self.mode = "view"
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Gesture Trainer closed")

def main():
    """Test the gesture trainer"""
    trainer = GestureTrainer()
    trainer.run()

if __name__ == "__main__":
    main()

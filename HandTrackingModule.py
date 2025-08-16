import cv2
import mediapipe as mp
import time
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
            model_complexity=1)  # Use model complexity 1 for better accuracy

        self.mpDraw = mp.solutions.drawing_utils
        
        # Add landmark smoothing
        self.landmark_history = []
        self.smoothing_factor = 0.7
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_fps = 0

    def findHands(self, img, draw=True):
        """Detect hands with improved performance"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process image
        self.results = self.hands.process(imgRGB)
        
        # Draw landmarks if requested
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Find hand landmarks with improved accuracy and smoothing"""
        lmList = []

        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                
                # Apply landmark smoothing
                smoothed_landmarks = self._smooth_landmarks(myHand.landmark)
                
                for id, lm in enumerate(smoothed_landmarks):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Add confidence score (estimated from tracking confidence)
                    confidence = self.trackCon
                    lmList.append([id, cx, cy, confidence])
                    
                    if draw:
                        # Draw landmarks with different colors for key points
                        if id in [4, 8, 12, 16, 20]:  # Fingertips
                            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
                        elif id in [3, 7, 11, 15, 19]:  # Finger PIP joints
                            cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
                        else:  # Other landmarks
                            cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
                
                # Update detection time for FPS calculation
                current_time = time.time()
                if self.last_detection_time > 0:
                    self.detection_fps = 1.0 / (current_time - self.last_detection_time)
                self.last_detection_time = current_time

        return lmList
    
    def _smooth_landmarks(self, landmarks):
        """Apply temporal smoothing to landmarks to reduce jitter"""
        if not self.landmark_history:
            self.landmark_history = [landmarks]
            return landmarks
        
        # Add current landmarks to history
        self.landmark_history.append(landmarks)
        
        # Keep only recent history for smoothing
        if len(self.landmark_history) > 5:
            self.landmark_history.pop(0)
        
        # Apply exponential smoothing
        smoothed_landmarks = []
        for i in range(len(landmarks)):
            smoothed_x = 0
            smoothed_y = 0
            smoothed_z = 0
            
            # Weight recent landmarks more heavily
            weight_sum = 0
            for j, hist_landmarks in enumerate(self.landmark_history):
                weight = self.smoothing_factor ** (len(self.landmark_history) - j - 1)
                smoothed_x += hist_landmarks[i].x * weight
                smoothed_y += hist_landmarks[i].y * weight
                smoothed_z += hist_landmarks[i].z * weight
                weight_sum += weight
            
            # Normalize
            if weight_sum > 0:
                smoothed_x /= weight_sum
                smoothed_y /= weight_sum
                smoothed_z /= weight_sum
            
            # Create smoothed landmark
            smoothed_landmark = type(landmarks[i])()
            smoothed_landmark.x = smoothed_x
            smoothed_landmark.y = smoothed_y
            smoothed_landmark.z = smoothed_z
            
            smoothed_landmarks.append(smoothed_landmark)
        
        return smoothed_landmarks
    
    def get_hand_orientation(self, lmList):
        """Get hand orientation for gesture recognition"""
        if len(lmList) < 21:
            return None
        
        # Calculate palm normal vector
        palm_center = np.array([lmList[0][1], lmList[0][2]])  # Wrist
        palm_mid = np.array([lmList[9][1], lmList[9][2]])     # Middle finger MCP
        
        # Calculate palm direction
        palm_vector = palm_mid - palm_center
        palm_angle = np.arctan2(palm_vector[1], palm_vector[0]) * 180 / np.pi
        
        return palm_angle
    
    def get_hand_size(self, lmList):
        """Get normalized hand size for depth estimation"""
        if len(lmList) < 21:
            return 0
        
        # Calculate bounding box of hand
        x_coords = [lm[1] for lm in lmList]
        y_coords = [lm[2] for lm in lmList]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Normalize by image dimensions
        return (width * height) / (640 * 480)
    
    def is_hand_stable(self, lmList, threshold=5):
        """Check if hand is stable (not moving much)"""
        if len(self.landmark_history) < 3:
            return False
        
        # Compare current position with recent history
        current_center = np.array([lmList[9][1], lmList[9][2]])  # Middle finger MCP
        
        for hist_landmarks in self.landmark_history[-3:]:
            if len(hist_landmarks) >= 10:
                hist_center = np.array([hist_landmarks[9].x * 640, hist_landmarks[9].y * 480])
                distance = np.linalg.norm(current_center - hist_center)
                if distance > threshold:
                    return False
        
        return True


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            # Show hand information
            hand_size = detector.get_hand_size(lmList)
            hand_orientation = detector.get_hand_orientation(lmList)
            is_stable = detector.is_hand_stable(lmList)
            
            cv2.putText(img, f"Hand Size: {hand_size:.3f}", (10, 120), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
            cv2.putText(img, f"Orientation: {hand_orientation:.1f}°", (10, 140), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
            cv2.putText(img, f"Stable: {is_stable}", (10, 160), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime + 0.0001)
        pTime = cTime
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        if detector.detection_fps > 0:
            cv2.putText(img, f"Detection FPS: {detector.detection_fps:.1f}", (10, 200), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hand Tracking Module is running. Press 'q' to exit.")
    main()
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################
wCam, hCam = 640, 480  # Width and height of the camera feed
####################

cap = cv2.VideoCapture(0)
cap.set(3,wCam )
cap.set(4,hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.7)


#get default audio device using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

volBar = 400

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
        cx, cy = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # Replace the current volume mapping section with this
        length = math.hypot(x2 - x1, y2 - y1)

        # Define a more sensitive range - adjust these values based on your preference
        min_length = 30  # Minimum hand distance (closed)
        max_length = 200  # Maximum hand distance (open)

        # Normalize with a tighter range for better sensitivity
        normalized_length = np.clip((length - min_length) / (max_length - min_length), 0, 1)

        # Use a square root curve for more sensitivity at lower volumes (opposite of what we had before)
        adjusted_value = normalized_length ** 0.5

        # Map the adjusted value to volume range
        vol = np.interp(adjusted_value, [0, 1], [minVol, maxVol])
        volBar = np.interp(adjusted_value, [0, 1], [400, 150])

        # Add visual feedback of current volume percentage
        volPercentage = np.interp(adjusted_value, [0, 1], [0, 100])
        cv2.putText(img, f"{int(volPercentage)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)


    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

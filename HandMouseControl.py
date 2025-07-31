import cv2
import numpy as np
import time
import HandTrackingModule as htm
import pyautogui


def constrain_to_cardinal(start_x, start_y, current_x, current_y):
    dx = current_x - start_x
    dy = current_y - start_y

    if abs(dx) > abs(dy):
        return current_x, start_y
    else:
        return start_x, current_y


# Initialize
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.8)
wScr, hScr = pyautogui.size()
frameR = 100
smoothening = 3
sensitivity = 1.5

# Previous coordinates for smoothing
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# For click detection
clickCooldown = 0
clicking = False
dragActive = False
pinchStartTime = 0
DRAG_THRESHOLD = 0.5  # Hold pinch for this many seconds to start dragging
CLICK_DISTANCE_THRESHOLD = 35

preClickX, preClickY = 0, 0
clickPositionStabilized = False
PRE_CLICK_FRAMES = 3
preClickFrames = 0

# For drag tracking
dragStartX, dragStartY = 0, 0
lastDragX, lastDragY = 0, 0





# For movement control
movementEnabled = True
toggleCooldown = 0

# For FPS calculation
pTime = 0
targetFPS = 60
frameInterval = 1.0 / targetFPS

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

while True:
    loopStartTime = time.time()

    success, img = cap.read()
    if not success:
        break

    h, w, c = img.shape
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get finger positions
        index_x, index_y = lmList[8][1], lmList[8][2]  # Index finger tip
        thumb_x, thumb_y = lmList[4][1], lmList[4][2]  # Thumb tip
        middle_x, middle_y = lmList[12][1], lmList[12][2]  # Middle finger tip

        # Draw index finger position
        cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)

        # Calculate distances
        click_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
        toggle_distance = ((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2) ** 0.5



        # Process mouse movement
        if movementEnabled:
            padX = frameR * sensitivity
            padY = frameR * sensitivity

            x3 = np.interp(index_x, (frameR, w - padX), (0, wScr * sensitivity))
            y3 = np.interp(index_y, (frameR, h - padY), (0, hScr * sensitivity))

            x3 = max(0, min(wScr, x3))
            y3 = max(0, min(hScr, y3))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            try:
                if not dragActive:
                    pyautogui.moveTo(wScr - clocX, clocY, _pause=False)
            except:
                pass

            plocX, plocY = clocX, clocY

        # Click and drag functionality
        if clickCooldown > 0:
            clickCooldown -= 1

        # Track position before pinch occurs to stabilize click position
        if click_distance > 40:  # Fingers are apart - store position for potential click
            if not clickPositionStabilized:
                preClickFrames += 1
                if preClickFrames >= PRE_CLICK_FRAMES:
                    preClickX, preClickY = clocX, clocY
                    clickPositionStabilized = True
        else:
            # Fingers close together - reset stabilization counter
            preClickFrames = 0
            if not clicking and not dragActive:
                clickPositionStabilized = False


        # Close pinch detected
        if click_distance < CLICK_DISTANCE_THRESHOLD:
            # If not already clicking or dragging, start timer
            if not clicking and not dragActive:
                pinchStartTime = time.time()
                clicking = True
                # Use pre-click position to avoid position drift
                if clickPositionStabilized:
                    # Move cursor back to pre-pinch position
                    try:
                        pyautogui.moveTo(wScr - preClickX, preClickY, _pause=False)
                        clocX, clocY = preClickX, preClickY
                        plocX, plocY = preClickX, preClickY
                    except:
                        pass

            # Remainder of your clicking/dragging code unchanged...

        # Close pinch detected
        if click_distance < CLICK_DISTANCE_THRESHOLD:
            # If not already clicking or dragging, start timer
            if not clicking and not dragActive:
                pinchStartTime = time.time()
                clicking = True

            # Already clicking, check if we should transition to dragging
            elif clicking and not dragActive and (time.time() - pinchStartTime > DRAG_THRESHOLD):
                # Convert to drag after holding pinch
                dragActive = True
                clicking = False
                dragStartX, dragStartY = clocX, clocY
                lastDragX, lastDragY = clocX, clocY
                pyautogui.mouseDown()
                cv2.putText(img, "DRAGGING", (w - 200, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            # Visual feedback
            color = (0, 0, 255) if dragActive else (0, 255, 0)
            cv2.circle(img, (index_x, index_y), 15, color, cv2.FILLED)

            # Show drag timer
            if clicking and not dragActive:
                hold_time = time.time() - pinchStartTime
                cv2.putText(img, f"Hold: {hold_time:.1f}s", (w - 200, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

            # Apply directional constraint for dragging
            if dragActive and movementEnabled:
                constrained_x, constrained_y = constrain_to_cardinal(dragStartX, dragStartY, clocX, clocY)

                # Only move if position changed significantly to avoid jitter
                if abs(constrained_x - lastDragX) > 3 or abs(constrained_y - lastDragY) > 3:
                    try:
                        pyautogui.moveTo(wScr - constrained_x, constrained_y, _pause=False)
                        lastDragX, lastDragY = constrained_x, constrained_y
                    except:
                        pass

                # Display drag direction
                direction = "HORIZONTAL" if abs(clocX - dragStartX) > abs(clocY - dragStartY) else "VERTICAL"
                cv2.putText(img, f"DRAG: {direction}", (w - 200, 110),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # Pinch released
        else:
            if clicking:
                # Quick pinch and release = click
                if not dragActive and clickCooldown == 0:
                    pyautogui.click()
                    clickCooldown = 10

            if dragActive:
                # End drag operation
                pyautogui.mouseUp()
                dragActive = False

            clicking = False

    else:
        # No hand detected - ensure drag is released
        if dragActive:
            pyautogui.mouseUp()
            dragActive = False
            clicking = False

    # Display interface elements
    cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)

    # Show FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 0.0001)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Hand Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    sleepTime = frameInterval - (time.time() - loopStartTime)
    if sleepTime > 0:
        time.sleep(sleepTime)

cap.release()
cv2.destroyAllWindows()
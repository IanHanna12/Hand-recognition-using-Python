import cv2
import websocket
from cvzone.HandTrackingModule import HandDetector
import time

# Initialize HandDetector with a lower detection confidence for faster detection
detector = HandDetector(detectionCon=0.5, maxHands=2)

# Initialize WebSocket client
ws = websocket.WebSocket()
ws.connect("ws://127.0.0.1:8080")

# Open webcam
cap = cv2.VideoCapture(0)


def is_hand_open(hand_landmarks):
    # Check if all fingers are extended
    return all(
        hand_landmarks[tip][1] < hand_landmarks[mcp][1]
        for tip, mcp in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )


def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_mcp = hand_landmarks[2]
    thumb_cmc = hand_landmarks[1]
    return (thumb_tip[1] < thumb_ip[1] < thumb_mcp[1] < thumb_cmc[1]) and all(
        hand_landmarks[tip][1] > hand_landmarks[mcp][1]
        for tip, mcp in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )


def is_fist(hand_landmarks):
    # Check if all fingers are curled inwards
    return all(
        hand_landmarks[tip][1] > hand_landmarks[mcp][1]
        for tip, mcp in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )


# State machine for rotation
rotation_state = None
last_state_change_time = time.time()

# Adjustable state durations
state_durations = {
    "up": 2.0,  # Slower duration for up rotation
    "down": 2.0  # Slower duration for down rotation
}

# Debounce settings
gesture_hold_time = 1.0  # Time to hold the gesture to confirm
last_gesture_time = time.time()
current_gesture = None

# State durations for left/right gestures
left_right_duration = 0.5  # Slower but still responsive


def set_state_duration(state, duration):
    """Set the duration for a specific state."""
    if state in state_durations:
        state_durations[state] = duration


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Find hands
    hands, img = detector.findHands(frame, flipType=False)

    if hands:
        thumbs_up_count = 0
        left_hand_open = False
        right_hand_open = False
        left_hand_fist = False
        right_hand_fist = False

        for hand in hands:
            hand_landmarks = hand['lmList']
            hand_type = hand['type']

            # Check if thumb is up
            if is_thumb_up(hand_landmarks):
                thumbs_up_count += 1

            # Check if hand is open or not
            middle_finger_tip_x = hand_landmarks[12][0]
            if is_hand_open(hand_landmarks):
                if middle_finger_tip_x < frame.shape[1] // 2:
                    left_hand_open = True
                else:
                    right_hand_open = True

            # Check if hand is a fist
            if is_fist(hand_landmarks):
                if middle_finger_tip_x < frame.shape[1] // 2:
                    left_hand_fist = True
                else:
                    right_hand_fist = True

        # If both thumbs are up, send "a" but do not skip other checks
        if thumbs_up_count == 2:
            ws.send("a")

        # Handle left and right gestures immediately but with a debounce
        current_time = time.time()
        if left_hand_fist and right_hand_open:
            if current_gesture != "left" or (current_time - last_gesture_time > left_right_duration):
                ws.send("l")  # Move left
                current_gesture = "left"
                last_gesture_time = current_time
        elif right_hand_fist and left_hand_open:
            if current_gesture != "right" or (current_time - last_gesture_time > left_right_duration):
                ws.send("r")  # Move right
                current_gesture = "right"
                last_gesture_time = current_time

        # Determine the current gesture based on hand positions for rotation
        if left_hand_open and right_hand_open:
            new_gesture = "up"
        elif left_hand_fist and right_hand_fist:
            new_gesture = "down"
        else:
            new_gesture = None

        # Debounce logic for rotation gestures
        if new_gesture != current_gesture:
            current_gesture = new_gesture
            last_gesture_time = current_time
        elif current_time - last_gesture_time > gesture_hold_time:
            rotation_state = current_gesture

        # Determine the current state based on the confirmed gesture
        if rotation_state:
            state_duration = state_durations.get(rotation_state, 1.5)
        else:
            state_duration = 1.5

        if current_time - last_state_change_time > state_duration:
            last_state_change_time = current_time

            # Send messages based on the current rotation state
            if rotation_state == "up":
                ws.send("u")  # Move up
            elif rotation_state == "down":
                ws.send("d")  # Move down

    # Show the image
    cv2.imshow('MediaPipe Hands', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Close WebSocket connection
ws.close()
cap.release()
cv2.destroyAllWindows()

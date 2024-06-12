import cv2
import websocket
from cvzone.HandTrackingModule import HandDetector
import time

# Initialize HandDetector with a higher detection confidence for better accuracy
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize WebSocket client
ws = websocket.WebSocket()
ws.connect("ws://127.0.0.1:8080")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3, 500)
cap.set(4, 400)


# Gesture detection functions
def is_hand_open(hand_landmarks):
    return all(hand_landmarks[tip][1] < hand_landmarks[mcp][1] for tip, mcp in [(8, 6), (12, 10), (16, 14), (20, 18)])


##def is_thumb_up(hand_landmarks):
##    thumb_tip, thumb_ip, thumb_mcp, thumb_cmc = hand_landmarks[4], hand_landmarks[3], hand_landmarks[2], hand_landmarks[1]
##    return (thumb_tip[1] < thumb_ip[1] < thumb_mcp[1] < thumb_cmc[1]) and is_hand_open(hand_landmarks)

def is_fist(hand_landmarks):
    return all(hand_landmarks[tip][1] > hand_landmarks[mcp][1] for tip, mcp in [(8, 6), (12, 10), (16, 14), (20, 18)])


# State machine for rotation
rotation_state = None
last_state_change_time = time.time()

# Adjustable state durations
state_durations = {"up": 0.5, "down": 0.5}
gesture_hold_time = 0.2
last_gesture_time = time.time()
current_gesture = None
left_right_duration = 1
rotate_executed = False  # Flag to ensure rotate command is executed only once per gesture


def set_state_duration(state, duration):
    if state in state_durations:
        state_durations[state] = duration


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, flipType=False)

    if hands:
        thumbs_up_count = 0
        left_hand_open, right_hand_open = False, False
        left_hand_fist, right_hand_fist = False, False

        for hand in hands:
            hand_landmarks = hand['lmList']
            middle_finger_tip_x = hand_landmarks[12][0]

            ##  if is_thumb_up(hand_landmarks):
            ##     thumbs_up_count += 1

            if is_hand_open(hand_landmarks):
                if middle_finger_tip_x < frame.shape[1] // 2:
                    left_hand_open = True
                else:
                    right_hand_open = True

            if is_fist(hand_landmarks):
                if middle_finger_tip_x < frame.shape[1] // 2:
                    left_hand_fist = True
                else:
                    right_hand_fist = True

        current_time = time.time()

        # Send rotate command immediately when both fists are closed
        if left_hand_fist and right_hand_fist and not rotate_executed:
            ws.send("d")  # Rotate the piece
            print("Rotate command sent")
            rotate_executed = True  # Set rotate flag to prevent repeated execution
            last_gesture_time = current_time

        new_gesture = None

        if left_hand_fist and right_hand_open and (
                current_gesture != "left" or current_time - last_gesture_time > left_right_duration):
            ws.send("l")  # Move the piece left
            print("Move left command sent")
            current_gesture = "left"
            last_gesture_time = current_time
            rotate_executed = False  # Reset rotate flag
        elif right_hand_fist and left_hand_open and (
                current_gesture != "right" or current_time - last_gesture_time > left_right_duration):
            ws.send("r")  # Move the piece right
            print("Move right command sent")
            current_gesture = "right"
            last_gesture_time = current_time
            rotate_executed = False  # Reset rotate flag

        if new_gesture != current_gesture:
            current_gesture = new_gesture
            last_gesture_time = current_time
        elif current_time - last_gesture_time > gesture_hold_time:
            rotation_state = current_gesture
            state_duration = state_durations.get(rotation_state, 0.5)
            if current_time - last_state_change_time > state_duration:
                last_state_change_time = current_time
                if rotation_state == "up":
                    ws.send("u")
                    print("Move up command sent")
                elif rotation_state == "down":
                    ws.send("d")
                    print("Move down command sent")

    cv2.imshow('MediaPipe Hands', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

ws.close()
cap.release()
cv2.destroyAllWindows()

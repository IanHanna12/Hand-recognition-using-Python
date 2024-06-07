import cv2
import mediapipe as mp
import websocket

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize WebSocket client
ws = websocket.WebSocket()
ws.connect("ws://127.0.0.1:8080")

# Open webcam
cap = cv2.VideoCapture(0)

def is_hand_open(hand_landmarks):
    thumb_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[
        mp_hands.HandLandmark.THUMB_MCP].x
    fingers_are_open = sum(
        [f[0].x < f[1].x for f in [(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]),
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]),
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]),
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])]])
    return thumb_is_open and fingers_are_open

def is_thumb_up(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y

def is_fist(hand_landmarks):
    thumb_is_closed = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[
        mp_hands.HandLandmark.THUMB_MCP].x
    fingers_are_closed = sum(
        [f[0].x > f[1].x for f in [(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]),
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]),
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]),
                                   (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])]])
    return thumb_is_closed and fingers_are_closed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        thumbs_up_count = 0
        left_hand_open = False
        right_hand_open = False
        left_hand_fist = False
        right_hand_fist = False

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if thumb is up
            if is_thumb_up(hand_landmarks):
                thumbs_up_count += 1

            # Check if hand is open or not
            if is_hand_open(hand_landmarks):
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x < 0.5:
                    left_hand_open = True
                elif hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > 0.5:
                    right_hand_open = True

            # Check if hand is a fist
            if is_fist(hand_landmarks):
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x < 0.5:
                    left_hand_fist = True
                elif hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > 0.5:
                    right_hand_fist = True

        # If both thumbs are up, send "a" and skip other checks
        if thumbs_up_count == 2:
            ws.send("a")
        else:
            # Send messages based on hand states
            if left_hand_open and right_hand_fist:
                ws.send("l")  # Move left
            elif right_hand_open and left_hand_fist:
                ws.send("r")  # Move right

    # Show the image
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Close WebSocket connection
ws.close()

# Release the webcam resources
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

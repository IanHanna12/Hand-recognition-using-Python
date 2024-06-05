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


def both_hands_open(multi_hand_landmarks):
    return is_hand_open(multi_hand_landmarks[0]) and is_hand_open(multi_hand_landmarks[1])


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
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if hand is open or not
            if is_hand_open(hand_landmarks):
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x < 0.5:
                    ws.send("l")  # Left hand open
            elif hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > 0.5:  # Right hand open
                ws.send("r")

                if both_hands_open(results.multi_hand_landmarks):
                    ws.send(" ")  # Both hands open

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

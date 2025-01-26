import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define the function to detect SOS gesture (e.g., raised open palm)
def is_sos_gesture(hand_landmarks):
    # Check if the hand is open (all fingers extended)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    # Check if all fingertips are higher than the respective MCP joints (indicating an open hand)
    if (thumb_tip < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y and
        index_tip < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
        middle_tip < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
        ring_tip < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
        pinky_tip < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
        return True
    return False

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check for SOS gesture
            if is_sos_gesture(hand_landmarks):
                cv2.putText(frame, 'SOS Gesture Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Trigger an alert or send a notification here

    # Display the frame
    cv2.imshow('SOS Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

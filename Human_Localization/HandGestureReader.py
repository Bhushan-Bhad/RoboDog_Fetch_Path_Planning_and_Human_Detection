import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up video capture (from webcam)
cap = cv2.VideoCapture(0)

# Initialize the hand detection model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB as Mediapipe uses RGB images
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame to detect hands
        results = hands.process(image)

        # Convert the image color back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand landmarks
                landmarks = hand_landmarks.landmark

                # Get the wrist and middle finger tip positions
                wrist = landmarks[0]   # Wrist landmark
                middle_finger_tip = landmarks[12]  # Middle finger tip landmark
                thumb_tip = landmarks[4]
                pinky_tip = landmarks[20]

                # Check hand handedness
                hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

                # Check hand is vertical: Compare y-coordinates of wrist and middle finger
                vertical = abs(wrist.x - middle_finger_tip.x) < 0.1  # Check if x-coordinates are almost aligned

                if hand_label == "Right":
                    # For right hand: thumb is on the left side of the wrist
                    palm_open = thumb_tip.x < wrist.x < pinky_tip.x
                else:
                    # For left hand: thumb is on the right side of the wrist
                    palm_open = pinky_tip.x < wrist.x < thumb_tip.x

                # Finger openness check
                fingers_open = all(
                    landmarks[tip].y < landmarks[tip - 2].y  # Compare tips and lower joints for all fingers
                    for tip in [8, 12, 16, 20]
                )

                # Final condition for the Hi gesture
                if vertical and palm_open and fingers_open:
                    cv2.putText(image, f"Hi Gesture Detected! ({hand_label} hand)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Hand Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

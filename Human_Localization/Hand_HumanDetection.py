import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO  # Make sure you have installed the ultralytics package

# Load YOLOv8 model (nano version)
model = YOLO("yolo-Weights/yolov8n.pt")  # Ensure the correct path to your YOLOv8 weights

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

        # Run YOLOv8 to detect objects (including humans)
        results = model(frame)

        # Convert the frame to RGB for Mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process YOLO results (filter only human detections)
        humans = [r for r in results if r.names[r.boxes[0].cls[0].item()] == 'person']

        # If human(s) detected, proceed with hand gesture detection
        for human in humans:
            box = human.boxes.xyxy[0].cpu().numpy()  # Get bounding box
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates

            # Draw bounding box around the detected human
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the region of interest to focus on the detected human
            roi = image[y1:y2, x1:x2]

            # Ensure that the ROI is C-contiguous
            roi_contiguous = np.ascontiguousarray(roi)

            # Detect hands in the cropped ROI (upper body region)
            results = hands.process(roi_contiguous)

            if results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract hand landmarks and perform gesture detection (similar to before)
                    landmarks = hand_landmarks.landmark
                    wrist = landmarks[0]
                    middle_finger_tip = landmarks[12]
                    thumb_tip = landmarks[4]
                    pinky_tip = landmarks[20]

                    hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                    vertical = abs(wrist.x - middle_finger_tip.x) < 0.1

                    if hand_label == "Right":
                        palm_open = thumb_tip.x < wrist.x < pinky_tip.x
                    else:
                        palm_open = pinky_tip.x < wrist.x < thumb_tip.x

                    fingers_open = all(landmarks[tip].y < landmarks[tip - 2].y for tip in [8, 12, 16, 20])

                    if vertical and palm_open and fingers_open:
                        cv2.putText(frame, f"Hi Gesture Detected! ({hand_label} hand)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with detected humans and hand gestures
        cv2.imshow('Human and Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

import platform
import ctypes

if platform.system() == "Linux":
    try:
        ctypes.cdll.LoadLibrary("/usr/lib/aarch64-linux-gnu/libgomp.so.1")
    except OSError as e:
        print(f"Error loading libgomp: {e}")

import sklearn
import mediapipe as mp
import numpy as np
import joblib
import pyautogui as pag
from tensorflow.keras.models import load_model
import cv2

model_name_rf = 'model_rf__date_time_2024_07_21__20_33_58__acc_1.0__hand__oneimage.pkl'
model_name_xgb = 'model_xgb__date_time_2024_07_21__20_30_17__acc_0.9833333333333333__hand__oneimage.pkl'
model_name_nn = 'model_nn__date_time_2024_07_21__20_35_40__loss_0.3929634690284729__acc_1.0__hand__oneimage.h5'
model = joblib.load(model_name_rf)
# model = load_model(model_name_nn)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
idx_to_class = {
    0: 'Fist',
    1: 'Closed',
    2: 'Open',
}
class_to_key = {
    'Fist': 'recall',
    'Closed': 'stop',
    'Open': 'stop',
}

current_command = None

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)
capture = cv2.VideoCapture(0)  # 0 integrated | 1 plugged

# Simplified GStreamer pipeline for testing
gst_str = (
    'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! '
    'udpsink host=192.168.0.103 port=5000'
)

# Initialize VideoWriter with the GStreamer pipeline
out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 20, (640, 480), True)

if not out.isOpened():
    print("Error: Could not open GStreamer pipeline.")
    exit()

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    height, width = frame.shape[:-1]
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_image = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    x = []

    if detected_image.multi_hand_landmarks:
        for hand_lms in detected_image.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_lms,
                                      mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(255, 0, 255), thickness=4, circle_radius=2),
                                      connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(20, 180, 90), thickness=2, circle_radius=2)
                                      )
            for lm in hand_lms.landmark:
                x.extend([lm.x, lm.y])

        x = np.array(x)  # 1D array of current landmark positions
        x_max = int(width * np.max(x[::2]))
        x_min = int(width * np.min(x[::2]))
        y_max = int(height * np.max(x[1::2]))
        y_min = int(height * np.min(x[1::2]))
        x = x[None, :]  # adds a new dimension to x to avoid input shape error
        yhat_idx = int(model.predict(x)[0])
        yhat = idx_to_class[yhat_idx]
        print(yhat)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # draws a rectangle and types the predicted class
        cv2.putText(image, f'{yhat}', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

        if current_command != yhat:  # performs the new action once
            pag.press(class_to_key[yhat])
            current_command = yhat

    # Write the frame to the GStreamer pipeline
    out.write(frame)

    cv2.imshow('Hand Gesture Reader', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
out.release()
cv2.destroyAllWindows()

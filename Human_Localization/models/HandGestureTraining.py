import mediapipe as mp  
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
subdir = 'fist' 
n_frames_save = 16
iteration_counter = n_frames_save + 1
folder_counter = 1

capture = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        image = cv2.flip(frame, 1)
        detected_image = hands.process(image)

        if detected_image.multi_hand_landmarks:
            for hand_lms in detected_image.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                                              color = (255, 0, 255), thickness = 4, circle_radius = 2),
                                              connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                                                  color = (20, 180, 90), thickness = 2, circle_radius = 2)
                                              )
        cv2.imshow('webcam', image)

        if cv2.waitKey(10) & 0xFF == ord('r'):
            seq_folder_path = os.path.join('RoboDog_Fetch_Path_Planning_and_Human_Detection/Human_Localization/models/data', subdir, f'sequence{folder_counter}')
            os.mkdir(seq_folder_path)
            folder_counter += 1
            iteration_counter = 1
        if iteration_counter < n_frames_save + 1:
            cv2.imwrite(os.path.join(seq_folder_path, F'{subdir}_sequence{folder_counter}_frame{iteration_counter}.jpg'),image)
            if iteration_counter == n_frames_save:
                print(f'Images for sequence {folder_counter -1} saved.')
            iteration_counter += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


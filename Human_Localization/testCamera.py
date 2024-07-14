import cv2

def display_camera_preview(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from webcam at index {index}")
            break

        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example: Display preview for each available camera
    for index in range(10):  # Adjust range based on the number of cameras you have
        display_camera_preview(index)

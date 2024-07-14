import cv2

# List available video devices
def list_video_devices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# Print available video devices
devices = list_video_devices()
print("Available video devices:", devices)

# Use the correct index for your external webcam
external_webcam_index = 0  # Replace with the correct index if needed

# Initialize video capture
cap = cv2.VideoCapture(external_webcam_index, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 480)

# Check if the webcam opened successfully
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {external_webcam_index}.")
    exit()

try:
    # Your remaining code here

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

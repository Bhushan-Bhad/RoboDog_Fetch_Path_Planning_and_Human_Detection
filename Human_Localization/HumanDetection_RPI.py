from ultralytics import YOLO
import cv2
import math

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
external_webcam_index = 1  # Replace with the correct index

# Initialize video capture
cap = cv2.VideoCapture(external_webcam_index, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 480)

# Check if the webcam opened successfully
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {external_webcam_index}.")
    exit()

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame.")
        break

    # Run YOLO model on the frame
    results = model(frame, stream=True)

    # Process detection results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Display class name and confidence
            cls = int(box.cls[0])
            cv2.putText(frame, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

import cv2

# Test different capture APIs
for api in [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]:
    cap = cv2.VideoCapture(1, api)

    if not cap.isOpened():
        print(f"API {api}: Error: Could not open webcam.")
    else:
        print(f"API {api}: Successfully opened webcam.")
        while True:
            ret, frame = cap.read()
            cv2.imshow(f'Test Camera (API {api})', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

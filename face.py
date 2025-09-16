from ultralytics import YOLO
import cv2

# Load YOLOv8 model with higher accuracy (YOLOv8x or YOLOv8l instead of yolov8n)
# yolov8n: nano (fast, less accurate)
# yolov8s: small
# yolov8m: medium
# yolov8l: large
# yolov8x: extra-large (most accurate, slowest)
model = YOLO('yolov8x.pt')  # You can also try 'yolov8l.pt' if performance is too slow

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process frame with YOLO model
    results = model(frame)[0]
    annotated = results.plot()

    # Display output
    cv2.imshow('YOLOv8 - High Accuracy', annotated)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import math
import time
from ultralytics import YOLO

# Load the YOLO model (using YOLOv8 nano as per manual example)
# The manual references "yolo-Weights/yolov8n.pt" [cite: 49]
model = YOLO("yolov8n.pt") 

# Class names as listed in the manual [cite: 51]
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Width
cap.set(4, 480) # Height

while True:
    success, img = cap.read()
    if not success:
        break

    # Calculate FPS for performance monitoring
    start_time = time.time()
    
    # Run YOLO inference [cite: 69]
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box Coordinates [cite: 87]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw Rectangle [cite: 91]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence Score [cite: 93]
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class Name [cite: 95]
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Display Text [cite: 104]
            label = f'{currentClass} {confidence}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # FPS Calculation
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLO Webcam', img)
    
    # Press 'q' to quit [cite: 42]
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
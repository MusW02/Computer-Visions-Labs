import cv2
import numpy as np
import time
import os
import torch
import tensorflow as tf
import tensorflow_hub as hub
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

# --- SETUP: Load Class Names ---
# This matches the COCO dataset used by all three models
classes = []
if os.path.exists("coco.names"):
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
else:
    # Fallback if file is missing
    classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light']

# --- MODEL 1: YOLOv3 (OpenCV) ---
def load_yolo():
    print("Loading YOLOv3...")
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_yolo(net, output_layers, img):
    height, width, _ = img.shape
    # YOLO uses 1/255 scaling and 416x416 input
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        # i = i[0] # Uncomment if using older OpenCV versions
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"YOLO: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# --- MODEL 2: SSD (TensorFlow) ---
def load_ssd():
    print("Loading SSD...")
    return hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

def detect_ssd(model, img):
    # Convert to Tensor
    input_tensor = tf.convert_to_tensor([img])
    input_tensor = tf.image.convert_image_dtype(input_tensor, tf.uint8)
    detections = model(input_tensor)
    
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    h, w, _ = img.shape
    
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (int(xmin*w), int(xmax*w), int(ymin*h), int(ymax*h))
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img, f"SSD: {scores[i]:.2f}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

# --- MODEL 3: RetinaNet (PyTorch) ---
def load_retinanet():
    print("Loading RetinaNet...")
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights)
    model.eval()
    return model, weights.transforms()

def detect_retinanet(model, transforms, img):
    # Transform input
    input_tensor = transforms(torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
        
    for i in range(len(prediction[0]['scores'])):
        if prediction[0]['scores'][i] > 0.5:
            x1, y1, x2, y2 = prediction[0]['boxes'][i].numpy().astype("int")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "RetinaNet", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize Models
try:
    net_yolo, out_yolo = load_yolo()
    yolo_ok = True
except:
    print("YOLO files missing! Skipping YOLO.")
    yolo_ok = False

model_ssd = load_ssd()
model_retina, transform_retina = load_retinanet()

mode = "ssd" # Default start mode

print("\n--- CONTROLS ---")
print("Press 'y' for YOLOv3")
print("Press 's' for SSD")
print("Press 'r' for RetinaNet")
print("Press 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    start = time.time()
    
    if mode == "yolo" and yolo_ok:
        frame = detect_yolo(net_yolo, out_yolo, frame)
    elif mode == "retina":
        frame = detect_retinanet(model_retina, transform_retina, frame)
    else:
        # SSD requires RGB conversion
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = detect_ssd(model_ssd, rgb)
        frame = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        mode = "ssd"

    end = time.time()
    fps = 1 / (end - start)
    
    # Dashboard
    cv2.rectangle(frame, (0,0), (200, 80), (0,0,0), -1)
    cv2.putText(frame, f"Model: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Task 2 Comparison", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    elif key == ord('y'): mode = "yolo"
    elif key == ord('s'): mode = "ssd"
    elif key == ord('r'): mode = "retina"

cap.release()
cv2.destroyAllWindows()
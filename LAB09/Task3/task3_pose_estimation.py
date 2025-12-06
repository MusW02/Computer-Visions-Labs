import cv2
import mediapipe as mp
import math
import numpy as np

# 1. Initialize MediaPipe Pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# 2. Initialize Drawing Utils for the skeleton
mp_drawing = mp.solutions.drawing_utils

# 3. Access Webcam
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (Shoulder, Elbow, Wrist).
    Requirement: Optional Task 4
    """
    a = np.array([a.x, a.y]) # Shoulder
    b = np.array([b.x, b.y]) # Elbow
    c = np.array([c.x, c.y]) # Wrist
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

print("[INFO] Starting Pose Estimation. Press 'q' to quit.")

while True:
    # 4. Read Frame
    success, img = cap.read()
    if not success:
        break

    # Convert BGR (OpenCV standard) to RGB (MediaPipe requirement)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 5. Apply Pose Estimation
    results = pose.process(img_rgb)
    
    # 6. Draw Skeletons
    if results.pose_landmarks:
        # Draw the connections (lines and dots)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # --- Logic for Optional Joint Angle (Elbow) ---
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for Right Arm
            # Index 12=Right Shoulder, 14=Right Elbow, 16=Right Wrist
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Calculate the angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize the angle on screen
            h, w, _ = img.shape
            cx, cy = int(elbow.x * w), int(elbow.y * h)
            
            # Draw a dashboard rectangle for the angle
            cv2.putText(img, str(int(angle)), (cx, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Simple logic: Check if arm is "UP" or "DOWN"
            status = "Arm Flexed" if angle < 90 else "Arm Straight"
            cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except:
            pass

    # Show result
    cv2.imshow('Lab 9 Task 3: Pose Estimation', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import RTDETR

def run_rtdetr_detection(source=0, conf_threshold=0.5):
    """
    Runs real-time object detection using RT-DETR.
    Args:
        source: 0 for webcam, or path to a video file (e.g., 'traffic.mp4')
    """
    # Load the pre-trained RT-DETR model (large version)
    # It will download automatically if not present
    model = RTDETR("rtdetr-l.pt") 
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    print("Starting RT-DETR... Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run inference on the frame
        results = model(frame, conf=conf_threshold, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("RT-DETR Real-Time Detection", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rtdetr_detection('east.mp4')
    
    
import cv2
from ultralytics import YOLO

model = YOLO("detection_yolo/best.pt").to("cpu")
model2 = YOLO("classification_yolo/best.pt").to("cpu")
# Open the video file
video_path = "example.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        if results[0].boxes.xyxy.nelement() != 0:
            results1 = model2(annotated_frame, conf=0.5)
            annotated_frame1 = results1[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame1)
        # Display the annotated frame
        else:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

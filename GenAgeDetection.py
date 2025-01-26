import cv2
import numpy as np
from keras.models import load_model
import torch

# Load age and gender model
model_path = "./model.h5"
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading age and gender model: {e}")
    exit()

# Load the YOLOv5 model using torch
yolov5_model_path = 'crowdhuman_yolov5m.pt'
try:
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path)  # Loading the model from yolov5 repo
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

# Load the video
video_path = "C:/Users/anish/Downloads/WSM5.mp4"
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Get the video's width, height, and frames per second (fps)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the video
output_file = 'output_video.mp4'
video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process each frame of the video
while True:
    # Read the next frame
    success, frame = video.read()
    if not success:
        break

    # Perform object detection on the frame
    results = yolov5_model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Convert detections to numpy array

    # Process each detection
    for detection in detections:
        xmin, ymin, xmax, ymax, score, class_id = detection
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # Threshold score
        if score >= 0.3:
            color = (255, 0, 0) if class_id == 0 else (0, 0, 225)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            
            if class_id != 0:
                face = frame[ymin:ymax, xmin:xmax]
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (200, 200))
                img = np.expand_dims(img, axis=0)  # Adding batch dimension

                # Age and gender prediction
                predict = model.predict(img)
                age = int(predict[0][0])
                gender = 'Male' if np.argmax(predict[1]) == 0 else 'Female'

                # Annotating the frame with age and gender
                cv2.putText(frame, f"Age : {age}", (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 224, 0), 2)
                cv2.putText(frame, f"Gender : {gender}", (xmin, ymax + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 224, 0), 2)

    # Display the frame
    cv2.imshow("Video", frame)
    video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()

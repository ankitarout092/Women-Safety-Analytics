from keras.models import load_model
from collections import deque
import numpy as np
import cv2
import os

def print_results(video_path, model_path, output_dir='output', limit=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Loading model ...")
    model = load_model(model_path)
    Q = deque(maxlen=128)

    vs = cv2.VideoCapture(video_path)
    writer = None
    (W, H) = (None, None)

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32") / 255.0
        frame = np.expand_dims(frame, axis=0)

        preds = model.predict(frame)[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        label = (results > 0.50)[0]

        text_color = (0, 0, 255) if label else (0, 255, 0)
        text = f"Violence: {int(label)}"
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(os.path.join(output_dir, "v_output.avi"), fourcc, 30, (W, H), True)

        writer.write(output)

        # Uncomment the line below if you're running this locally, not on Google Colab.
        # cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "D:/Gen AI/VAS/Violence Detection/Testing videos/vio5.mp4"
model_path = "D:/Gen AI/VAS/Violence Detection/modelnew.h5"
print_results(video_path, model_path)

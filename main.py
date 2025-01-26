from tkinter import *
from tkinter import ttk, filedialog
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
from twilio.rest import Client
import numpy as np
import cv2
import os
import cvlib as cv
import sqlite3
from datetime import datetime
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
import folium
import pandas as pd

# Load the pre-trained gender classification model
def load_gender_model():
    dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
    model_path = get_file("gender_detection.model", dwnld_link, cache_subdir="pre-trained", cache_dir=os.getcwd())
    return load_model(model_path)

# Load the gesture recognition model (e.g., OpenPose)
def load_gesture_model():
    # Placeholder for loading an actual model like OpenPose
    # You would need to load the necessary files and set up the model here
    pass

# Initialize database
def init_db():
    conn = sqlite3.connect('gender_counts.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS GenderCounts
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      location TEXT,
                      men_count INTEGER,
                      women_count INTEGER,
                      incident_type TEXT)''')  # Added incident_type for different alert types
    conn.commit()
    conn.close()

def add_incident_type_column():
    conn = sqlite3.connect('gender_counts.db')
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE GenderCounts ADD COLUMN incident_type TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Column 'incident_type' already exists.")
        else:
            raise
    conn.commit()
    conn.close()

# Call this function after initializing the database
init_db()
add_incident_type_column()

# Save gender count to database
def save_to_db(location, men_count, women_count, incident_type="", latitude=None, longitude=None):
    conn = sqlite3.connect('gender_counts.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO GenderCounts (timestamp, location, men_count, women_count, incident_type, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, location, men_count, women_count, incident_type, latitude, longitude))
    conn.commit()
    conn.close()

def add_location_columns():
    conn = sqlite3.connect('gender_counts.db')
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE GenderCounts ADD COLUMN latitude REAL")
        cursor.execute("ALTER TABLE GenderCounts ADD COLUMN longitude REAL")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Columns 'latitude' and 'longitude' already exist.")
        else:
            raise
    conn.commit()
    conn.close()

add_location_columns()

# Check if it's night time
def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= 22 or current_hour < 5

# Detect a lone woman at night
def detect_lone_woman(men_count, women_count, location):
    if is_night_time() and women_count == 1:
        alert_message = f"Alert: Lone woman detected at night at {location}!"
        print(alert_message)
        sms(alert_message)
        save_to_db(location, men_count, women_count, "Lone Woman at Night")

# Calculate proximity between two bounding boxes
def calculate_proximity(box1, box2):
    center_x1 = box1[0] + box1[2] / 2
    center_y1 = box1[1] + box1[3] / 2
    center_x2 = box2[0] + box2[2] / 2
    center_y2 = box2[1] + box2[3] / 2
    return ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5

# Detect a woman surrounded by men
def detect_woman_surrounded_by_men(detections, location):
    for i, detection in enumerate(detections):
        if detection['gender'] == 'woman':
            men_around = 0
            for j, other in enumerate(detections):
                if i != j and other['gender'] == 'man':
                    distance = calculate_proximity(detection['box'], other['box'])
                    if distance < 100:
                        men_around += 1
            if men_around >= 3:
                alert_message = f"Alert: Woman surrounded by men detected at {location}!"
                print(alert_message)
                sms(alert_message)
                save_to_db(location, men_count, women_count, "Woman Surrounded by Men")

# Gesture Recognition for SOS Situations
def detect_sos_gesture(frame, gesture_model, location):
    # Placeholder function: In actual implementation, detect specific gestures using the gesture_model
    # Example: if gesture_model.detect_sos_gesture(frame):
    sos_detected = False  # Replace with actual detection logic

    if sos_detected:
        alert_message = f"Alert: SOS gesture detected at {location}!"
        print(alert_message)
        sms(alert_message)
        save_to_db(location, 0, 0, "SOS Gesture")

# Process video or photo for person detection, gesture recognition, and gender classification
def process_media(video_url, gender_model, gesture_model, location, frame_interval=10):
    try:
        if video_url.startswith(('http://', 'https://')):
            print(f"Downloading video from: {video_url}")
            local_video_path, _ = urllib.request.urlretrieve(video_url)
        else:
            print(f"Using local file: {video_url}")
            local_video_path = video_url

        webcam = cv2.VideoCapture(local_video_path)
        if not webcam.isOpened():
            print("Could not open video")
            return

        frame_count = 0
        while webcam.isOpened():
            status, frame = webcam.read()
            if not status:
                break

            if frame_count % frame_interval == 0:
                start_time = time.time()
                
                processed_frame, detections = process_frame(frame, gender_model)
                detect_sos_gesture(frame, gesture_model, location)

                men_count = sum(1 for d in detections if d['gender'] == 'man')
                women_count = sum(1 for d in detections if d['gender'] == 'woman')
                save_to_db(location, men_count, women_count)
                detect_lone_woman(men_count, women_count, location)
                detect_woman_surrounded_by_men(detections, location)

                end_time = time.time()
                print(f"Frame processed in {end_time - start_time:.2f} seconds")
                cv2.imshow("Gender Detection", processed_frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

# Detect faces and annotate the gender, counting men and women
def process_frame(frame, model, resize_factor=0.5, use_grayscale=True):
    if resize_factor != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    
    if use_grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    classes = ['man', 'woman']
    faces, confidence = cv.detect_face(frame)
    detections = []

    for idx, f in enumerate(faces):
        (startX, startY, endX, endY) = f
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = "{}: {:.2f}%".format(classes[idx], conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        color = (0, 0, 255) if classes[idx] == 'man' else (0, 255, 0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        detections.append({'box': f, 'gender': classes[idx]})

    men_count = sum(1 for d in detections if d['gender'] == 'man')
    women_count = sum(1 for d in detections if d['gender'] == 'woman')
    cv2.putText(frame, f'Men: {men_count}, Women: {women_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, detections

# Send SMS using Twilio
def sms(message):
    account_sid = ""
    auth_token = ""
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message,
        from_='',  # Replace with your Twilio number
        to=''      # Replace with the target number
    )
    print(f"Alert sent: {message.sid}")

# Identify hotspots based on past alerts
def identify_hotspots():
    conn = sqlite3.connect('gender_counts.db')
    df = pd.read_sql_query("SELECT * FROM GenderCounts", conn)
    conn.close()

    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("No data available for hotspot analysis or missing latitude/longitude.")
        return

    coords = df[['latitude', 'longitude']].dropna().values
    if len(coords) == 0:
        print("No valid latitude/longitude data available.")
        return

    dbscan = DBSCAN(eps=0.05, min_samples=3).fit(coords)  # Adjust eps and min_samples as needed
    df['cluster'] = dbscan.labels_

    hotspot_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=12)

    for idx, row in df.iterrows():
        color = 'red' if row['cluster'] != -1 else 'blue'
        folium.Marker(location=[row['latitude'], row['longitude']],
                    popup=f"Incident Type: {row['incident_type']} | Time: {row['timestamp']}",
                    icon=folium.Icon(color=color)).add_to(hotspot_map)

    hotspot_map.save("hotspots.html")
    print("Hotspot map generated and saved as 'hotspots.html'")

# GUI and main logic
def main():
    # Initialize models
    gender_model = load_gender_model()
    gesture_model = load_gesture_model()

    # Initialize database
    init_db()

    # GUI setup
    root = Tk()
    root.title("Gender Detection and SOS Alert System")
    root.geometry("400x200")

    text1 = Text(root, height=1, width=40)
    text1.insert(END, "Enter media file path or URL")
    text1.pack()

    text2 = Text(root, height=1, width=40)
    text2.insert(END, "Enter location")
    text2.pack()

    submit_button = ttk.Button(root, text="Submit", command=lambda: start_processing(text1.get("1.0", END).strip(), text2.get("1.0", END).strip(), gender_model, gesture_model))
    submit_button.pack()

    hotspot_button = ttk.Button(root, text="Identify Hotspots", command=identify_hotspots)
    hotspot_button.pack()

    root.mainloop()

def start_processing(video_url, location, gender_model, gesture_model):
    if video_url and location:
        with ThreadPoolExecutor() as executor:
            executor.submit(process_media, video_url, gender_model, gesture_model, location)
    else:
        print("Please provide both media file path and location.")

if __name__ == "__main__":
    main()

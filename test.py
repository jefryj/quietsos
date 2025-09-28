import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Sigmoid function for aspect ratio scoring
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def aspect_ratio_score(w, h):
    return sigmoid((w / h - 1.2) * 4)

# Folder for abnormal videos
os.makedirs("abnormal_clips", exist_ok=True)

# Initialize YOLOv8 model
cap = cv2.VideoCapture("http://192.168.0.102:8080/video")
model = YOLO("yolov8n.pt")
model.to("cuda:0")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Dictionaries for scores, video writers, and cooldowns
scores = {}         # track_id -> abnormal score
writers = {}        # track_id -> VideoWriter
cooldowns = {}      # track_id -> cooldown frames
COOLDOWN_FRAMES = 10
FIXED_SIZE = (224, 224)  # Resize all crops to this

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model.predict(frame, device="cuda:0", imgsz=640, conf=0.25, classes=[0], verbose=False)

    # Prepare detections for DeepSORT: [(bbox, confidence)]
    detections = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2] floats
            conf = float(box.conf[0])
            detections.append((xyxy, conf))  # tuple format required by DeepSORT

    # Update tracker and get persistent track IDs
    tracks = tracker.update_tracks(detections, frame=frame)

    current_abnormal_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Aspect ratio score
        w, h = x2 - x1, y2 - y1
        prev_score = scores.get(track_id, 0)
        score = 0.8 * prev_score + 0.2 * aspect_ratio_score(w, h)
        scores[track_id] = score

        abnormal = score > 0.6

        # Draw bounding box
        color = (0,0,255) if abnormal else (0,255,0)
        label = f"{'ABNORMAL' if abnormal else 'NORMAL'} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Record abnormal frames
        if abnormal:
            current_abnormal_ids.add(track_id)
            cropped = frame[y1:y2, x1:x2]
            cropped_resized = cv2.resize(cropped, FIXED_SIZE)
            if track_id not in writers:
                filename = f"abnormal_clips/person_{track_id}_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writers[track_id] = cv2.VideoWriter(filename, fourcc, 20.0, FIXED_SIZE)
                print(f"Started recording for person {track_id}: {filename}")
            writers[track_id].write(cropped_resized)

    # Handle cooldowns and stop writers
    for tid in list(writers.keys()):
        if tid not in current_abnormal_ids:
            if tid not in cooldowns:
                cooldowns[tid] = COOLDOWN_FRAMES
            else:
                cooldowns[tid] -= 1
            if cooldowns[tid] <= 0:
                writers[tid].release()
                print(f"Stopped recording for person {tid}")
                del writers[tid]
                del cooldowns[tid]
        else:
            if tid in cooldowns:
                del cooldowns[tid]

    cv2.imshow("YOLO Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release remaining writers
for w in writers.values():
    w.release()
cap.release()
cv2.destroyAllWindows()

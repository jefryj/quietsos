import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import supervision as sv
from dotenv import load_dotenv
load_dotenv()
# ------------------------------
# Utility functions
# ------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def aspect_ratio_score(w, h):
    return sigmoid((w / h - 1.2) * 4)

# ------------------------------
# Setup
# ------------------------------
os.makedirs("abnormal_clips", exist_ok=True)
video_url = os.getenv("VIDEO_URL")
if not video_url:
    raise ValueError("VIDEO_URL is not set in .env")
cap = cv2.VideoCapture(video_url)
model = YOLO("yolov8n.pt")
model.to("cuda:0")

# Initialize ByteTrack
tracker = sv.ByteTrack()

scores = {}      # track_id -> abnormal score
writers = {}     # track_id -> VideoWriter
cooldowns = {}   # track_id -> cooldown frames
COOLDOWN_FRAMES = 10
FIXED_SIZE = (224, 224)

# ------------------------------
# Main loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (only people: class=0)
    results = model.predict(frame, device="cuda:0", imgsz=640, conf=0.25, classes=[0], verbose=False)[0]

    # Convert detections for supervision
    detections = sv.Detections.from_ultralytics(results)

    # Update tracker
    tracks = tracker.update_with_detections(detections)

    current_abnormal_ids = set()

    # Iterate through tracked persons
    for xyxy, track_id, conf, class_id in zip(
        tracks.xyxy, tracks.tracker_id, tracks.confidence, tracks.class_id
    ):
        if track_id is None:
            continue  # skip if tracker couldn't assign ID

        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1

        prev_score = scores.get(track_id, 0)
        score = 0.8 * prev_score + 0.2 * aspect_ratio_score(w, h)
        scores[track_id] = score

        abnormal = score > 0.6

        # Draw visualization
        color = (0, 0, 255) if abnormal else (0, 255, 0)
        label = f"ID {track_id} | {'ABNORMAL' if abnormal else 'NORMAL'} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Record abnormal clips
        if abnormal:
            current_abnormal_ids.add(track_id)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            cropped_resized = cv2.resize(cropped, FIXED_SIZE)

            if track_id not in writers:
                filename = f"abnormal_clips/person_{track_id}_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writers[track_id] = cv2.VideoWriter(filename, fourcc, 20.0, FIXED_SIZE)
                print(f"Started recording for person {track_id}: {filename}")

            writers[track_id].write(cropped_resized)

    # Handle cooldowns
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

    # Show result
    cv2.imshow("YOLO + ByteTrack Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Cleanup
# ------------------------------
for w in writers.values():
    w.release()
cap.release()
cv2.destroyAllWindows()

import cv2

# Replace with your IP Webcam stream URL
url = "http://192.168.0.102:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Failed to open camera stream")
    exit()

print("✅ Camera stream opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("IP Camera Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

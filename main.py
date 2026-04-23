import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "input/video.mp4"
cap = cv2.VideoCapture(video_path)

# video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# output save
out = cv2.VideoWriter(
    "output/output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)

        # UI Window
        cv2.imshow("Multi Object Tracking (Press SPACE to Pause)", frame)

    key = cv2.waitKey(30)

    if key == 27:  # ESC → exit
        break
    elif key == 32:  # SPACE → pause/play
        paused = not paused

cap.release()
out.release()
cv2.destroyAllWindows()
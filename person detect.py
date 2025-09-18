import cv2
from ultralytics import YOLO
import csv
import os

# === Load YOLO model ===
model = YOLO("people.pt")   # your trained YOLO model (only 'person' class)

# === Video source ===
video_path = "ved2.mov"
cap = cv2.VideoCapture(video_path)

# === Save output video ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30,
                      (int(cap.get(3)), int(cap.get(4))))

# === Vertical Line (Gate position) ===
line_x = 1290   # adjust this to match the gate (try with imshow)
line_y1 = 197
line_y2 = 730

# === Counters ===
enter_count = 0
exit_count = 0
track_history = {}  # store (cx, cy)
last_side = {}      # store last known side (-1=left, +1=right)

# === Speed up video by skipping frames ===
frame_skip = 8
frame_index = 0

# === CSV logging setup ===
csv_file = "people_log.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["TrackID", "Action", "Frame", "Time(sec)"])  # header


def get_side(cx, line_x):
    """Return -1 if left of line, +1 if right of line"""
    return -1 if cx < line_x else 1


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % frame_skip != 0:
        continue

    # Resize for faster YOLO detection
    resized = cv2.resize(frame, (640, 360))
    results = model.track(resized, persist=True, show=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        # Scale detections back to original frame size
        h_ratio = frame.shape[0] / resized.shape[0]
        w_ratio = frame.shape[1] / resized.shape[1]

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Save history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))

            # Get current side
            current_side = get_side(cx, line_x)
            previous_side = last_side.get(track_id, current_side)
            last_side[track_id] = current_side  # update last side

            # Only count if they crossed line AND within gate vertical bounds
            if previous_side != current_side and line_y1 <= cy <= line_y2:
                if previous_side == -1 and current_side == 1:
                    enter_count += 1
                    print(f"Person {track_id} ENTERED")
                    with open(csv_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([track_id, "ENTER", frame_index,
                                         round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)])
                elif previous_side == 1 and current_side == -1:
                    exit_count += 1
                    print(f"Person {track_id} EXITED")
                    with open(csv_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([track_id, "EXIT", frame_index,
                                         round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)])

    # === Draw Line (Gate) ===
    cv2.line(frame, (line_x, line_y1), (line_x, line_y2), (0, 0, 255), 2)

    # === Show Counts ===
    cv2.putText(frame, f"Entered: {enter_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Exited: {exit_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow("People Counter", cv2.resize(frame, (960, 540)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# YOLO-Based-People-Counter-for-Store-Entry-Exit

This project is a real-time people counting system designed to track and count the number of people entering and exiting through a defined gate (e.g., a store entrance).
It uses a YOLO model trained on the person class and leverages OpenCV for drawing bounding boxes, tracking object IDs, and detecting line crossings.

# 🚀 Features
✅ Real-Time Detection & Tracking – Uses YOLO + built-in tracker to maintain unique person IDs.

✅ Entry & Exit Counting – Counts people crossing a virtual line (gate).

✅ CSV Logging – Automatically logs TrackID, Action (ENTER/EXIT), Frame, and Time into a CSV file.

✅ Video Output – Saves processed video with bounding boxes, line, and live counts.

✅ Frame Skipping – Speeds up video processing by skipping frames for faster results.

# 🛠️ Tech Stack
Python 3.8+

YOLO (Ultralytics) – for object detection

OpenCV – for video processing and visualization

CSV – for logging events

# ⚙️ How It Works

Load YOLO model (people.pt) trained on person class.

Open video stream (file or webcam).

Draw a virtual line (gate) on the frame.

Track people using YOLO tracking mode.

Detect crossings:

Left → Right ➝ Count as Entry

Right → Left ➝ Count as Exit

Log data to CSV file and display results live.

# ▶️ Usage

Clone the repository

git clone (https://github.com/aminrustmani/YOLO-Based-People-Counter-for-Store-Entry-Exit).git

cd People-Counter-YOLO

Install dependencies

pip install ultralytics opencv-python

Run the script

python person detect.py

Press ESC to stop the video processing.

# 🖼️ Example Output
✅ Green Bounding Boxes – Show detected persons

✅ Red Vertical Line – Represents the counting gate

✅ Counters – Displayed on top-left (Entered / Exited)

✅ CSV Log – Stores detailed entry/exit events

# 📊 Sample CSV Output
TrackID	Action	Frame	Time (sec)

1	ENTER	56	1.87

1	EXIT	204	6.80

2	ENTER	290	9.87

# 🎯 Use Cases
🏪 Retail Stores – Count customers entering/exiting

🏢 Office Buildings – Monitor employee movement

🎟️ Events – Track attendees in real-time

🏫 Schools – Track students entering/exiting premises

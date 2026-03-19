🧠 Smart AI Surveillance System
📌 Overview

The Smart AI Surveillance System is an intelligent security solution that uses Artificial Intelligence and Computer Vision to monitor real-time video streams and detect suspicious activities automatically.

It reduces the need for constant human monitoring by analyzing CCTV footage and generating alerts when unusual events occur.

🚀 Features

👤 Real-time Human Detection
🕵️ Suspicious Activity Detection (Mock Weapon Detection with Knife/Bat)
😊 Face Recognition (Known vs Unknown)
📊 Crowd Detection
🚨 Instant Alerts (Streamlit Dashboard & Database Logging)
📁 Event Logging with Timestamp
🌐 Web Dashboard for Monitoring

🛠️ Tech Stack
- Python Core programming
- OpenCV Video processing
- YOLOv8 (Ultralytics) Object & Human detection
- Streamlit Web interface
- SQLite Data storage
- face-recognition Face detection & recognition

⚙️ How to Run

1. **Install Dependencies (if not already done)**
   `pip install -r requirements.txt`

2. **Run the Application**
   Simply run the `run.bat` file OR use the following command:
   `streamlit run app.py`

3. **Dashboard Usage**
   - Access the dashboard at `http://localhost:8501`.
   - Select your Camera Source (0 for Webcam).
   - Click "Start Surveillance" to begin.
   - Adjust the Crowd Threshold as needed.
   - Add images of known people (e.g., `john_doe.jpg`) to the `known_faces/` directory for face recognition.

🧠 How It Works

📷 Capture live video from camera
🧠 Process frames using YOLOv8 model for object detection (Humans & Weapons)
🔍 Utilize `face-recognition` to find unknown individuals
⚠️ Apply rules to identify suspicious activity (Crowding, Unknown Faces, Weapons)
💾 Store event logs in SQLite database `events.db`
🌐 Display real-time results on the Streamlit dashboard

📜 License
This project is for educational purposes.

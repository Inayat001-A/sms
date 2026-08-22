# 🧠 Smart AI Surveillance System

## 📌 Overview

The **Smart AI Surveillance System** is an intelligent security solution powered by Computer Vision and Deep Learning (YOLOv8 + Face Recognition) to monitor real-time video streams, detect threats/weapons, identify known/unknown faces, detect crowds, and maintain persistent event logs.

---

## 🚀 Key Features

- **👤 Real-time Human Tracking & Detection**: Identifies and tracks people across frames using YOLOv8.
- **🗡️ Weapon / Threat Detection**: Automatic detection of suspicious objects (knives, bats, weapons) with instant threat escalation.
- **😊 Face Recognition**: Distinguishes known personnel from unknown faces using dlib encodings in real-time.
- **📊 Dynamic Crowd Detection**: Live crowd counter with configurable threshold alerts.
- **🚨 Real-Time Security Alert Feed**: Instant color-coded alerts and logging to SQLite (`events.db`).
- **📸 In-App Face Registration**: Snapshot and register new identities directly from the live camera stream.
- **🖥️ Native Desktop GUI (Antigravity Mode)**: High-performance, low-latency CustomTkinter desktop interface without needing a browser.
- **🌐 Web Browser Mode (Streamlit)**: Alternative web-based dashboard if browser viewing is desired.

---

## 🛠️ Tech Stack

- **GUI / Desktop**: `CustomTkinter`, `Pillow`, `Tkinter`
- **Vision & AI**: `OpenCV`, `Ultralytics YOLOv8`, `face_recognition`, `dlib`
- **Database**: `SQLite3` (`events.db`)
- **Web UI**: `Streamlit`

---

## ⚙️ How to Run

### 1. 🖥️ Native Desktop Mode (Inside Antigravity / No Browser)
Run with Python:
```bash
python desktop_app.py
```
*Or via entry point:*
```bash
python main.py
```
*Or double click:*
`run.bat`

### 2. 🌐 Web Browser Mode (Streamlit)
If you wish to run in a web browser:
```bash
streamlit run app.py
```
*Or double click:*
`run_web.bat`

---

## 🎮 Desktop Interface Controls

1. **Camera Source**: Select your camera device (e.g. `Camera 0 (Webcam)`, `Camera 1 (External)`, or `📱 IP / Mobile Camera (URL)`).
2. **Mobile / IP Camera URL**: When IP camera is selected, paste your phone's Wi-Fi stream URL (e.g. `http://192.168.1.15:8080/video`).
3. **Start / Stop Surveillance**: Toggle real-time AI processing with the main action button.
4. **Crowd Alert Threshold**: Drag the slider to set sensitivity (e.g. 3 people).
5. **Register Face**: Enter a name and click **📸 Capture & Register Face** while looking at the camera.
6. **Clear Logs**: Purge the SQLite alert logs whenever needed.
7. **Open Faces Folder**: View/manage saved known face images in Windows Explorer.


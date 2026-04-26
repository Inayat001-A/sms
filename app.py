import streamlit as st
import cv2
import pandas as pd
import os
from surveillance_pipeline import SurveillancePipeline
from database import get_recent_logs
import time

st.set_page_config(page_title="Smart Surveillance Dashboard", page_icon="📷", layout="wide")

st.title("🧠 Smart AI Surveillance Dashboard")

# Sidebar configurations
st.sidebar.header("⚙️ Configuration")
camera_source = st.sidebar.selectbox("Camera Source", ["0 (Webcam)", "1 (External Camera)"])
crowd_val = st.sidebar.slider("Crowd Alert Threshold", min_value=1, max_value=20, value=3)

cam_idx = int(camera_source.split(" ")[0])

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = SurveillancePipeline(camera_index=cam_idx, crowd_threshold=crowd_val)
else:
    st.session_state.pipeline.camera_index = cam_idx
    st.session_state.pipeline.crowd_threshold = crowd_val

st.sidebar.header("🚨 Notification Settings")
enable_email = st.sidebar.checkbox("📬 Enable Email Alerts", value=True)
enable_sms = st.sidebar.checkbox("📱 Enable SMS Alerts", value=False)

st.sidebar.subheader("Events to Alert On")
alert_crowd = st.sidebar.checkbox("👥 Crowd Detected", value=True)
alert_weapon = st.sidebar.checkbox("🔪 Weapon Detected", value=True)
alert_unknown = st.sidebar.checkbox("❓ Unknown Face", value=True)

st.session_state.pipeline.enable_email = enable_email
st.session_state.pipeline.enable_sms = enable_sms
st.session_state.pipeline.alert_crowd = alert_crowd
st.session_state.pipeline.alert_weapon = alert_weapon
st.session_state.pipeline.alert_unknown = alert_unknown

with st.sidebar.expander("📷 Add Person Identity"):
    st.info("Start Surveillance, type a name, and click to capture from live feed.")
    new_name = st.text_input("Person Name")
    if st.button("Capture & Register Face"):
        if new_name.strip() == "":
            st.error("Please enter a name first!")
        else:
            st.session_state.capture_requested = new_name.strip()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔴 Live Camera Feed")
    # Streamlit image placeholder
    frame_placeholder = st.empty()
    run_camera = st.checkbox("Start Surveillance", value=False)

with col2:
    st.subheader("🚨 Recent Alerts")
    log_placeholder = st.empty()

def update_logs_table():
    logs = get_recent_logs(15)
    if logs:
        df = pd.DataFrame(logs, columns=["ID", "Timestamp", "Event Type", "Description", "Image Path"])
        # Format the dataframe better
        df = df[["Timestamp", "Event Type", "Description"]]
        
        # Color code the rows based on event type if we want, or just display
        log_placeholder.dataframe(df, hide_index=True)
    else:
        log_placeholder.info("No events logged yet.")

# Main loop
if run_camera:
    if 'camera' not in st.session_state:
        cap_temp = cv2.VideoCapture(cam_idx)
        cap_temp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state.camera = cap_temp
    
    cap = st.session_state.camera
    
    # Check if camera opened
    if not cap.isOpened():
        st.error(f"Cannot open camera {cam_idx}")
    else:
        update_time = time.time()
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from camera")
                break
                
            # Handle Face Capture Request
            if st.session_state.get('capture_requested'):
                name_to_save = st.session_state.capture_requested
                if not os.path.exists("known_faces"):
                    os.makedirs("known_faces")
                save_path = os.path.join("known_faces", f"{name_to_save}.jpg")
                cv2.imwrite(save_path, frame)
                st.session_state.pipeline._load_known_faces("known_faces")
                st.sidebar.success(f"Registered {name_to_save} successfully!")
                st.session_state.capture_requested = None
                
            # Process the frame
            processed_frame, p_cnt, w_det = st.session_state.pipeline.process_frame(frame)
            
            # Convert colorspace for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Show the video
            frame_placeholder.image(processed_frame_rgb, channels="RGB")
            
            # Periodically update the logs table (every 2 seconds to save performance)
            if time.time() - update_time > 2.0:
                update_logs_table()
                update_time = time.time()
                
        # Do NOT release the camera here anymore, since we keep it open for reruns
else:
    if 'camera' in st.session_state:
        # User unchecked the box, release the camera hardware
        st.session_state.camera.release()
        del st.session_state.camera
        
    frame_placeholder.info("Camera is currently stopped. Check 'Start Surveillance' to begin.")
    update_logs_table()

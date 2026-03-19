import streamlit as st
import cv2
import pandas as pd
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
        log_placeholder.dataframe(df, use_container_width=True, hide_index=True)
    else:
        log_placeholder.info("No events logged yet.")

# Main loop
if run_camera:
    cap = cv2.VideoCapture(cam_idx)
    
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
                
            # Process the frame
            processed_frame, p_cnt, w_det = st.session_state.pipeline.process_frame(frame)
            
            # Convert colorspace for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Show the video
            frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            # Periodically update the logs table (every 2 seconds to save performance)
            if time.time() - update_time > 2.0:
                update_logs_table()
                update_time = time.time()
                
        cap.release()
else:
    frame_placeholder.info("Camera is currently stopped. Check 'Start Surveillance' to begin.")
    update_logs_table()

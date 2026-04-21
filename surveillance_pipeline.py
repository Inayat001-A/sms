import cv2
import time
from typing import Dict
import os
import numpy as np
from ultralytics import YOLO
import face_recognition

from database import log_event

class SurveillancePipeline:
    def __init__(self, camera_index=0, crowd_threshold=5):
        self.camera_index = camera_index
        self.crowd_threshold = crowd_threshold
        # Load YOLOv8 model for general object detection (COCO dataset)
        self.model = YOLO('yolov8n.pt') 
        
        # Load known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces("known_faces")
        
        # State tracking for alerts
        self.last_alert_times = {
            "crowd": 0.0,
            "weapon": 0.0,
            "unknown_face": 0.0
        }
        self.ALERT_COOLDOWN = 10 # seconds cooldown for same alert type
        
        # Track memory for persistent identities across frames
        self.track_names: Dict[int, str] = {}
        self.frame_count = 0

    def _load_known_faces(self, directory):
        if not os.path.exists(directory):
            print(f"Directory {directory} not found. Creating it.")
            os.makedirs(directory)
            return

        for filename in os.listdir(directory):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(directory, filename)
                try:
                    bgr_img = cv2.imread(path)
                    if bgr_img is not None:
                        image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(os.path.splitext(filename)[0])
                except Exception as e:
                    print(f"Error loading face {filename}: {e}")

    def process_frame(self, frame):
        current_time = time.time()
        self.frame_count += 1
            
        # 1. Face Recognition Processing First
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        recognized_faces = []
        has_unknown = False
        
        # Only run face recognition every 10th frame to prevent camera lag
        if len(self.known_face_encodings) > 0 and self.frame_count % 10 == 0:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                else:
                    has_unknown = True
                    
                # Scale back up for center calculation
                top *= 4; right *= 4; bottom *= 4; left *= 4
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2
                recognized_faces.append({
                    "name": name,
                    "center": (face_center_x, face_center_y)
                })

            if has_unknown and (current_time - self.last_alert_times["unknown_face"] > self.ALERT_COOLDOWN):
                log_event("UNKNOWN_FACE", "Unrecognized person detected.")
                self.last_alert_times["unknown_face"] = current_time

        # 2. Run YOLO object detection with tracking
        results = self.model.track(frame, persist=True, verbose=False, conf=0.45)
        
        person_count = 0
        weapon_detected = False
        suspicious_classes = [43, 34] # 43: knife, 34: baseball bat
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_id == 0: # Person
                    person_count += 1
                    track_id = int(box.id[0]) if (box.id is not None) else person_count
                    
                    display_name = f"Person {track_id}"
                    person_color = (0, 255, 0)
                    
                    # See if any face belongs to this YOLO person
                    for face in recognized_faces:
                        cx, cy = face["center"]
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            # Update our memory. Overwrite if we recognized a name.
                            if face["name"] != "Unknown" or track_id not in self.track_names:
                                self.track_names[track_id] = face["name"]
                            break
                            
                    # Override the YOLO name if we know who it is
                    if track_id in self.track_names:
                        display_name = self.track_names[track_id]
                        if display_name == "Unknown":
                            person_color = (0, 0, 255) # Red for unknown
                        else:
                            person_color = (0, 255, 0) # Green for known!
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                    cv2.putText(frame, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)
                
                elif cls_id in suspicious_classes:
                    weapon_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    weapon_name = self.model.names[cls_id]
                    cv2.putText(frame, f"SUSPICIOUS: {weapon_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Alerts Logic
        # Crowd Detection Alert
        if person_count >= self.crowd_threshold:
            if current_time - self.last_alert_times["crowd"] > self.ALERT_COOLDOWN:
                log_event("CROWD_DETECTED", f"Crowd of {person_count} people detected.")
                self.last_alert_times["crowd"] = current_time
                cv2.putText(frame, "ALERT: CROWD DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Weapon Detection Alert
        if weapon_detected:
            if current_time - self.last_alert_times["weapon"] > self.ALERT_COOLDOWN:
                log_event("WEAPON_DETECTED", "Suspicious object/weapon detected!")
                self.last_alert_times["weapon"] = current_time
            cv2.putText(frame, "ALERT: WEAPON DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return frame, person_count, weapon_detected

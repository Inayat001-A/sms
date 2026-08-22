import warnings
warnings.filterwarnings("ignore")

import os
import sys

import time
import threading
import subprocess
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

from surveillance_pipeline import SurveillancePipeline
from database import get_recent_logs, clear_logs, log_event, init_db

# Initialize database
init_db()

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SurveillanceDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🛡️ AI Smart Surveillance System")
        self.geometry("1280x820")
        self.minsize(1100, 700)

        # Application state
        self.is_running = False
        self.cap = None
        self.pipeline = None
        self.video_thread = None
        self.stop_event = threading.Event()
        self.capture_name_request = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.fps = 0.0
        self.person_count = 0
        self.weapon_detected = False
        self.last_log_check = 0

        # Build UI layout
        self._build_ui()

        # Handle window close cleanly
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start periodic GUI updates
        self.update_video_canvas()
        self.update_logs_display()
        self.update_clock()

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ----------------- Top Header Bar -----------------
        self.header_frame = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#1a1c23")
        self.header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        
        self.title_label = ctk.CTkLabel(
            title_box,
            text="🧠 SMART AI SURVEILLANCE",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00adb5"
        )
        self.title_label.pack(side="left")

        self.status_badge = ctk.CTkLabel(
            title_box,
            text="● STANDBY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888888",
            fg_color="#2b2d38",
            corner_radius=8,
            padx=10,
            pady=3
        )
        self.status_badge.pack(side="left", padx=15)

        # Header Stats
        self.stats_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.stats_box.grid(row=0, column=2, padx=20, pady=10, sticky="e")

        self.fps_label = ctk.CTkLabel(
            self.stats_box,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#a0a5b5"
        )
        self.fps_label.pack(side="left", padx=10)

        self.clock_label = ctk.CTkLabel(
            self.stats_box,
            text="--:--:--",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e0e0e0"
        )
        self.clock_label.pack(side="left", padx=10)

        # ----------------- Left Panel: Controls -----------------
        self.left_panel = ctk.CTkScrollableFrame(self, width=280, corner_radius=10, fg_color="#1f232d")
        self.left_panel.grid(row=1, column=0, padx=(15, 10), pady=15, sticky="nsew")

        # Section: Camera Controls
        ctrl_title = ctk.CTkLabel(
            self.left_panel, 
            text="⚙️ SURVEILLANCE CONTROLS", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e2e8f0"
        )
        ctrl_title.pack(anchor="w", pady=(5, 10))

        # Camera Source
        ctk.CTkLabel(self.left_panel, text="Camera Source:", font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.cam_select = ctk.CTkOptionMenu(
            self.left_panel,
            values=["Camera 0 (Webcam)", "Camera 1 (External)", "Camera 2", "📱 IP / Mobile Camera (URL)"],
            fg_color="#2b3240",
            button_color="#3b4455",
            button_hover_color="#4f5b72",
            command=self._on_cam_source_change
        )
        self.cam_select.set("Camera 0 (Webcam)")
        self.cam_select.pack(fill="x", pady=(2, 8))

        # Dynamic IP Camera URL Entry Frame
        self.ip_url_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        ctk.CTkLabel(
            self.ip_url_frame,
            text="🔗 Mobile Stream URL:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00adb5"
        ).pack(anchor="w")
        self.ip_url_entry = ctk.CTkEntry(
            self.ip_url_frame,
            placeholder_text="http://192.168.1.15:8080/video",
            fg_color="#181a20",
            border_color="#00adb5"
        )
        self.ip_url_entry.pack(fill="x", pady=(2, 6))


        # Start / Stop Button
        self.toggle_btn = ctk.CTkButton(
            self.left_panel,
            text="▶ START SURVEILLANCE",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#00adb5",
            hover_color="#008891",
            height=42,
            command=self.toggle_surveillance
        )
        self.toggle_btn.pack(fill="x", pady=(0, 15))

        # Section: AI Configuration
        ctk.CTkFrame(self.left_panel, height=2, fg_color="#2c3342").pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            self.left_panel, 
            text="🎯 DETECTION SETTINGS", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e2e8f0"
        ).pack(anchor="w", pady=(0, 8))

        self.threshold_label = ctk.CTkLabel(
            self.left_panel, 
            text="Crowd Alert Threshold: 3", 
            font=ctk.CTkFont(size=12)
        )
        self.threshold_label.pack(anchor="w")

        self.threshold_slider = ctk.CTkSlider(
            self.left_panel,
            from_=1,
            to=15,
            number_of_steps=14,
            command=self._on_threshold_change
        )
        self.threshold_slider.set(3)
        self.threshold_slider.pack(fill="x", pady=(2, 15))

        # Section: Face Registration
        ctk.CTkFrame(self.left_panel, height=2, fg_color="#2c3342").pack(fill="x", pady=10)

        ctk.CTkLabel(
            self.left_panel, 
            text="👤 ADD PERSON IDENTITY", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e2e8f0"
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            self.left_panel,
            text="Type a name & snapshot current face.",
            font=ctk.CTkFont(size=11),
            text_color="#8c96a8"
        ).pack(anchor="w", pady=(0, 6))

        self.name_entry = ctk.CTkEntry(
            self.left_panel,
            placeholder_text="Enter Person Name...",
            fg_color="#181a20",
            border_color="#363c4e"
        )
        self.name_entry.pack(fill="x", pady=(0, 8))

        self.capture_btn = ctk.CTkButton(
            self.left_panel,
            text="📸 Capture & Register Face",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#394254",
            hover_color="#4b576e",
            height=34,
            command=self.request_face_capture
        )
        self.capture_btn.pack(fill="x", pady=(0, 15))

        # Section: Quick Utilities
        ctk.CTkFrame(self.left_panel, height=2, fg_color="#2c3342").pack(fill="x", pady=10)

        ctk.CTkLabel(
            self.left_panel, 
            text="📁 MANAGEMENT", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e2e8f0"
        ).pack(anchor="w", pady=(0, 8))

        self.open_folder_btn = ctk.CTkButton(
            self.left_panel,
            text="📂 Open Known Faces Folder",
            font=ctk.CTkFont(size=12),
            fg_color="#272d3b",
            hover_color="#363f52",
            command=self.open_faces_folder
        )
        self.open_folder_btn.pack(fill="x", pady=(0, 6))

        self.clear_logs_btn = ctk.CTkButton(
            self.left_panel,
            text="🗑️ Clear Event Logs",
            font=ctk.CTkFont(size=12),
            fg_color="#452323",
            hover_color="#633131",
            command=self.clear_database_logs
        )
        self.clear_logs_btn.pack(fill="x", pady=(0, 10))

        # ----------------- Middle Panel: Video Feed Canvas -----------------
        self.center_panel = ctk.CTkFrame(self, corner_radius=10, fg_color="#14161d")
        self.center_panel.grid(row=1, column=1, padx=10, pady=15, sticky="nsew")
        self.center_panel.grid_rowconfigure(1, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        # Video Header / Threat Banner
        self.threat_banner = ctk.CTkLabel(
            self.center_panel,
            text="🟢 SYSTEM MONITORING SECURE",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1b2e24",
            text_color="#48bb78",
            corner_radius=6,
            height=32
        )
        self.threat_banner.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        # Video Display Container
        self.video_canvas = tk.Canvas(
            self.center_panel, 
            bg="#0c0d12", 
            highlightthickness=0,
            bd=0
        )
        self.video_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Video Footer info
        self.video_footer = ctk.CTkFrame(self.center_panel, height=35, fg_color="#1f232d", corner_radius=6)
        self.video_footer.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.person_count_label = ctk.CTkLabel(
            self.video_footer,
            text="👥 Persons Detected: 0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e2e8f0"
        )
        self.person_count_label.pack(side="left", padx=15)

        self.alert_status_label = ctk.CTkLabel(
            self.video_footer,
            text="Threat Level: LOW",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38a169"
        )
        self.alert_status_label.pack(side="right", padx=15)

        # ----------------- Right Panel: Recent Alerts Feed -----------------
        self.right_panel = ctk.CTkFrame(self, width=320, corner_radius=10, fg_color="#1f232d")
        self.right_panel.grid(row=1, column=2, padx=(10, 15), pady=15, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Alerts Feed Header
        alerts_header = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        alerts_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        ctk.CTkLabel(
            alerts_header,
            text="🚨 REAL-TIME ALERTS",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e2e8f0"
        ).pack(side="left")

        self.log_count_badge = ctk.CTkLabel(
            alerts_header,
            text="0 Events",
            font=ctk.CTkFont(size=11),
            fg_color="#2d3748",
            corner_radius=6,
            padx=6,
            pady=2
        )
        self.log_count_badge.pack(side="right")

        # Scrollable log cards container
        self.log_feed = ctk.CTkScrollableFrame(self.right_panel, fg_color="#14161d", corner_radius=8)
        self.log_feed.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # Initial placeholder on canvas
        self._draw_standby_placeholder()

    def _draw_standby_placeholder(self):
        self.video_canvas.delete("all")
        w = self.video_canvas.winfo_width() or 640
        h = self.video_canvas.winfo_height() or 480
        cx, cy = w // 2, h // 2
        
        self.video_canvas.create_text(
            cx, cy - 20,
            text="📷 CAMERA IS STANDBY",
            font=("Arial", 16, "bold"),
            fill="#4a5568"
        )
        self.video_canvas.create_text(
            cx, cy + 15,
            text="Select camera source and click 'Start Surveillance' to begin",
            font=("Arial", 11),
            fill="#718096"
        )

    def _on_cam_source_change(self, value):
        if "IP" in value or "URL" in value:
            self.ip_url_frame.pack(fill="x", pady=(0, 8), before=self.toggle_btn)
        else:
            self.ip_url_frame.pack_forget()

    def _on_threshold_change(self, value):
        val = int(value)
        self.threshold_label.configure(text=f"Crowd Alert Threshold: {val}")
        if self.pipeline:
            self.pipeline.crowd_threshold = val

    def toggle_surveillance(self):
        if not self.is_running:
            self.start_surveillance()
        else:
            self.stop_surveillance()

    def start_surveillance(self):
        cam_str = self.cam_select.get()
        if "IP" in cam_str or "URL" in cam_str:
            cam_source = self.ip_url_entry.get().strip()
            if not cam_source:
                messagebox.showwarning(
                    "IP Camera URL Required",
                    "Please enter the Mobile / IP Camera Stream URL.\n\nExample:\nhttp://192.168.1.15:8080/video"
                )
                return
        else:
            try:
                cam_source = int(cam_str.split(" ")[1])
            except Exception:
                cam_source = 0

        crowd_thresh = int(self.threshold_slider.get())

        # Initialize pipeline
        self.pipeline = SurveillancePipeline(camera_index=cam_source, crowd_threshold=crowd_thresh)
        
        # Test open camera
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            display_name = cam_source if isinstance(cam_source, str) else f"Camera {cam_source}"
            messagebox.showerror(
                "Camera Error",
                f"Could not connect to:\n{display_name}\n\nTips for Mobile Cameras:\n- Ensure Phone and PC are on the SAME Wi-Fi network\n- Verify the IP URL matches your phone's screen (e.g. http://192.168.1.X:8080/video)\n- Make sure 'Start Server' is running in the phone app"
            )
            self.cap.release()
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self.stop_event.clear()
        
        # Update UI state
        self.toggle_btn.configure(
            text="⏹ STOP SURVEILLANCE",
            fg_color="#e53e3e",
            hover_color="#c53030"
        )
        self.status_badge.configure(
            text="● LIVE MONITORING",
            text_color="#48bb78",
            fg_color="#1c3829"
        )
        self.cam_select.configure(state="disabled")
        self.ip_url_entry.configure(state="disabled")

        # Launch background processing thread
        self.video_thread = threading.Thread(target=self._video_worker_loop, daemon=True)
        self.video_thread.start()

    def stop_surveillance(self):
        self.is_running = False
        self.stop_event.set()

        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

        # Reset UI
        self.toggle_btn.configure(
            text="▶ START SURVEILLANCE",
            fg_color="#00adb5",
            hover_color="#008891"
        )
        self.status_badge.configure(
            text="● STANDBY",
            text_color="#888888",
            fg_color="#2b2d38"
        )
        self.cam_select.configure(state="normal")
        self.ip_url_entry.configure(state="normal")
        self.fps_label.configure(text="FPS: 0.0")

        self.person_count_label.configure(text="👥 Persons Detected: 0")
        self.threat_banner.configure(
            text="🟢 SYSTEM MONITORING STANDBY",
            fg_color="#1b2e24",
            text_color="#48bb78"
        )
        self.alert_status_label.configure(text="Threat Level: LOW", text_color="#38a169")

        self._draw_standby_placeholder()

    def request_face_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter a person name before registering.")
            return

        if not self.is_running:
            messagebox.showinfo("Surveillance Inactive", "Please start surveillance first to capture a live face frame.")
            return

        self.capture_name_request = name

    def _video_worker_loop(self):
        prev_time = time.time()
        fps_smoothing = 0.9

        while not self.stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Handle Face Capture Request
            if self.capture_name_request:
                name_to_save = self.capture_name_request
                self.capture_name_request = None
                
                faces_dir = "known_faces"
                if not os.path.exists(faces_dir):
                    os.makedirs(faces_dir)
                save_path = os.path.join(faces_dir, f"{name_to_save}.jpg")
                cv2.imwrite(save_path, frame)
                
                # Reload known faces in pipeline
                if self.pipeline:
                    self.pipeline._load_known_faces(faces_dir)
                
                log_event("FACE_REGISTERED", f"Successfully registered face identity: {name_to_save}")
                
                # Show quick non-blocking notification
                self.after(0, lambda n=name_to_save: messagebox.showinfo("Registered", f"Successfully registered '{n}'!"))

            # Process frame with AI Pipeline (YOLOv8 + Face Recognition)
            try:
                processed_frame, p_cnt, w_det = self.pipeline.process_frame(frame)
                self.person_count = p_cnt
                self.weapon_detected = w_det
            except Exception as e:
                print(f"Frame processing error: {e}")
                processed_frame = frame

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = (self.fps * fps_smoothing) + (inst_fps * (1.0 - fps_smoothing))

            # Store processed frame safely for GUI canvas
            with self.frame_lock:
                self.current_frame = processed_frame.copy()

            time.sleep(0.001)

    def update_video_canvas(self):
        if self.is_running:
            frame_to_render = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_to_render = self.current_frame

            if frame_to_render is not None:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(frame_to_render, cv2.COLOR_BGR2RGB)
                
                canvas_w = self.video_canvas.winfo_width()
                canvas_h = self.video_canvas.winfo_height()

                if canvas_w > 50 and canvas_h > 50:
                    img_h, img_w, _ = rgb_image.shape
                    scale = min(canvas_w / img_w, canvas_h / img_h)
                    new_w = int(img_w * scale)
                    new_h = int(img_h * scale)

                    resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    pil_img = Image.fromarray(resized)
                    self.photo_img = ImageTk.PhotoImage(pil_img)

                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(
                        canvas_w // 2, 
                        canvas_h // 2, 
                        image=self.photo_img, 
                        anchor="center"
                    )

            # Update status labels & Threat Banner
            self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
            self.person_count_label.configure(text=f"👥 Persons Detected: {self.person_count}")

            # Threat analysis
            crowd_thresh = int(self.threshold_slider.get())
            is_crowd = self.person_count >= crowd_thresh

            if self.weapon_detected and is_crowd:
                self.threat_banner.configure(
                    text="🚨 CRITICAL ALERT: WEAPON & CROWD DETECTED!",
                    fg_color="#742a2a",
                    text_color="#feb2b2"
                )
                self.alert_status_label.configure(text="Threat Level: CRITICAL", text_color="#e53e3e")
            elif self.weapon_detected:
                self.threat_banner.configure(
                    text="⚠️ SUSPICIOUS ACTIVITY: WEAPON/OBJECT DETECTED!",
                    fg_color="#742a2a",
                    text_color="#feb2b2"
                )
                self.alert_status_label.configure(text="Threat Level: HIGH", text_color="#e53e3e")
            elif is_crowd:
                self.threat_banner.configure(
                    text=f"⚠️ CROWD ALERT: {self.person_count} PERSONS DETECTED (Threshold: {crowd_thresh})",
                    fg_color="#7b341e",
                    text_color="#fbd38d"
                )
                self.alert_status_label.configure(text="Threat Level: MEDIUM", text_color="#dd6b20")
            else:
                self.threat_banner.configure(
                    text="🟢 SYSTEM MONITORING SECURE",
                    fg_color="#1b2e24",
                    text_color="#48bb78"
                )
                self.alert_status_label.configure(text="Threat Level: LOW", text_color="#38a169")

        # Schedule next video frame refresh (approx 30-40 fps UI update)
        self.after(25, self.update_video_canvas)

    def update_logs_display(self):
        curr_time = time.time()
        if curr_time - self.last_log_check > 1.5:
            self.last_log_check = curr_time
            logs = get_recent_logs(20)

            # Clear current log cards in scrollable frame
            for child in self.log_feed.winfo_children():
                child.destroy()

            if not logs:
                no_logs = ctk.CTkLabel(
                    self.log_feed,
                    text="No security alerts recorded yet.",
                    font=ctk.CTkFont(size=12),
                    text_color="#718096"
                )
                no_logs.pack(pady=20)
                self.log_count_badge.configure(text="0 Events")
            else:
                self.log_count_badge.configure(text=f"{len(logs)} Events")
                for log in logs:
                    log_id, timestamp, event_type, desc, _ = log
                    self._create_log_card(timestamp, event_type, desc)

        self.after(1500, self.update_logs_display)

    def _create_log_card(self, timestamp, event_type, desc):
        # Color palette depending on event
        if "WEAPON" in event_type:
            bg_color = "#2d1619"
            accent_color = "#e53e3e"
            icon = "🗡️"
        elif "CROWD" in event_type:
            bg_color = "#2d2013"
            accent_color = "#dd6b20"
            icon = "👥"
        elif "UNKNOWN" in event_type:
            bg_color = "#2b2614"
            accent_color = "#d69e2e"
            icon = "❓"
        elif "REGISTERED" in event_type:
            bg_color = "#14281e"
            accent_color = "#38a169"
            icon = "✅"
        else:
            bg_color = "#1f2430"
            accent_color = "#00adb5"
            icon = "📌"

        card = ctk.CTkFrame(self.log_feed, fg_color=bg_color, corner_radius=6, border_width=1, border_color=accent_color)
        card.pack(fill="x", pady=4, padx=2)

        # Header row inside card
        header_row = ctk.CTkFrame(card, fg_color="transparent")
        header_row.pack(fill="x", padx=8, pady=(6, 2))

        type_lbl = ctk.CTkLabel(
            header_row,
            text=f"{icon} {event_type}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=accent_color
        )
        type_lbl.pack(side="left")

        time_part = timestamp.split(" ")[-1] if " " in timestamp else timestamp
        time_lbl = ctk.CTkLabel(
            header_row,
            text=time_part,
            font=ctk.CTkFont(size=10),
            text_color="#a0aec0"
        )
        time_lbl.pack(side="right")

        # Description row
        desc_lbl = ctk.CTkLabel(
            card,
            text=desc,
            font=ctk.CTkFont(size=11),
            text_color="#e2e8f0",
            wraplength=260,
            justify="left"
        )
        desc_lbl.pack(anchor="w", padx=8, pady=(0, 6))

    def update_clock(self):
        curr = time.strftime("%Y-%m-%d  %H:%M:%S")
        self.clock_label.configure(text=curr)
        self.after(1000, self.update_clock)

    def open_faces_folder(self):
        folder_path = os.path.abspath("known_faces")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if sys.platform == "win32":
            os.startfile(folder_path)
        else:
            subprocess.Popen(["xdg-open", folder_path])

    def clear_database_logs(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all surveillance event logs?"):
            clear_logs()
            self.last_log_check = 0
            self.update_logs_display()

    def on_closing(self):
        self.stop_surveillance()
        self.destroy()


def main():
    app = SurveillanceDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import joblib
import mediapipe as mp
import math
import time
from collections import deque
import warnings
import csv
import os
from datetime import datetime
import atexit
import signal
import sys
import threading
warnings.filterwarnings("ignore", category=UserWarning)

# Load model
model_data = joblib.load('enhanced_model.pkl')
clf = model_data['classifier']
le = model_data['label_encoder']
scaler = model_data['scaler']

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables for data logging
last_data_to_log = None
csv_filename = "C:\\Users\\bhumi\\OneDrive\\Desktop\\professional\\internship\\mini\\data\\final_fatigue_dataset.csv"
session_start_time = None
last_log_time = None
LOG_INTERVAL = 15 * 60  # 15 minutes in seconds
session_id = None
emergency_save_called = False
data_lock = threading.Lock()

def initialize_csv():
    """Initialize CSV file with headers if it doesn't exist"""
    global csv_filename
    
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'session_id', 'start_timestamp', 'end_timestamp', 'start_datetime', 'end_datetime',
                'ear', 'mar', 'pitch', 'yaw', 'roll', 'blink_count', 'yawn_count', 
                'active_time_min', 'inactive_time_min', 'efficiency', 'status', 
                'fatigue', 'model_prediction', 'confidence'
            ])
        print(f"[INFO] Created new CSV file: {csv_filename}")
    else:
        print(f"[INFO] Using existing CSV file: {csv_filename}")

def generate_session_id():
    """Generate a unique session ID with timestamp"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def log_data_to_csv(data):
    """Log data to CSV file with thread safety"""
    try:
        with data_lock:
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        print(f"[INFO] Data logged - Session: {data[0]}")
    except Exception as e:
        print(f"[ERROR] Failed to log data: {e}")

def save_session_data(session_id, start_time, end_time, cumulative_data):
    """Save session data with start and end timestamps"""
    try:
        start_datetime = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        end_datetime = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get the latest data values with defaults
        ear = cumulative_data.get('ear', 0.0)
        mar = cumulative_data.get('mar', 0.0)
        pitch = cumulative_data.get('pitch', 0.0)
        yaw = cumulative_data.get('yaw', 0.0)
        roll = cumulative_data.get('roll', 0.0)
        blink_count = cumulative_data.get('blink_count', 0)
        yawn_count = cumulative_data.get('yawn_count', 0)
        active_time_min = cumulative_data.get('active_time_min', 0)
        inactive_time_min = cumulative_data.get('inactive_time_min', 0)
        efficiency = cumulative_data.get('efficiency', 0)
        status = cumulative_data.get('status', 'Unknown')
        fatigue = cumulative_data.get('fatigue', 'Fresh')
        model_prediction = cumulative_data.get('model_prediction', 'unknown')
        confidence = cumulative_data.get('confidence', 0.0)
        
        data_row = [
            session_id, start_time, end_time, start_datetime, end_datetime,
            ear, mar, pitch, yaw, roll, blink_count, yawn_count,
            active_time_min, inactive_time_min, efficiency, status,
            fatigue, model_prediction, confidence
        ]
        
        log_data_to_csv(data_row)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save session data: {e}")
        return False

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print(f"\n[INFO] Signal {sig} received, saving data...")
    emergency_save()
    cv2.destroyAllWindows()
    sys.exit(0)

def emergency_save():
    """Emergency save function called on exit"""
    global session_start_time, last_log_time, last_data_to_log, session_id, emergency_save_called
    
    with data_lock:
        if emergency_save_called:
            print("[INFO] Emergency save already called, skipping...")
            return
        emergency_save_called = True
    
    print("[💾] Emergency save triggered...")
    current_time = time.time()

    if session_start_time is None:
        print("[❌] Session never started. Nothing to save.")
        return

    if last_data_to_log is None:
        print("[⚠️] No data collected. Saving minimal session info.")
        dummy_data = {
            'session_id': f"{session_id}_no_data",
            'ear': 0.0, 'mar': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
            'blink_count': 0, 'yawn_count': 0, 'active_time_min': 0, 'inactive_time_min': 0,
            'efficiency': 0, 'status': 'No Data', 'fatigue': 'Unknown',
            'model_prediction': 'no_face', 'confidence': 0.0
        }
        save_session_data(dummy_data['session_id'], session_start_time, current_time, dummy_data)
        return

    # Save final session data
    if last_log_time is None:
        last_log_time = session_start_time

    final_session_id = f"{session_id}_final"
    session_duration = current_time - last_log_time
    
    print(f"[✅] Saving final session: {session_duration/60:.1f} minutes")
    
    success = save_session_data(final_session_id, last_log_time, current_time, last_data_to_log)
    if success:
        print(f"[✅] Emergency save completed successfully!")
    else:
        print(f"[❌] Emergency save failed!")

def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    try:
        A = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
        B = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
        C = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
        return (A + B) / (2.0 * C) if C != 0 else 0
    except:
        return 0

def mouth_aspect_ratio(landmarks):
    """Calculate Mouth Aspect Ratio"""
    try:
        A = np.linalg.norm(landmarks[13] - landmarks[14])
        B = np.linalg.norm(landmarks[12] - landmarks[15])
        C = np.linalg.norm(landmarks[78] - landmarks[308])
        return (A + B) / (2.0 * C) if C != 0 else 0
    except:
        return 0

def estimate_head_pose(landmarks):
    """Estimate head pose angles"""
    try:
        nose_tip = landmarks[1]
        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        left_mouth_corner = landmarks[61]
        right_mouth_corner = landmarks[291]
        eye_center = (left_eye_corner + right_eye_corner) / 2
        mouth_center = (left_mouth_corner + right_mouth_corner) / 2
        eye_width = np.linalg.norm(right_eye_corner - left_eye_corner)
        
        yaw = math.atan2(np.linalg.norm(nose_tip - right_eye_corner) - np.linalg.norm(nose_tip - left_eye_corner), eye_width) * 180 / math.pi
        pitch = math.atan2(np.linalg.norm(nose_tip - eye_center), np.linalg.norm(nose_tip - mouth_center)) * 180 / math.pi - 90
        roll = math.atan((right_eye_corner[1] - left_eye_corner[1]) / (right_eye_corner[0] - left_eye_corner[0])) * 180 / math.pi
        
        return pitch, yaw, roll
    except:
        return 0, 0, 0

def extract_features(image):
    """Extract facial features from image"""
    h, w = image.shape[:2]
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, False
    
    face = results.multi_face_landmarks[0]
    coords = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])

    left_ear = eye_aspect_ratio(coords, [33, 160, 158, 133, 153, 144])
    right_ear = eye_aspect_ratio(coords, [362, 385, 387, 263, 373, 380])
    avg_ear = (left_ear + right_ear) / 2.0
    mar = mouth_aspect_ratio(coords)
    pitch, yaw, roll = estimate_head_pose(coords)
    
    # Additional features
    eye_distance = np.linalg.norm(coords[33] - coords[263])
    face_symmetry = abs(np.linalg.norm(coords[33] - coords[1]) - np.linalg.norm(coords[263] - coords[1]))
    left_eye_height = np.linalg.norm(coords[159] - coords[145])
    right_eye_height = np.linalg.norm(coords[386] - coords[374])
    avg_eye_height = (left_eye_height + right_eye_height) / 2.0
    mouth_width = np.linalg.norm(coords[61] - coords[291])
    mouth_height = np.linalg.norm(coords[13] - coords[14])
    face_confidence = min(1.0, (avg_ear + mar + avg_eye_height/100) / 3.0)

    features = [avg_ear, mar, pitch, yaw, roll, eye_distance, face_symmetry, avg_eye_height,
                mouth_width, mouth_height, left_ear, right_ear, face_confidence]
    return features, True

def predict_fatigue(image):
    """Predict fatigue level from image"""
    features, detected = extract_features(image)
    if not detected:
        return 'no_face', 0.0, 'No Face', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Not Available'
    
    try:
        feats_scaled = scaler.transform([features])
        pred = clf.predict(feats_scaled)[0]
        probs = clf.predict_proba(feats_scaled)[0]
        pred_class = le.inverse_transform([pred])[0]
        confidence = max(probs)
        
        base_eff = int(70 + confidence * 30) if pred_class == 'active' else \
                   int(20 + confidence * 30) if pred_class == 'fatigued' else \
                   int(30 + confidence * 40) if pred_class == 'distracted' else 0
        
        return pred_class, confidence, pred_class.capitalize(), base_eff, features[0], features[1], features[2], features[3], features[4], 'Fresh'
    except:
        return 'error', 0.0, 'Error', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Not Available'

def main():
    global last_data_to_log, last_log_time, session_start_time, session_id
    
    # Initialize CSV file
    initialize_csv()
    
    # Generate unique session ID for this run
    session_id = generate_session_id()
    session_start_time = time.time()
    last_log_time = session_start_time
    
    print(f"[INFO] Session ID: {session_id}")
    print(f"[INFO] Session started at: {datetime.fromtimestamp(session_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup signal handlers and emergency save
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(emergency_save)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    # Time tracking variables
    active_start = session_start_time
    inactive_start = None
    total_active_time = 0.0
    total_inactive_time = 0.0
    
    # IMPROVED BLINK DETECTION VARIABLES
    blink_count = 0
    yawn_count = 0
    blink_thresh = 0.18
    sleep_thresh = 0.15
    mar_thresh = 0.65
    
    # Blink detection improvement
    blink_state = False
    blink_frames = 0
    MIN_BLINK_FRAMES = 3  # Minimum frames for a valid blink
    MAX_BLINK_FRAMES = 15  # Maximum frames for a valid blink
    blink_cooldown = 0
    BLINK_COOLDOWN_FRAMES = 5  # Cooldown between blinks
    
    # Yawn detection improvement
    yawn_state = False
    yawn_frames = 0
    MIN_YAWN_FRAMES = 10
    yawn_cooldown = 0
    YAWN_COOLDOWN_FRAMES = 30
    
    # Fatigue detection
    closed_eye_frames = 0
    MIN_SLEEPY_FRAMES = 30
    YAWN_THRESHOLD = 10
    fatigue = "Fresh"
    yawn_window = deque(maxlen=300)

    # Smoothing variables
    efficiency_window = deque(maxlen=30)
    status_window = deque(maxlen=20)
    current_efficiency = 70
    current_status = "Initializing"
    last_status_change = time.time()
    MIN_STATUS_CHANGE_INTERVAL = 2  # Reduced from 20 to 2 seconds

    print(f"[INFO] Starting real-time detection. Data will be logged every {LOG_INTERVAL/60} minutes.")
    print(f"[INFO] CSV file: {csv_filename}")
    print(f"[INFO] Press 'q' to quit and save final data.")
    print(f"[INFO] Data will be auto-saved on any exit (crash/close/Ctrl+C).")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        frame_count += 1

        # Check if we need to log data for completed 15-minute intervals
        time_since_last_log = current_time - last_log_time
        if time_since_last_log >= LOG_INTERVAL and last_data_to_log is not None:
            complete_intervals = int(time_since_last_log // LOG_INTERVAL)
            
            for i in range(complete_intervals):
                interval_start = last_log_time + (i * LOG_INTERVAL)
                interval_end = interval_start + LOG_INTERVAL
                
                interval_data = last_data_to_log.copy()
                interval_data['session_id'] = f"{session_id}_15min_{i+1}"
                
                print(f"[INFO] Logging 15-minute interval {i+1}")
                save_session_data(interval_data['session_id'], interval_start, interval_end, interval_data)
            
            last_log_time += complete_intervals * LOG_INTERVAL

        # Face detection and prediction
        try:
            pred_class, conf, status, eff, ear, mar, pitch, yaw, roll, _ = predict_fatigue(frame)
            no_face = (pred_class == 'no_face')
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            no_face = True
            pred_class, conf, status, eff, ear, mar, pitch, yaw, roll, _ = 'no_face', 0.0, 'No Face', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Not Available'

        # Time tracking logic
        if no_face:
            if inactive_start is None:
                inactive_start = current_time
            if active_start is not None:
                active_duration = current_time - active_start
                total_active_time += active_duration
                active_start = None
        else:
            if active_start is None:
                active_start = current_time
            if inactive_start is not None:
                inactive_duration = current_time - inactive_start
                total_inactive_time += inactive_duration
                inactive_start = None

        # IMPROVED BLINK DETECTION
        if not no_face and blink_cooldown <= 0:
            if ear < blink_thresh:
                if not blink_state:
                    blink_state = True
                    blink_frames = 1
                else:
                    blink_frames += 1
                    if blink_frames > MAX_BLINK_FRAMES:
                        blink_state = False
                        blink_frames = 0
            else:
                if blink_state and blink_frames >= MIN_BLINK_FRAMES:
                    blink_count += 1
                    blink_cooldown = BLINK_COOLDOWN_FRAMES
                    
                blink_state = False
                blink_frames = 0
        
        if blink_cooldown > 0:
            blink_cooldown -= 1

        # Enhanced fatigue detection
        if not no_face:
            if ear < sleep_thresh:
                closed_eye_frames += 1
                if closed_eye_frames > MIN_SLEEPY_FRAMES:
                    fatigue = "Sleepy"
            else:
                closed_eye_frames = 0
                if yawn_count < YAWN_THRESHOLD and fatigue != "Tired":
                    fatigue = "Fresh"

        # IMPROVED YAWN DETECTION
        if not no_face and yawn_cooldown <= 0:
            if mar > mar_thresh:
                if not yawn_state:
                    yawn_state = True
                    yawn_frames = 1
                else:
                    yawn_frames += 1
            else:
                if yawn_state and yawn_frames >= MIN_YAWN_FRAMES:
                    yawn_count += 1
                    yawn_window.append(1)
                    yawn_cooldown = YAWN_COOLDOWN_FRAMES
                    
                    if sum(yawn_window) >= YAWN_THRESHOLD:
                        fatigue = "Tired"
                else:
                    yawn_window.append(0)
                yawn_state = False
                yawn_frames = 0
        
        if yawn_cooldown > 0:
            yawn_cooldown -= 1

        # Smooth efficiency and status changes
                # Smooth efficiency and status changes
        if not no_face:
            efficiency_window.append(eff)
            status_window.append(status)

            if len(efficiency_window) > 5:
                current_efficiency = sum(efficiency_window) / len(efficiency_window)

            if len(status_window) > 3:
                proposed_status = max(set(status_window), key=status_window.count)
                if proposed_status != current_status:
                    if (current_time - last_status_change) > MIN_STATUS_CHANGE_INTERVAL:
                        current_status = proposed_status
                        last_status_change = current_time
        else:
            current_status = "No Face"
            current_efficiency = 0



        # Calculate current session times
        current_active_time = total_active_time
        current_inactive_time = total_inactive_time
        
        if active_start is not None:
            current_active_time += (current_time - active_start)
        if inactive_start is not None:
            current_inactive_time += (current_time - inactive_start)
        
        display_active_min = int(current_active_time / 60)
        display_inactive_min = int(current_inactive_time / 60)

        # Update last_data_to_log with current data
        last_data_to_log = {
            'session_id': session_id,
            'ear': ear,
            'mar': mar,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'blink_count': blink_count,
            'yawn_count': yawn_count,
            'active_time_min': display_active_min,
            'inactive_time_min': display_inactive_min,
            'efficiency': int(current_efficiency),
            'status': current_status,
            'fatigue': fatigue,
            'model_prediction': pred_class,
            'confidence': conf
        }

        # Display information
        cv2.putText(frame, f"EAR: {ear:.3f} | MAR: {mar:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Pose: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count} | Yawns: {yawn_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Active: {display_active_min}m | Inactive: {display_inactive_min}m", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
        cv2.putText(frame, f"Efficiency: {int(current_efficiency)}% | Status: {current_status}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 200, 160), 2)
        
        # Session time display
        session_duration = int((current_time - session_start_time) / 60)
        cv2.putText(frame, f"Session: {session_duration}m", (10, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Fatigue display with color coding
        if fatigue == "Fresh":
            fatigue_color = (0, 255, 0)  # Green
        elif fatigue == "Sleepy":
            fatigue_color = (0, 255, 255)  # Yellow
        else:  # "Tired"
            fatigue_color = (0, 0, 255)  # Red
            
        cv2.putText(frame, f"Fatigue: {fatigue}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)
        cv2.putText(frame, f"Model: {pred_class.upper()} ({conf:.2f})", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Debug info
        cv2.putText(frame, f"Frame: {frame_count}", (400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.imshow("Enhanced Productivity Analyzer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed - saving final data before exit...")
            emergency_save()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
        emergency_save()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        emergency_save()
    finally:
        cv2.destroyAllWindows()
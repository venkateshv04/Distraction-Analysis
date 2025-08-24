import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
import mediapipe as mp
from gaze_tracking import GazeTracking
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.ndimage import gaussian_filter1d

# Define relative paths for the models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPE_PREDICTOR_PATH = os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Initialize MediaPipe for face orientation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.8)

# Initialize GazeTracking
gaze = GazeTracking()

# Data recording
timestamps = []
concentration_values = []
drowsiness_timeline = []

# Weightages for concentration percentage
gaze_weight = 0.4
drowsiness_weight = 0.3
orientation_weight = 0.3

# No-face tracking adjustment
no_face_counter = 0
no_face_threshold = 10  # Tolerate up to 10 frames with no face

# Start timer
start_time = time.time()

# Helper functions
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    if down == 0:  # Avoid division by zero
        return 2 
    ratio = up / (2.0 * down)
    
    if ratio > 0.25:
        return 2  # Eyes open
    elif ratio > 0.21 and ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Eyes closed/sleeping

def get_head_orientation(landmarks, img_w, img_h):
    face_2d = []
    face_3d = []
    for idx, lm in enumerate(landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    if not face_2d or not face_3d:
        return None, None, None

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success:
        return None, None, None
        
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360, angles[1] * 360, angles[2] * 360

# Add status tracking for blink logic
sleep_frames = 0
drowsy_frames = 0
active_frames = 0
blink_status = ""
blink_color = (0, 0, 0)
orientation_status = "Facing Forward"
orientation_score = orientation_weight


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_dlib = detector(gray)
    results_mesh = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    expression_score = 0
    drowsy_score = 0
    elapsed_time = int(time.time() - start_time)
    current_state = "normal"
    
    if blink_status == "Drowsy !":
        current_state = "drowsy"
    elif blink_status == "SLEEPING !!!":
        current_state = "sleepy"
    drowsiness_timeline.append((elapsed_time, current_state))
    
    if not faces_dlib or not results_mesh.multi_face_landmarks:
        no_face_counter += 1
        if no_face_counter > no_face_threshold:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        no_face_counter = 0
        face = faces_dlib[0] # Take the first detected face
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Get head orientation from MediaPipe
        x, y, _ = get_head_orientation(results_mesh.multi_face_landmarks[0], frame.shape[1], frame.shape[0])
        
        if x is not None:
            if y < -10:
                orientation_status = "Facing Left"
                orientation_score = 0
            elif y > 10:
                orientation_status = "Facing Right"
                orientation_score = 0
            elif x < -10:
                orientation_status = "Facing Down"
                orientation_score = orientation_weight * 0.5
            elif x > 10:
                orientation_status = "Facing Up"
                orientation_score = orientation_weight * 0.5
            else:
                orientation_status = "Facing Forward"
                orientation_score = orientation_weight

        # Blink detection
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        # Drowsiness detection logic with orientation check
        if orientation_status == "Facing Down":
            # Do not register as drowsy or sleeping when looking down
            sleep_frames = 0
            drowsy_frames = 0
            active_frames += 1
            blink_status = "Active :)"
            blink_color = (0, 255, 0)
            drowsy_score = drowsiness_weight
        else:
            if left_blink == 0 or right_blink == 0:
                sleep_frames += 1
                drowsy_frames = 0
                active_frames = 0
                if sleep_frames > 6:
                    blink_status = "SLEEPING !!!"
                    blink_color = (0, 0, 255)
                    drowsy_score = 0
            elif left_blink == 1 or right_blink == 1:
                sleep_frames = 0
                active_frames = 0
                drowsy_frames += 1
                if drowsy_frames > 6:
                    blink_status = "Drowsy !"
                    blink_color = (0, 255, 255)
                    drowsy_score = drowsiness_weight * 0.5
            else:
                drowsy_frames = 0
                sleep_frames = 0
                active_frames += 1
                if active_frames > 6:
                    blink_status = "Active :)"
                    blink_color = (0, 255, 0)
                    drowsy_score = drowsiness_weight
                    
        gaze.refresh(frame)
        gaze_status = "Looking Center"
        gaze_score = gaze_weight
        if not gaze.pupils_located:
            gaze_status = "Looking Center"
            gaze_score = gaze_weight * 1
        elif gaze.is_blinking():
            gaze_status = "Blinking"
            gaze_score = gaze_weight * 1
        elif gaze.is_right():
            gaze_status = "Looking Right"
            gaze_score = 0
        elif gaze.is_left():
            gaze_status = "Looking Left"
            gaze_score = 0
        elif gaze.is_up():
            gaze_status = "Looking Up"
            gaze_score = 0
        elif gaze.is_down():
            gaze_status = "Looking Down"
            gaze_score = 0
        
        concentration_percentage = int((drowsy_score + gaze_score + orientation_score) * 100)
        text_color = (0, 255, 0) if concentration_percentage >= 50 else (0, 0, 255)
        
        # Store data
        timestamps.append(elapsed_time)
        concentration_values.append(concentration_percentage)
        
        # Display outputs
        cv2.putText(frame, f"Drowsiness: {blink_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, blink_color, 2)
        cv2.putText(frame, f"Orientation: {orientation_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Forward" in orientation_status else (0, 0, 255), 2)
        cv2.putText(frame, f"Gaze: {gaze_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if gaze_status == "Looking Center" else (0, 0, 255), 2)
        cv2.putText(frame, f"Concentration: {concentration_percentage}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(frame, f"Time: {elapsed_time}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Concentration Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Generate Graph
def moving_average(data, window_size=5):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Convert data
timestamps = np.array(timestamps)
concentration_values = np.array(concentration_values)

if len(concentration_values) > 1:
    smoothed_concentration = gaussian_filter1d(concentration_values, sigma=1)
    smoothed_concentration = moving_average(smoothed_concentration, window_size=5)
    
    # Adjust x-axis for smoothing
    smoothed_timestamps = timestamps[len(timestamps) - len(smoothed_concentration):]

    # Create drowsiness segments for graph
    drowsiness_segments = []
    if drowsiness_timeline:
        prev_state = drowsiness_timeline[0][1]
        start_time_d = drowsiness_timeline[0][0]

        for t, state in drowsiness_timeline[1:]:
            if state != prev_state:
                if prev_state in ["drowsy", "sleepy"]:
                    drowsiness_segments.append((start_time_d, t))
                start_time_d = t
                prev_state = state
        if prev_state in ["drowsy", "sleepy"]:
            drowsiness_segments.append((start_time_d, timestamps[-1]))

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed_timestamps, smoothed_concentration, color='black', linewidth=2)
    plt.fill_between(smoothed_timestamps, smoothed_concentration, color='green', alpha=0.6)

    # Add Drowsy/Sleepy segments
    has_added = False
    for start, end in drowsiness_segments:
        if not has_added:
            plt.axvspan(start, end, color='lightcoral', alpha=0.4, label='Drowsy/Sleepy')
            has_added = True
        else:
            plt.axvspan(start, end, color='lightcoral', alpha=0.4)

    legend_patch = plt.Rectangle((0, 0), 1, 1, color='lightcoral', alpha=0.3)
    plt.legend([legend_patch], ['Drowsy/Sleepy'], loc="upper left")

    # Labels
    plt.xlabel("Time (seconds)")
    plt.ylabel("Concentration (%)")
    plt.title("Concentration vs Time")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from IPython.display import display

# --- CONFIGURATION ---
DATA_SUBFOLDER = 'data'
INPUT_VIDEO_PATH = os.path.join(DATA_SUBFOLDER, 'library 1 min.mp4')
FACE_MODEL_PATH = os.path.join(DATA_SUBFOLDER, 'face_landmarker.task')
POSE_MODEL_PATH = os.path.join(DATA_SUBFOLDER, 'pose_landmarker.task')
YOLO_MODEL_PATH = 'yolov8n.pt'
OUTPUT_CSV_PATH = os.path.join(DATA_SUBFOLDER, 'final_thesis_report.csv')
OUTPUT_VIDEO_PATH = os.path.join(DATA_SUBFOLDER, 'final_labeled_output.mp4')

# Distance threshold (in pixels) to merge IDs. 
# If a new ID is within 50px of an old one, merge them.
MERGE_THRESHOLD = 50 

# --- INITIALIZE MODELS ---
base_options_face = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
options_face = vision.FaceLandmarkerOptions(base_options=base_options_face, output_face_blendshapes=True, running_mode=vision.RunningMode.IMAGE)
face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose, running_mode=vision.RunningMode.IMAGE)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

yolo_model = YOLO(YOLO_MODEL_PATH)

def calculate_smile_robust(face_landmarks):
    left, right = face_landmarks[0][61], face_landmarks[0][291]
    top, bottom = face_landmarks[0][13], face_landmarks[0][14]
    width = np.linalg.norm(np.array([left.x, left.y]) - np.array([right.x, right.y]))
    return width > 0.07 

def run_thesis_model(video_path):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    student_db = {} # Main database: {mapped_id: {data}}
    id_map = {}     # Maps YOLO_ID -> mapped_id (The "Merge" logic)
    last_positions = {} # {mapped_id: (center_x, center_y)}
    
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Use botsort for more stable tracking in crowded classrooms
        results = yolo_model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for i, yolo_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # --- ID MERGING LOGIC ---
                if yolo_id not in id_map:
                    found_match = False
                    # Check if this new YOLO ID is close to a known student's last position
                    for m_id, pos in last_positions.items():
                        dist = np.sqrt((cx - pos[0])**2 + (cy - pos[1])**2)
                        if dist < MERGE_THRESHOLD:
                            id_map[yolo_id] = m_id
                            found_match = True
                            break
                    
                    if not found_match:
                        # Truly a new student entry
                        new_id = len(student_db) + 1
                        id_map[yolo_id] = new_id
                        student_db[new_id] = {'smiles': 0, 'head_down': 0, 'total': 0}

                m_id = id_map[yolo_id]
                student_db[m_id]['total'] += 1
                last_positions[m_id] = (cx, cy)

                # --- ANALYSIS ---
                pad = 20
                crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                if crop.size > 0:
                    mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    f_res = face_landmarker.detect(mp_crop)
                    p_res = pose_landmarker.detect(mp_crop)

                    if f_res.face_landmarks and calculate_smile_robust(f_res.face_landmarks):
                        student_db[m_id]['smiles'] += 1
                        cv2.putText(frame, "Smiling", (x1, y1-10), 1, 1, (0, 255, 255), 2)

                    if p_res.pose_landmarks:
                        nose = p_res.pose_landmarks[0][0]
                        if nose.visibility < 0.5 or nose.y > 0.4:
                            student_db[m_id]['head_down'] += 1
                            cv2.putText(frame, "Head Down", (x1, y1-25), 1, 1, (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {m_id}", (x1, y1+20), 1, 1, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0: print(f"Processing frame {frame_count}...")

    cap.release()
    out.release()

    # --- FINAL REPORT FILTERING ---
    report = []
    for s_id, d in student_db.items():
        # Only include if they were present for a significant portion of the video
        if d['total'] < (fps * 2): continue 
        
        s_rate = (d['smiles']/d['total']) * 100
        h_rate = (d['head_down']/d['total']) * 100
        status = "Active" if s_rate > 5 else ("Passive" if h_rate > 20 else "Focusing")
        
        report.append({'ID': s_id, 'Smile%': round(s_rate, 1), 'HeadDown%': round(h_rate, 1), 'Status': status, 'Total_Frames': d['total']})

    # FINAL STEP: Strictly return only the top 8 students based on visibility duration
    df = pd.DataFrame(report)
    if not df.empty:
        df = df.sort_values(by='Total_Frames', ascending=False).head(8).sort_values(by='ID')
    
    return df.drop(columns=['Total_Frames'])

# EXECUTE
print("Starting Merged ID Analysis...")
final_df = run_thesis_model(INPUT_VIDEO_PATH)
display(final_df)
final_df.to_csv(OUTPUT_CSV_PATH, index=False)

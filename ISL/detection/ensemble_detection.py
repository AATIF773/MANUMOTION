import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

# ==========================================
# 1. DYNAMIC PATH SETUP (Cloud Friendly)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjusting paths to find your ISL/models folder
if "pages" in BASE_DIR or "detection" in BASE_DIR:
    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "ISL", "models"))
else:
    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "ISL", "models"))

def load_resource(name):
    full_path = os.path.join(MODEL_PATH, name)
    if not os.path.exists(full_path):
        st.error(f"Missing File: {full_path}")
        return None
    return joblib.load(full_path)

# Load the Ensemble "Super Model" components
model = load_resource("ensemble_model.pkl")
scaler = load_resource("scaler40.pkl") # Note: using your specific 40-feature scaler
encoder = load_resource("label_encoder.pkl")

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)

st.title("🤟 ISL Recognition (Ensemble Super Model)")
st.info("This model uses distance-based transformations for higher accuracy.")

# ==========================================
# 3. CORE PROCESSING CALLBACK
# ==========================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirroring for user
    
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # 126 raw coordinates (21 landmarks * 3 axis * 2 hands)
    row = [0] * 126 

    if results.multi_hand_landmarks:
        for res_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if res_idx >= 2: break 
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Populate raw coords
            for i, lm in enumerate(hand_landmarks.landmark):
                index = res_idx * 63 + i * 3
                row[index] = lm.x
                row[index+1] = lm.y
                row[index+2] = lm.z

        # --- FEATURE CONVERSION (40 DISTANCES) ---
        distance_features = []
        
        # Left Hand: Wrist (index 0,1,2)
        lx0, ly0, lz0 = row[0], row[1], row[2]
        for i in range(1, 21):
            lx, ly, lz = row[i*3], row[i*3+1], row[i*3+2]
            # Only compute if wrist is detected
            dist = np.sqrt((lx - lx0)**2 + (ly - ly0)**2 + (lz - lz0)**2) if (lx0 != 0 or ly0 != 0) else 0
            distance_features.append(dist)
            
        # Right Hand: Wrist (index 63,64,65)
        rx0, ry0, rz0 = row[63], row[64], row[65]
        for i in range(1, 21):
            rx, ry, rz = row[63 + i*3], row[63 + i*3+1], row[63 + i*3+2]
            dist = np.sqrt((rx - rx0)**2 + (ry - ry0)**2 + (rz - rz0)**2) if (rx0 != 0 or ry0 != 0) else 0
            distance_features.append(dist)

        # --- PREDICTION ---
        try:
            features = scaler.transform([distance_features])
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
            
            prediction = model.predict(features)
            label = encoder.inverse_transform(prediction)[0]

            # --- UI FEEDBACK ---
            if confidence >= 30.0:
                text = f"Ensemble: {label} ({confidence:.1f}%)"
                color = (0, 255, 0) # Green for confidence
            else:
                text = f"Searching... ({confidence:.1f}%)"
                color = (0, 165, 255) # Orange for low confidence
                
            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception:
            pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. WEBRTC STREAMER
# ==========================================
webrtc_streamer(
    key="isl-ensemble",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)
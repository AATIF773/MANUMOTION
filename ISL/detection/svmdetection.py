import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque

# ==========================================
# 1. DYNAMIC PATH SETUP (The Fix)
# ==========================================
# This finds the directory where THIS file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If this file is in 'pages/', we go up one level then into 'ISL/models'
# If this file is in 'ISL/detection/', we go up one level then into 'models'
if "pages" in BASE_DIR or "detection" in BASE_DIR:
    MODEL_PATH = os.path.join(BASE_DIR, "..", "ISL", "models")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "ISL", "models")

# Helper to load models safely
def load_resource(name):
    full_path = os.path.abspath(os.path.join(MODEL_PATH, name))
    if not os.path.exists(full_path):
        st.error(f"Missing File: {full_path}")
        return None
    return joblib.load(full_path)

# Load the SVM components
model = load_resource("svm_model.pkl")
scaler = load_resource("scaler.pkl")
encoder = load_resource("label_encoder.pkl")

# ==========================================
# 2. MEDIAPIPE & UI SETUP
# ==========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

st.title("🤟 ISL Recognition (SVM Mode)")
st.write("Live hand gesture recognition for Indian Sign Language.")

# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirror for user experience
    
    h, w, _ = img.shape
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Initialize 126 features (21 landmarks * 3 coords * 2 hands)
    coords = [0] * 126 

    if results.multi_hand_landmarks:
        for res_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if res_idx >= 2: break # Only process two hands
            
            # Draw landmarks on the screen
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Fill the feature array
            for i, lm in enumerate(hand_landmarks.landmark):
                index = res_idx * 63 + i * 3
                coords[index] = lm.x
                coords[index+1] = lm.y
                coords[index+2] = lm.z

        # Prediction Logic
        try:
            features = scaler.transform([coords])
            prediction = model.predict(features)
            label = encoder.inverse_transform(prediction)[0]
            
            # Get Probability/Confidence
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs) * 100

            # UI Overlay
            color = (0, 255, 0) if confidence > 70 else (0, 165, 255)
            cv2.putText(img, f"{label} ({confidence:.1f}%)", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        except Exception as e:
            pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. DEPLOYMENT CONFIGURATION
# ==========================================
webrtc_streamer(
    key="isl-svm",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)
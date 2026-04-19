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
# 1. DYNAMIC PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correctly navigating to the models folder based on current script location
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

# Load MLP Components
model = load_resource("mlp_model.pkl")
scaler = load_resource("scaler.pkl")
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

st.title("🤟 ISL Recognition (MLP Neural Network)")
st.write("Real-time Sign Language detection using a Multi-Layer Perceptron.")

# ==========================================
# 3. CORE PROCESSING CALLBACK
# ==========================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Flip for mirror effect
    
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Feature vector for two hands (21 landmarks * 3 axis * 2 hands)
    coords = [0] * 126 

    if results.multi_hand_landmarks:
        for res_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if res_idx >= 2: break 
            
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Map landmarks to coordinate list
            for i, lm in enumerate(hand_landmarks.landmark):
                index = res_idx * 63 + i * 3
                coords[index] = lm.x
                coords[index+1] = lm.y
                coords[index+2] = lm.z

        try:
            # 1. Scaling and Prediction
            features = scaler.transform([coords])
            prediction = model.predict(features)
            label = encoder.inverse_transform(prediction)[0]
            
            # 2. Calculating Confidence
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs) * 100

            # 3. Visual Feedback
            # Change color based on confidence (Green for high, Orange for low)
            text_color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
            
            cv2.putText(img, f"MLP: {label} ({confidence:.1f}%)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        except Exception:
            # Silently handle frames where prediction might fail
            pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. WEBRTC STREAMER SETUP
# ==========================================
webrtc_streamer(
    key="isl-mlp",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)
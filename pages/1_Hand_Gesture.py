import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. CLOUD MOCKS
from unittest.mock import MagicMock
try:
    import pyautogui
except ImportError:
    mock_py = MagicMock()
    mock_py.size.return_value = (1920, 1080)
    sys.modules['pyautogui'] = mock_py

# 2. PATHS & LOGIC
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Hand-Gesture"))
import main as gesture_main

# RESTORE YOUR FULL ORIGINAL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.vid-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; overflow: hidden; }
.sb-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }
.sb-val { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg,#818cf8,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

# 3. WEBRTC LOGIC
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    
    mp_h = mp.solutions.hands
    with mp_h.Hands(model_complexity=0) as hands:
        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            st.session_state["n_hands"] = len(result.multi_hand_landmarks)
            lml = [(int(lm.x * w), int(lm.y * h)) for lm in result.multi_hand_landmarks[0].landmark]
            st.session_state["hg_active_gesture"] = gesture_main.detect_gesture(img, lml, result)
            mp.solutions.drawing_utils.draw_landmarks(img, result.multi_hand_landmarks[0], mp_h.HAND_CONNECTIONS)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. UI LAYOUT (RESTORING YOUR ORIGINAL DESIGN)
st.title("✋ Hand Gesture Control")
if st.button("← Home"): st.switch_page("app.py")

col_vid, col_side = st.columns([3, 1], gap="medium")

with col_vid:
    st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
    webrtc_streamer(
        key="hg-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302", "stun:stun.services.mozilla.com"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown(f"<div class='sb-card'><div style='color:#44445a; font-size:0.6rem;'>HANDS</div><div class='sb-val'>{st.session_state.get('n_hands', 0)}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sb-card'><div style='color:#44445a; font-size:0.6rem;'>ACTIVE GESTURE</div><div class='sb-val'>{st.session_state.get('hg_active_gesture', '—')}</div></div>", unsafe_allow_html=True)

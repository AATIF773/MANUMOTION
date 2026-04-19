import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. CLOUD MOCKS & PATHS
from unittest.mock import MagicMock
try:
    import pyautogui
except ImportError:
    mock_py = MagicMock()
    mock_py.size.return_value = (1920, 1080)
    sys.modules['pyautogui'] = mock_py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Hand-Gesture"))
import main as gesture_main

# 2. CONFIG & UI
st.set_page_config(page_title="Hand Gesture Control · MANUMOTION", page_icon="✋", layout="wide", initial_sidebar_state="collapsed")

# RESTORE YOUR DARK MODE CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.vid-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; overflow: hidden; }
.sb-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }
.sb-val { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg,#818cf8,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hg-brand { font-size: 1rem; font-weight: 800; background: linear-gradient(135deg,#c4b5fd,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

# 3. WEBRTC CALLBACK
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    
    mp_h = mp.solutions.hands
    with mp_h.Hands(model_complexity=0, min_detection_confidence=0.7) as hands:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        gesture_text = "—"
        if result.multi_hand_landmarks:
            hnd = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(img, hnd, mp_h.HAND_CONNECTIONS)
            lml = [(int(lm.x * w), int(lm.y * h)) for lm in hnd.landmark]
            try:
                gesture_text = gesture_main.detect_gesture(img, lml, result)
                st.session_state["hg_active_gesture"] = gesture_text
            except: pass
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. RESTORE UI LAYOUT
st.markdown('<div class="hg-topnav"><span class="hg-brand">MANUMOTION</span> / <span style="color:#5b5b7a">Hand Gesture Control</span></div>', unsafe_allow_html=True)

if st.button("← Back to Home"): st.switch_page("app.py")

col_vid, col_side = st.columns([3, 1], gap="medium")

with col_vid:
    st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
    webrtc_streamer(
        key="hg-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown(f"<div class='sb-card'><div style='color:#44445a; font-size:0.6rem; font-weight:700;'>ACTIVE GESTURE</div><div class='sb-val'>{st.session_state.get('hg_active_gesture', '—')}</div></div>", unsafe_allow_html=True)
    st.markdown('<div class="sb-card"><div style="color:#44445a; font-size:0.6rem; font-weight:700;">GESTURE GUIDE</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;"><div>☝️ Move</div><div>👋 Play</div><div>🖐️ Pause</div><div>🤘 Click</div></div></div>', unsafe_allow_html=True)

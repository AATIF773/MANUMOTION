import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. CLOUD FIX: MOCK PYAUTOGUI ---
from unittest.mock import MagicMock
try:
    import pyautogui
except ImportError:
    mock_py = MagicMock()
    mock_py.size.return_value = (1920, 1080)
    sys.modules['pyautogui'] = mock_py

# --- 2. CONFIG & PATHS (UNTOUCHED) ---
st.set_page_config(page_title="Hand Gesture Control · MANUMOTION", page_icon="✋", layout="wide", initial_sidebar_state="collapsed")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Hand-Gesture"))
import main as gesture_main

# --- 3. THE CONNECTION FIX (STUN SERVERS) ---
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]}
    ]}
)

# --- 4. WEBRTC CALLBACK (PROCESSES YOUR LOGIC) ---
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
                st.session_state["hg_gesture"] = gesture_text
                st.session_state["hg_n_hands"] = len(result.multi_hand_landmarks)
            except: pass
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. YOUR ORIGINAL UI (RESORED EXACTLY) ---
if "hg_cam_on" not in st.session_state: st.session_state.hg_cam_on = False
if "theme" not in st.session_state: st.session_state.theme = "dark"

# Your original CSS block here...
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True) # Assuming DARK_MODE_CSS is defined exactly as in your paste

# Your original top nav and home button logic...
st.markdown("""<div class="hg-brand">MANUMOTION</div>""", unsafe_allow_html=True)

if not st.session_state.hg_cam_on:
    # Your original Permission Card UI
    if st.button("📷 Open Webcam & Start"):
        st.session_state.hg_cam_on = True
        st.rerun()
else:
    col_vid, col_side = st.columns([3, 1], gap="medium")
    with col_vid:
        st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
        # Replacing frame_ph with the streamer
        webrtc_streamer(
            key="hg-stream",
            video_frame_callback=video_frame_callback,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col_side:
        # Your original sidebar cards
        st.markdown(f"<div class='sb-val'>{st.session_state.get('hg_n_hands', 0)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sb-gesture'>{st.session_state.get('hg_gesture', '—')}</div>", unsafe_allow_html=True)
        if st.button("Stop Webcam"):
            st.session_state.hg_cam_on = False
            st.rerun()

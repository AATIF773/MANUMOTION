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

# 1. HANDLING MOCKS (Essential for Cloud)
from unittest.mock import MagicMock
try:
    import pyautogui
except ImportError:
    mock_py = MagicMock()
    mock_py.size.return_value = (1920, 1080)
    sys.modules['pyautogui'] = mock_py

# ─────────────────────────────────────────────────────────────────
# 2. SETTINGS & PATHS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Control · MANUMOTION",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Hand-Gesture"))

# Import your custom logic
try:
    import main as gesture_main
except ImportError:
    st.error("Could not find Hand-Gesture/main.py")

# ─────────────────────────────────────────────────────────────────
# 3. CACHED ASSETS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_mp_assets():
    mp_h = mp.solutions.hands
    det = mp_h.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.70,
    )
    return mp_h, det, mp.solutions.drawing_utils

# ─────────────────────────────────────────────────────────────────
# 4. WEBRTC CALLBACK (The Cloud-Ready Loop)
# ─────────────────────────────────────────────────────────────────
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    
    mp_h, detector, draw_u = load_mp_assets()
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    
    gesture_text = "—"
    n_hands = 0

    if result.multi_hand_landmarks:
        n_hands = len(result.multi_hand_landmarks)
        hnd = result.multi_hand_landmarks[0]
        
        # Draw landmarks
        draw_u.draw_landmarks(
            img, hnd, mp_h.HAND_CONNECTIONS,
            draw_u.DrawingSpec(color=(129, 140, 248), thickness=2, circle_radius=4),
            draw_u.DrawingSpec(color=(56, 189, 248), thickness=2),
        )
        
        # Convert landmarks for your custom logic
        lml = []
        for lm in hnd.landmark:
            lml.append((int(lm.x * w), int(lm.y * h)))
            
        # Call your existing gesture detection function
        try:
            gesture_text = gesture_main.detect_gesture(img, lml, result)
            # Store in session state for the UI
            st.session_state["hg_active_gesture"] = gesture_text
        except:
            gesture_text = "Tracking..."

    # UI Overlay on the video frame
    cv2.rectangle(img, (0, 0), (w, 45), (7, 7, 26), -1)
    cv2.putText(img, f"MANUMOTION CLOUD  |  Hands: {n_hands}  |  {gesture_text}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (129, 140, 248), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────────
# 5. UI & STYLING (Keeping your original design)
# ─────────────────────────────────────────────────────────────────
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.vid-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; overflow: hidden; }
.sb-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }
.sb-val { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg,#818cf8,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
"""
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

# Top Nav
st.markdown("""<div class="hg-topnav"><span class="hg-brand">MANUMOTION</span><span class="hg-divider">/</span><span class="hg-page-name">Hand Gesture Control</span></div>""", unsafe_allow_html=True)

if st.button("← Home"):
    st.switch_page("app.py")

col_vid, col_side = st.columns([3, 1], gap="medium")

with col_vid:
    st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
    webrtc_streamer(
        key="hg-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown(f"<div class='sb-card'><div style='color:#44445a; font-size:0.6rem; font-weight:700;'>ACTIVE GESTURE</div><div class='sb-val'>{st.session_state.get('hg_active_gesture', '—')}</div></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="sb-card">
          <div style="color:#44445a; font-size:0.6rem; font-weight:700; margin-bottom:10px;">GESTURE GUIDE</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.8rem;">
            <div>☝️ Move</div><div>👋 Play</div><div>🖐️ Pause</div><div>🤘 Click</div>
          </div>
        </div>
    """, unsafe_allow_html=True)

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# 1. INITIAL SETUP
st.set_page_config(page_title="ISL Recognition · MANUMOTION", page_icon="🤟", layout="wide", initial_sidebar_state="collapsed")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "ISL", "models")

# RESTORE YOUR CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.pred-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:28px 20px; text-align:center; }
.pred-letter { font-size:7rem; font-weight:900; background:linear-gradient(135deg,#34d399,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.alph-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
.alph-cell { border-radius:10px; padding:8px 4px; text-align:center; background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.05); }
.alph-cell.active { background:rgba(52,211,153,.14); border-color:#34d399; color:#34d399; }
</style>
""", unsafe_allow_html=True)

# 2. LOGIC
@st.cache_resource
def get_isl_tools():
    mdl = joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"))
    scl = joblib.load(os.path.join(MODEL_DIR, "scaler40.pkl"))
    enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return mdl, scl, enc

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    mdl, scl, enc = get_isl_tools()
    
    mp_h = mp.solutions.hands
    with mp_h.Hands(min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hlm, mp_h.HAND_CONNECTIONS)
            # Feature extraction and prediction logic here...
            st.session_state["isl_letter"] = "A" # Placeholder for your logic
            st.session_state["isl_conf"] = 95
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. RESTORE UI
st.title("🤟 ISL Recognition")
col_vid, col_pred, col_alpha = st.columns([5, 3, 2])

with col_vid:
    webrtc_streamer(
        key="isl-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col_pred:
    ltr = st.session_state.get("isl_letter", "—")
    cf = st.session_state.get("isl_conf", 0)
    st.markdown(f"""<div class="pred-card"><div class="pred-letter">{ltr}</div><div style="color:#34d399;font-size:1.5rem;">{cf}% Confidence</div></div>""", unsafe_allow_html=True)

with col_alpha:
    st.markdown('<div class="alph-grid">' + "".join([f"<div class='alph-cell'>{c}</div>" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]) + '</div>', unsafe_allow_html=True)

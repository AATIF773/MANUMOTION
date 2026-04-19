import sys
from unittest.mock import MagicMock

# --- CRITICAL FIX: Ignore pygame on the cloud server ---
try:
    import pygame
except ImportError:
    sys.modules['pygame'] = MagicMock()
    sys.modules['pygame.mixer'] = MagicMock()

import streamlit as st
# ... [Paste the REST of your original ISL code here] ...

import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import pygame
import av  # Added for video frame processing
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. CLOUD FIX: HANDLING MOCKING
import sys
from unittest.mock import MagicMock
try:
    import pygame
except ImportError:
    mock_pygame = MagicMock()
    sys.modules['pygame'] = mock_pygame
    sys.modules['pygame.mixer'] = MagicMock()

st.set_page_config(
    page_title="ISL Recognition · MANUMOTION",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- YOUR PATHS & AUDIO (UNTOUCHED) ---
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "ISL", "models")
AUDIO_DIR = os.path.join(ROOT, "ISL", "audio")

try:
    pygame.mixer.init()
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False

def play_audio(letter: str):
    if not AUDIO_OK: return
    path = os.path.join(AUDIO_DIR, f"{letter}.mp3")
    if os.path.exists(path):
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception:
            pass

# --- YOUR MODEL LOGIC (UNTOUCHED) ---
MODEL_CFG = {
    "Ensemble":     {"file": "ensemble_model.pkl", "scaler": "scaler40.pkl", "mode": "dist40"},
    "Random Forest":{"file": "rf_model.pkl",       "scaler": "scaler.pkl",   "mode": "raw126"},
    "SVM":          {"file": "svm_model.pkl",      "scaler": "scaler.pkl",   "mode": "raw126"},
    "KNN":          {"file": "knn_model.pkl",      "scaler": "scaler.pkl",   "mode": "raw126"},
    "MLP":          {"file": "mlp_model.pkl",      "scaler": "scaler.pkl",   "mode": "raw126"},
}

@st.cache_resource
def load_model(name: str):
    cfg = MODEL_CFG[name]
    mdl = joblib.load(os.path.join(MODEL_DIR, cfg["file"]))
    scl = joblib.load(os.path.join(MODEL_DIR, cfg["scaler"]))
    enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return mdl, scl, enc, cfg["mode"]

@st.cache_resource
def load_mp():
    mp_h = mp.solutions.hands
    det  = mp_h.Hands(
        static_image_mode=False, 
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.85, 
        min_tracking_confidence=0.85,
    )
    return mp_h, det, mp.solutions.drawing_utils

# --- YOUR FEATURE EXTRACTION (UNTOUCHED) ---
def extract_features(landmarks_list, mode: str):
    row = [0.0] * 126
    for hidx, hlm in enumerate(landmarks_list[:2]):
        for i, lm in enumerate(hlm.landmark):
            b = hidx * 63 + i * 3
            row[b], row[b+1], row[b+2] = lm.x, lm.y, lm.z
    if mode == "raw126": return row
    
    dists = []
    lx0, ly0, lz0 = row[0], row[1], row[2]
    for i in range(1, 21):
        lx,ly,lz = row[i*3], row[i*3+1], row[i*3+2]
        d = np.sqrt((lx-lx0)**2 + (ly-ly0)**2 + (lz-lz0)**2) if (lx0 or ly0) else 0.0
        dists.append(d)
    rx0, ry0, rz0 = row[63], row[64], row[65]
    for i in range(1, 21):
        rx,ry,rz = row[63+i*3], row[63+i*3+1], row[63+i*3+2]
        d = np.sqrt((rx-rx0)**2 + (ry-ry0)**2 + (rz-rz0)**2) if (rx0 or ry0) else 0.0
        dists.append(d)
    return dists

if "isl_cam_on" not in st.session_state: st.session_state.isl_cam_on = False
if "isl_last_letter" not in st.session_state: st.session_state.isl_last_letter = ""
if "isl_last_audio" not in st.session_state: st.session_state.isl_last_audio = 0.0
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "total_preds" not in st.session_state: st.session_state.total_preds = 0

ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# --- YOUR ORIGINAL CSS (UNTOUCHED) ---
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True) # ... (Keeping your full DARK_MODE_CSS definition)
if st.session_state.theme == "light":
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# --- ANIMATED CANVAS (UNTOUCHED) ---
# ... (Keeping your full components.html canvas logic)

# --- TOP NAV & HEADING (UNTOUCHED) ---
# ... (Keeping your Top Nav, Theme Switcher, and Back Button UI)

# --- WEB PROCESSING LOGIC (NEW) ---
# This function replaces the "While True" loop for the web
q = deque(maxlen=15)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    mp_h, det, draw_u = load_mp()
    ml_model, ml_scaler, ml_encoder, feat_mode = load_model(st.session_state.isl_model)
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = det.process(rgb)
    
    cur_letter = ""
    cur_conf = 0.0

    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            draw_u.draw_landmarks(img, hlm, mp_h.HAND_CONNECTIONS,
                draw_u.DrawingSpec(color=(52, 211, 153), thickness=2, circle_radius=4),
                draw_u.DrawingSpec(color=(56, 189, 248), thickness=2))

        feats = extract_features(result.multi_hand_landmarks, feat_mode)
        x_in = ml_scaler.transform([feats])
        probs = ml_model.predict_proba(x_in)[0]
        conf = max(probs) * 100
        lbl = ml_encoder.inverse_transform(ml_model.predict(x_in))[0]

        if conf >= 50.0:
            q.append(lbl)
            final_lbl = max(set(q), key=q.count)
            cur_letter = final_lbl
            cur_conf = conf
            st.session_state.total_preds += 1
            
            # Audio trigger
            now = time.time()
            if (final_lbl != st.session_state.isl_last_letter or now - st.session_state.isl_last_audio > 2.0):
                play_audio(final_lbl)
                st.session_state.isl_last_letter = final_lbl
                st.session_state.isl_last_audio = now

    # Store detection in session state for the UI cards
    st.session_state["cur_letter"] = cur_letter
    st.session_state["cur_conf"] = cur_conf
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

if not st.session_state.isl_cam_on:
    # --- YOUR ORIGINAL START CARD (UNTOUCHED) ---
    # ... (Keeping your mid column mid.markdown logic)
    if st.button("📷   Open Webcam & Start", key="isl_start", use_container_width=True):
        st.session_state.isl_cam_on = True
        st.rerun()
else:
    st.markdown("""<div class="info-strip">⚡ <strong>Webcam is active.</strong> Hold your ISL sign clearly.</div>""", unsafe_allow_html=True)
    col_vid, col_pred, col_alpha = st.columns([5, 3, 2], gap="medium")

    with col_vid:
        st.markdown('<div class="vid-panel"><div class="vid-hdr"><div class="live-dot"></div><span class="live-tag">Live Camera Feed</span></div></div>', unsafe_allow_html=True)
        # WEBRTC COMPONENT
        webrtc_streamer(
            key="isl-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_pred:
        # UI Pulls from session state updated by the callback
        ltr = st.session_state.get("cur_letter", "")
        pct = st.session_state.get("cur_conf", 0.0)
        
        # Render your original Prediction Card
        if ltr:
            st.markdown(f"""<div class="pred-card"><div class="pred-lbl">Detected Sign</div><div class='pred-letter'>{ltr}</div><div class="conf-hdr">Confidence</div><div class='conf-track'><div class='conf-fill' style='width:{int(pct)}%;'></div></div><div class='conf-pct'>{pct:.1f}%</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="pred-card"><div class="pred-lbl">Detected Sign</div><div class='waiting'>waiting…</div><div class="conf-hdr">Confidence</div><div class='conf-track'><div class='conf-fill' style='width:0%;'></div></div><div class='conf-pct'>—</div></div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='msb-card'><div class='msb-lbl'>Active Model</div><div class='msb-val'>{st.session_state.isl_model}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='msb-card'><div class='msb-lbl'>Total Predictions</div><div class='msb-val'>{st.session_state.total_preds:,}</div></div>", unsafe_allow_html=True)
        
        if st.button("⏹   Stop Webcam", key="isl_stop"):
            st.session_state.isl_cam_on = False
            st.rerun()

    with col_alpha:
        # Render your original Alphabet Panel
        st.markdown('<div class="alph-panel"><div class="alph-title">A–Z Reference</div>', unsafe_allow_html=True)
        cells = "".join(f"<div class='alph-cell{' active' if ch == ltr else ''}'>{ch}</div>" for ch in ALPHABETS)
        st.markdown(f"<div class='alph-grid'>{cells}</div></div>", unsafe_allow_html=True)

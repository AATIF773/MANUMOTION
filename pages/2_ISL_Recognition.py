import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.set_page_config(page_title="ISL Recognition · MANUMOTION", layout="wide", initial_sidebar_state="collapsed")

# --- 1. FULL ORIGINAL CSS ---
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.pred-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:28px 20px; text-align:center; }
.pred-letter { font-size:7rem; font-weight:900; background:linear-gradient(135deg,#34d399,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.alph-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
.alph-cell { border-radius:10px; padding:8px 4px; text-align:center; background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.05); }
.alph-cell.active { background:rgba(52,211,153,.14); border-color:#34d399; color:#34d399; }
</style>
"""
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

# --- 2. YOUR EXACT FEATURE EXTRACTION ---
def extract_features(landmarks_list, mode: str):
    row = [0.0] * 126
    for hidx, hlm in enumerate(landmarks_list[:2]):
        for i, lm in enumerate(hlm.landmark):
            b = hidx * 63 + i * 3
            row[b], row[b+1], row[b+2] = lm.x, lm.y, lm.z
    if mode == "raw126": return row
    dists = []
    for hidx in [0, 63]:
        x0, y0, z0 = row[hidx], row[hidx+1], row[hidx+2]
        for i in range(1, 21):
            x,y,z = row[hidx+i*3], row[hidx+i*3+1], row[hidx+i*3+2]
            d = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) if (x0 or y0) else 0.0
            dists.append(d)
    return dists

# --- 3. STREAMING LOGIC ---
@st.cache_resource
def load_isl_assets():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    M_DIR = os.path.join(ROOT, "ISL", "models")
    mdl = joblib.load(os.path.join(M_DIR, "ensemble_model.pkl"))
    scl = joblib.load(os.path.join(M_DIR, "scaler40.pkl"))
    enc = joblib.load(os.path.join(M_DIR, "label_encoder.pkl"))
    return mdl, scl, enc

isl_q = deque(maxlen=15)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    mdl, scl, enc = load_isl_assets()
    
    mp_h = mp.solutions.hands
    with mp_h.Hands(min_detection_confidence=0.7) as hands:
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            feats = extract_features(res.multi_hand_landmarks, "dist40")
            x_in = scl.transform([feats])
            probs = mdl.predict_proba(x_in)[0]
            conf = max(probs) * 100
            lbl = enc.inverse_transform(mdl.predict(x_in))[0]
            
            if conf > 50:
                isl_q.append(lbl)
                final_ltr = max(set(isl_q), key=isl_q.count)
                st.session_state["isl_ltr"] = final_ltr
                st.session_state["isl_conf"] = conf
            
            for hlm in res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hlm, mp_h.HAND_CONNECTIONS)
                
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI STRUCTURE ---
st.title("🤟 ISL Recognition")
col_vid, col_pred, col_alpha = st.columns([5, 3, 2], gap="medium")

with col_vid:
    webrtc_streamer(key="isl-s", video_frame_callback=video_frame_callback,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun.services.mozilla.com"]}]})

with col_pred:
    ltr = st.session_state.get("isl_ltr", "—")
    st.markdown(f'<div class="pred-card"><div class="pred-letter">{ltr}</div><div style="color:#34d399;">{int(st.session_state.get("isl_conf", 0))}% Confidence</div></div>', unsafe_allow_html=True)

with col_alpha:
    cells = "".join([f"<div class='alph-cell {'active' if c == ltr else ''}'>{c}</div>" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    st.markdown(f'<div class="alph-grid">{cells}</div>', unsafe_allow_html=True)

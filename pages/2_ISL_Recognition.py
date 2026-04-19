import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ─────────────────────────────────────────────────────────────────
# 1. SETUP & PATHS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Recognition · MANUMOTION",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Force Dark Mode CSS
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.vid-panel { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:24px; overflow:hidden; }
.pred-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:28px 20px; text-align:center; }
.pred-letter { font-size:7rem; font-weight:900; background:linear-gradient(135deg,#34d399,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
</style>
"""
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "ISL", "models")

# ─────────────────────────────────────────────────────────────────
# 2. CACHED ASSETS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_isl_assets(model_name):
    # Map configurations
    cfg = {
        "Ensemble":      {"file": "ensemble_model.pkl", "scaler": "scaler40.pkl", "mode": "dist40"},
        "Random Forest": {"file": "rf_model.pkl",       "scaler": "scaler.pkl",   "mode": "raw126"},
    }.get(model_name, {"file": "ensemble_model.pkl", "scaler": "scaler40.pkl", "mode": "dist40"})
    
    mdl = joblib.load(os.path.join(MODEL_DIR, cfg["file"]))
    scl = joblib.load(os.path.join(MODEL_DIR, cfg["scaler"]))
    enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    
    return mdl, scl, enc, cfg["mode"], hands, mp.solutions.drawing_utils, mp_hands

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_features(landmarks_list, mode):
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
            x, y, z = row[hidx+i*3], row[hidx+i*3+1], row[hidx+i*3+2]
            d = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) if (x0 or y0) else 0.0
            dists.append(d)
    return dists

# ─────────────────────────────────────────────────────────────────
# 4. WEBRTC VIDEO CALLBACK (The magic happens here)
# ─────────────────────────────────────────────────────────────────
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    # Access the selected model from session state
    sel_model = st.session_state.get("sel_model", "Ensemble")
    mdl, scl, enc, mode, hands, draw_u, mp_h = load_isl_assets(sel_model)
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    label_text = "Searching..."
    conf_val = 0

    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            draw_u.draw_landmarks(img, hlm, mp_h.HAND_CONNECTIONS)
        
        feats = extract_features(result.multi_hand_landmarks, mode)
        x_in = scl.transform([feats])
        probs = mdl.predict_proba(x_in)[0]
        conf_val = max(probs) * 100
        
        if conf_val > 50:
            label_text = enc.inverse_transform(mdl.predict(x_in))[0]
            # Update prediction in state for UI display
            st.session_state["last_pred"] = label_text
            st.session_state["last_conf"] = conf_val

    # Draw overlay on video
    cv2.rectangle(img, (0,0), (250, 60), (7,7,26), -1)
    cv2.putText(img, f"{label_text} ({int(conf_val)}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (52,211,153), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────────
# 5. UI LAYOUT
# ─────────────────────────────────────────────────────────────────
st.title("🤟 ISL Recognition")
sel_model = st.selectbox("Select Model", ["Ensemble", "Random Forest"], key="sel_model")

col_vid, col_res = st.columns([2, 1])

with col_vid:
    st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
    webrtc_streamer(
        key="isl-streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    res_placeholder = st.empty()
    # Pull updates from session state
    ltr = st.session_state.get("last_pred", "—")
    cf = st.session_state.get("last_conf", 0)
    
    res_placeholder.markdown(f"""
    <div class="pred-card">
        <div style="color:#5b5b7a; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;">Detected Sign</div>
        <div class="pred-letter">{ltr}</div>
        <div style="color:#34d399; font-size:1.5rem; font-weight:800;">{int(cf)}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)

if st.button("← Back to Home"):
    st.switch_page("app.py")

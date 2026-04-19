import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import sys
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ─────────────────────────────────────────────────────────────────
# 1. CLOUD FIXES: MOCKING PYGAME (Prevents ModuleNotFoundError)
# ─────────────────────────────────────────────────────────────────
from unittest.mock import MagicMock
try:
    import pygame
except ImportError:
    mock_pygame = MagicMock()
    sys.modules['pygame'] = mock_pygame
    sys.modules['pygame.mixer'] = MagicMock()

# ─────────────────────────────────────────────────────────────────
# 2. INITIAL CONFIG & PATHS (UNTOUCHED)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Recognition · MANUMOTION",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "ISL", "models")
AUDIO_DIR = os.path.join(ROOT, "ISL", "audio")

try:
    import pygame
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

# ─────────────────────────────────────────────────────────────────
# SESSION STATE (Ensuring no errors!)
# ─────────────────────────────────────────────────────────────────
if "isl_cam_on" not in st.session_state: st.session_state.isl_cam_on = False
if "isl_last_letter" not in st.session_state: st.session_state.isl_last_letter = ""
if "isl_last_audio" not in st.session_state: st.session_state.isl_last_audio = 0.0
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "total_preds" not in st.session_state: st.session_state.total_preds = 0

ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ─────────────────────────────────────────────────────────────────
# CSS (UNTOUCHED - Restored from your paste)
# ─────────────────────────────────────────────────────────────────
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body, html, [data-testid="stApp"], .stApp, [data-testid="stAppViewContainer"] { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"],footer,#MainMenu { display: none !important; }
.block-container,[data-testid="stAppViewBlockContainer"] { padding: 32px 48px 48px !important; max-width: 100% !important; }
#isl-canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: 0; pointer-events: none; }
.isl-topnav { display:flex; align-items:center; gap:20px; margin-bottom:36px; }
.isl-brand { font-size:1rem; font-weight:800; letter-spacing:1px; background:linear-gradient(135deg,#c4b5fd,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.isl-divider { color:#333350; font-size:1.2rem; }
.isl-pname { font-size:1rem; font-weight:600; color:#5b5b7a; }
.section-title { font-size:2rem; font-weight:800; letter-spacing:-1px; background:linear-gradient(130deg,#fff,#34d399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.section-sub { font-size:.9rem; color:#5b5b7a; margin-top:8px; margin-bottom:24px; }
.model-row { display:flex; align-items:center; gap:16px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:14px 20px; margin-bottom:20px; }
.model-lbl { font-size:.7rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#44445a; white-space:nowrap; }
.perm-card { background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.07); border-radius:28px; padding:64px 48px; max-width:540px; text-align:center; backdrop-filter:blur(24px); animation:up .6s ease both; }
@keyframes up { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:none} }
.perm-icon { font-size:5rem; display:block; margin-bottom:24px; animation:pls 2s ease infinite; }
@keyframes pls { 0%,100%{transform:scale(1)} 50%{transform:scale(1.07)} }
.perm-h { font-size:1.7rem; font-weight:800; color:#f0f0fa; margin-bottom:12px; }
.perm-p { font-size:.92rem; color:#6b6b8a; line-height:1.75; margin-bottom:28px; }
.perm-pills { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-bottom:36px; }
.perm-pill { font-size:.68rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; padding:6px 16px; border-radius:99px; background:rgba(52,211,153,.1); border:1px solid rgba(52,211,153,.25); color:#34d399; }
.live-wrap { display:grid; grid-template-columns: 1fr 260px 220px; gap:20px; align-items:start; width:100%; }
.vid-panel { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:24px; overflow:hidden; }
.vid-hdr { display:flex; align-items:center; gap:10px; padding:13px 18px; border-bottom:1px solid rgba(255,255,255,.05); }
.live-dot { width:8px; height:8px; border-radius:50%; background:#ef4444; animation:blink 1.2s ease infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.15} }
.live-tag { font-size:.68rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#ef4444; }
.pred-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:28px 20px; text-align:center; backdrop-filter:blur(20px); }
.pred-lbl { font-size:.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#44445a; margin-bottom:16px; }
.pred-letter { font-size:7rem; font-weight:900; line-height:1; display:block; min-height:7.5rem; background:linear-gradient(135deg,#34d399,#38bdf8,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.conf-hdr { font-size:.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#44445a; margin:18px 0 8px; }
.conf-track { width:100%; height:8px; background:rgba(255,255,255,.06); border-radius:99px; overflow:hidden; margin-bottom:8px; }
.conf-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#34d399,#38bdf8); transition:width .25s ease; }
.conf-pct { font-size:1.6rem; font-weight:800; background:linear-gradient(90deg,#34d399,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.mini-sb { margin-bottom:14px; }
.msb-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:16px 18px; margin-bottom:12px; }
.msb-lbl { font-size:.62rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#44445a; margin-bottom:5px; }
.msb-val { font-size:1.1rem; font-weight:700; color:#e8e8f5; }
.alph-panel { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:24px; padding:20px 16px; }
.alph-title { font-size:.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#44445a; margin-bottom:14px; text-align:center; }
.alph-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
.alph-cell { border-radius:10px; padding:8px 4px; text-align:center; font-size:1rem; font-weight:700; color:#3a3a5a; background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.05); transition:all .25s; }
.alph-cell.active { background:rgba(52,211,153,.14); border-color:rgba(52,211,153,.45); color:#34d399; transform:scale(1.1); box-shadow:0 0 16px rgba(52,211,153,.18); }
.info-strip { background:rgba(52,211,153,.07); border:1px solid rgba(52,211,153,.18); border-radius:12px; padding:11px 18px; font-size:.85rem; color:#6ee7b7; margin-bottom:16px; }
.waiting { font-size:1.6rem; color:#333355; text-align:center; padding:32px 0; font-weight:400; }
div[data-testid="stButton"] { display: flex !important; justify-content: center !important; width: 100% !important; }
div[data-testid="stButton"] > button { background:linear-gradient(130deg,#34d399,#38bdf8) !important; color:#fff !important; border:none !important; border-radius:14px !important; padding:14px 28px !important; font-family:'Outfit',sans-serif !important; font-size:1rem !important; font-weight:700 !important; white-space: nowrap !important; margin: 0 auto !important; cursor:pointer !important; box-shadow:0 8px 28px rgba(52,211,153,.22) !important; transition:all .3s !important; }
div[data-testid="stButton"] > button:hover { transform:translateY(-2px) !important; box-shadow:0 14px 38px rgba(52,211,153,.38) !important; filter:brightness(1.08) !important; }
.stSelectbox label { color:#9090aa !important; font-family:'Outfit',sans-serif !important; font-size:.85rem !important; }
.stSelectbox [data-baseweb="select"] > div { background:rgba(255,255,255,.06) !important; border:1px solid rgba(255,255,255,.12) !important; border-radius:12px !important; color:#fff !important; }
</style>
"""

LIGHT_MODE_CSS = """
<style>
body, html, [data-testid="stApp"], .stApp, [data-testid="stAppViewContainer"] { background: #f1f5f9 !important; color: #1e293b !important; }
.perm-card, .vid-panel, .pred-card, .msb-card, .alph-panel { background: rgba(255,255,255,1) !important; border: 1px solid rgba(0,0,0,0.04) !important; box-shadow: 0 16px 40px -12px rgba(0,0,0,0.12), 0 4px 10px rgba(0,0,0,0.04) !important; }
.perm-h, .msb-val { color: #0f172a !important; }
.perm-p, .section-sub, .isl-pname, .model-lbl, .msb-lbl, .conf-hdr, .pred-lbl, .alph-title { color: #64748b !important; }
.alph-cell { color: #334155 !important; background: #f8fafc !important; border-color: #e2e8f0 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
.isl-brand { background: linear-gradient(135deg,#818cf8,#0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.section-title { background: linear-gradient(130deg,#4f46e5,#0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stSelectbox [data-baseweb="select"] > div { background:#fff !important; border-color:#e2e8f0 !important; color:#0f172a !important; box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important; }
.stSelectbox div[data-baseweb="popover"] { background:#fff !important; color:#0f172a !important; }
.waiting { color: #94a3b8 !important; }
</style>
"""
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
if st.session_state.theme == "light":
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 3. ANIMATED CANVAS (UNTOUCHED)
# ─────────────────────────────────────────────────────────────────
components.html("""
<script>
(function(){
  var pDoc = window.parent.document;
  ["mm-canvas", "hg-canvas", "isl-canvas"].forEach(function(id) { var elem = pDoc.getElementById(id); if (elem) elem.remove(); });
  var cvs = pDoc.createElement('canvas'); cvs.id = 'isl-canvas'; pDoc.body.prepend(cvs);
  var ctx=cvs.getContext('2d'),W,H,pts=[],raf;
  function resize(){W=cvs.width=window.parent.innerWidth;H=cvs.height=window.parent.innerHeight;}
  function init(){
    pts=[];
    var HAND=[[.50,.72],[.44,.58],[.43,.44],[.43,.32],[.43,.21],[.50,.42],[.50,.29],[.50,.18],[.50,.09],[.56,.43],[.57,.30],[.57,.19],[.57,.10],[.62,.46],[.63,.35],[.63,.25],[.63,.16],[.67,.56],[.71,.47],[.73,.39],[.73,.32]];
    HAND.forEach(function(p){ pts.push({x:p[0]*W,y:p[1]*H,vx:(Math.random()-.5)*.28,vy:(Math.random()-.5)*.28,ax:p[0],ay:p[1],r:3,anchor:true}); });
    for(var i=0;i<65;i++) pts.push({ x:Math.random()*W,y:Math.random()*H, vx:(Math.random()-.5)*.45,vy:(Math.random()-.5)*.45, ax:null,ay:null,r:1.2+Math.random()*2,anchor:false });
  }
  function tick(){
    ctx.clearRect(0,0,W,H);
    pts.forEach(function(n){ if(n.anchor){n.x+=(n.ax*W-n.x)*.006+n.vx;n.y+=(n.ay*H-n.y)*.006+n.vy;} else{n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>W)n.vx*=-1;if(n.y<0||n.y>H)n.vy*=-1;} });
    for(var i=0;i<pts.length;i++) for(var j=i+1;j<pts.length;j++){ var dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy); if(d<165){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y); ctx.strokeStyle='rgba(52,211,153,'+(1-d/165)*.22+')';ctx.lineWidth=.75;ctx.stroke();} }
    pts.forEach(function(n){ ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2); ctx.fillStyle=n.anchor?'rgba(52,211,153,.75)':'rgba(56,189,248,.5)';ctx.fill(); });
    if (window.parent.islRaf) { window.parent.cancelAnimationFrame(window.parent.islRaf); }
    raf=window.parent.requestAnimationFrame(tick); window.parent.islRaf = raf;
  }
  resize();init();tick();
})();
</script>
""", height=0, width=0)

# ─────────────────────────────────────────────────────────────────
# 4. TOP NAV & SELECTION (UNTOUCHED UI)
# ─────────────────────────────────────────────────────────────────
top_col, theme_col = st.columns([9.2, 0.8])
with top_col:
    st.markdown("""
    <div style="position:relative;z-index:10;">
      <div class="isl-topnav"><span class="isl-brand">MANUMOTION</span><span class="isl-divider">/</span><span class="isl-pname">ISL Recognition</span></div>
      <div><div class="section-title">🤟 Indian Sign Language Recognition</div><div class="section-sub">Real-time ISL alphabet detection · Ensemble ML · Audio pronunciation</div></div>
    </div>
    """, unsafe_allow_html=True)
with theme_col:
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button("☀️" if st.session_state.theme == "dark" else "🌙", key="theme_btn", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

col_back, _ = st.columns([1.5, 8.5])
with col_back:
    if st.button("← Back to Home", key="isl_back", use_container_width=True):
        st.switch_page("app.py")

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

MODEL_BADGES = {
    "Ensemble":     ("⭐ Default · Best Accuracy",   "#34d399"),
    "Random Forest":("🌲 Robust · Low Variance",     "#38bdf8"),
    "SVM":          ("⚡ Fast · High Precision",      "#818cf8"),
    "KNN":          ("🔵 Instance-based · Simple",    "#f472b6"),
    "MLP":          ("🧠 Neural Network · Deep",      "#fb923c"),
}

col_mlbl, col_msel, col_mbadge = st.columns([1.5, 3, 5.5])
with col_mlbl:
    st.markdown("<div style='padding-top:8px;font-size:.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#44445a;'>🧠 Model</div>", unsafe_allow_html=True)
with col_msel:
    sel_model = st.selectbox(
        "model", list(MODEL_CFG.keys()), index=0,
        label_visibility="collapsed", key="isl_model"
    )
with col_mbadge:
    badge_txt, badge_col = MODEL_BADGES[sel_model]
    st.markdown(
        f"<div style='margin-top:4px;display:inline-block;font-size:.75rem;font-weight:700;"
        f"letter-spacing:1px;padding:7px 18px;border-radius:99px;"
        f"background:rgba(255,255,255,.04);border:1px solid {badge_col}55;color:{badge_col};'>"
        f"{badge_txt}</div>",
        unsafe_allow_html=True,
    )

model_loaded = False
try:
    ml_model, ml_scaler, ml_encoder, feat_mode = load_model(sel_model)
    model_loaded = True
except Exception as e:
    st.error(f"❌ Could not load **{sel_model}** — {e}")

# ─────────────────────────────────────────────────────────────────
# 5. WEBRTC SURGERY (FIX FOR CAPTURING)
# ─────────────────────────────────────────────────────────────────
isl_queue = deque(maxlen=15)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    mp_h, det, draw_u = load_mp()
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = det.process(rgb)
    
    cur_letter = ""
    cur_conf = 0.0

    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            draw_u.draw_landmarks(
                img, hlm, mp_h.HAND_CONNECTIONS,
                draw_u.DrawingSpec(color=(52, 211, 153), thickness=2, circle_radius=4),
                draw_u.DrawingSpec(color=(56, 189, 248), thickness=2),
            )

        feats = extract_features(result.multi_hand_landmarks, feat_mode)
        x_in = ml_scaler.transform([feats])
        probs = ml_model.predict_proba(x_in)[0]
        conf = max(probs) * 100
        lbl = ml_encoder.inverse_transform(ml_model.predict(x_in))[0]

        if conf >= 50.0:
            isl_queue.append(lbl)
            final_lbl = max(set(isl_queue), key=isl_queue.count)
            cur_letter = final_lbl
            cur_conf = conf
            st.session_state.total_preds += 1
            
            now = time.time()
            if (final_lbl != st.session_state.isl_last_letter or now - st.session_state.isl_last_audio > 2.0):
                play_audio(final_lbl)
                st.session_state.isl_last_letter = final_lbl
                st.session_state.isl_last_audio = now

    st.session_state["isl_current_letter"] = cur_letter
    st.session_state["isl_current_conf"] = cur_conf
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────────
# 6. UI RENDER (UNTOUCHED LAYOUT)
# ─────────────────────────────────────────────────────────────────
if not st.session_state.isl_cam_on:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("""
        <div class="perm-card" style="margin: 0 auto; width: 100%;">
          <span class="perm-icon">🤟</span>
          <div class="perm-h">Ready to Recognise Signs</div>
          <p class="perm-p">Point your webcam at your signing hand.<br>MANUMOTION will recognise ISL alphabets in real-time and speak each letter aloud.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        if model_loaded:
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                if st.button("📷   Open Webcam & Start", key="isl_start", use_container_width=True):
                    st.session_state.isl_cam_on = True
                    st.rerun()
else:
    st.markdown("""<div class="info-strip">⚡ <strong>Webcam is active.</strong> Hold your ISL sign clearly in front of the camera.</div>""", unsafe_allow_html=True)
    col_vid, col_pred, col_alpha = st.columns([5, 3, 2], gap="medium")

    with col_vid:
        st.markdown('<div class="vid-panel"><div class="vid-hdr"><div class="live-dot"></div><span class="live-tag">Live Camera Feed</span></div></div>', unsafe_allow_html=True)
        
        webrtc_streamer(
            key="isl-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302", "stun:stun.services.mozilla.com"]}]}
            ),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_pred:
        ltr = st.session_state.get("isl_current_letter", "")
        pct = st.session_state.get("isl_current_conf", 0.0)
        
        if ltr:
            st.markdown(f"""<div class="pred-card"><div class="pred-lbl">Detected Sign</div><div class='pred-letter'>{ltr}</div><div class="conf-hdr">Confidence</div><div class='conf-track'><div class='conf-fill' style='width:{int(pct)}%;'></div></div><div class='conf-pct'>{pct:.1f}%</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="pred-card"><div class="pred-lbl">Detected Sign</div><div class='waiting'>waiting…</div><div class="conf-hdr">Confidence</div><div class='conf-track'><div class='conf-fill' style='width:0%;'></div></div><div class='conf-pct'>—</div></div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='msb-card'><div class='msb-lbl'>Active Model</div><div class='msb-val'>{sel_model}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='msb-card'><div class='msb-lbl'>Total Predictions</div><div class='msb-val'>{st.session_state.total_preds:,}</div></div>", unsafe_allow_html=True)
        
        if st.button("⏹   Stop Webcam", key="isl_stop"):
            st.session_state.isl_cam_on = False
            st.rerun()

    with col_alpha:
        st.markdown('<div class="alph-panel"><div class="alph-title">A–Z Reference</div>', unsafe_allow_html=True)
        cells = "".join(f"<div class='alph-cell{' active' if ch == ltr else ''}'>{ch}</div>" for ch in ALPHABETS)
        st.markdown(f"<div class='alph-grid'>{cells}</div></div>", unsafe_allow_html=True)

st.markdown("""<div style="text-align: center; color: #6b6b8a; font-size: 0.75rem; padding-top: 40px; padding-bottom: 10px; font-weight: 500; letter-spacing: 1px;">&copy; 2026 MANUMOTION. All rights reserved.</div>""", unsafe_allow_html=True)

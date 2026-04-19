import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

#HANDLING pyautogui
import sys
from unittest.mock import MagicMock

# Trick the app into thinking pyautogui exists on the cloud server
try:
    import pyautogui
except ImportError:
    sys.modules['pyautogui'] = MagicMock()

st.set_page_config(
    page_title="Hand Gesture Control · MANUMOTION",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# Import gesture logic from Hand-Gesture/
# ─────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Hand-Gesture"))
import main as gesture_main

@st.cache_resource
def load_mp():
    mp_h = mp.solutions.hands
    det  = mp_h.Hands(
        static_image_mode=False,
        model_complexity=0,       # 🚀 SPEED FIX 1: Lite Model
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.70,
    )
    return mp_h, det, mp.solutions.drawing_utils

if "hg_cam_on" not in st.session_state: st.session_state.hg_cam_on = False
if "theme" not in st.session_state: st.session_state.theme = "dark"

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body, html, [data-testid="stApp"], .stApp, [data-testid="stAppViewContainer"] { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"],footer, #MainMenu { display: none !important; }
.block-container, [data-testid="stAppViewBlockContainer"] { padding: 32px 48px 48px !important; max-width: 100% !important; }
#hg-canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: 0; pointer-events: none; }
.page-content { position: relative; z-index: 10; }
.hg-topnav { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.hg-brand { font-size: 1rem; font-weight: 800; letter-spacing: 1px; background: linear-gradient(135deg,#c4b5fd,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hg-divider { color: #333350; }
.hg-page-name { font-size: 0.9rem; font-weight: 600; color: #5b5b7a; }
.page-heading { margin-bottom: 20px; }
.page-heading-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px; background: linear-gradient(130deg, #fff, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.2; }
.page-heading-sub { font-size: 0.85rem; color: #5b5b7a; margin-top: 6px; }
.perm-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 24px; padding: 48px 40px; max-width: 500px; text-align: center; backdrop-filter: blur(20px); margin: 0 auto; }
.perm-icon { font-size: 4rem; display: block; margin-bottom: 20px; animation: pls 2.5s ease infinite; }
@keyframes pls { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
.perm-h { font-size: 1.5rem; font-weight: 800; color: #f0f0fa; margin-bottom: 10px; }
.perm-p { font-size: 0.88rem; color: #6b6b8a; line-height: 1.7; margin-bottom: 24px; }
.perm-pills { display:flex; gap:7px; justify-content:center; flex-wrap:wrap; margin-bottom: 28px; }
.perm-pill { font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; padding: 5px 13px; border-radius: 99px; background: rgba(129,140,248,0.1); border: 1px solid rgba(129,140,248,0.25); color: #818cf8; }
.info-bar { background: rgba(129,140,248,0.07); border: 1px solid rgba(129,140,248,0.2); border-radius: 12px; padding: 11px 16px; font-size: 0.83rem; color: #a5b4fc; margin-bottom: 16px; }
.vid-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; overflow: hidden; }
.vid-hdr { display: flex; align-items: center; gap: 9px; padding: 11px 16px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #ef4444; animation: blink 1.2s ease infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.15} }
.live-tag { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #ef4444; }
.sb-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }
.sb-lbl { font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #44445a; margin-bottom: 6px; }
.sb-val { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg,#818cf8,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sb-gesture { font-size: 1rem; font-weight: 700; color: #e8e8f5; min-height: 1.4rem; }
.gleg { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 4px; }
.gleg-item { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 8px 4px; text-align: center; }
.gleg-e { font-size: 1.25rem; }
.gleg-n { font-size: 0.58rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #55557a; margin-top: 4px; }
div[data-testid="stButton"] { display: flex !important; justify-content: center !important; width: 100% !important; }
div[data-testid="stButton"] > button { background: linear-gradient(130deg, #818cf8, #38bdf8) !important; color: #fff !important; border: none !important; border-radius: 14px !important; padding: 14px 28px !important; font-family: 'Outfit', sans-serif !important; font-size: 1rem !important; font-weight: 700 !important; white-space: nowrap !important; margin: 0 auto !important; cursor: pointer !important; box-shadow: 0 8px 28px rgba(129,140,248,0.22) !important; transition: all 0.3s !important; }
div[data-testid="stButton"] > button:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 32px rgba(129,140,248,0.38) !important; filter: brightness(1.08) !important; }
</style>
"""

LIGHT_MODE_CSS = """
<style>
body, html, [data-testid="stApp"], .stApp, [data-testid="stAppViewContainer"] { background: #f1f5f9 !important; color: #1e293b !important; }
.perm-card, .lh-card, .lh-box { background: rgba(255,255,255,1) !important; border: 1px solid rgba(0,0,0,0.04) !important; box-shadow: 0 16px 40px -12px rgba(0,0,0,0.12), 0 4px 10px rgba(0,0,0,0.04) !important; }
.gleg-item { background: #f8fafc !important; border: 1px solid #e2e8f0 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }
.perm-h, .lh-val { color: #0f172a !important; }
.perm-p, .page-heading-sub, .hg-page-name, .lh-lbl, .gleg-n { color: #64748b !important; }
.hg-brand { background: linear-gradient(135deg,#818cf8,#0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.page-heading-title { background: linear-gradient(130deg,#4f46e5,#0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
"""

st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
if st.session_state.theme == "light":
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

components.html("""
<script>
(function(){
  var pDoc = window.parent.document;
  ["mm-canvas", "hg-canvas", "isl-canvas"].forEach(function(id) { var elem = pDoc.getElementById(id); if (elem) elem.remove(); });
  var cvs = pDoc.createElement('canvas'); cvs.id = 'hg-canvas'; pDoc.body.prepend(cvs);
  var ctx=cvs.getContext('2d'),W,H,pts=[],raf;
  function resize(){W=cvs.width=window.parent.innerWidth;H=cvs.height=window.parent.innerHeight;}
  function init(){
    pts=[];
    var HAND=[[.50,.72],[.44,.58],[.43,.44],[.43,.32],[.43,.21],[.50,.42],[.50,.29],[.50,.18],[.50,.09],[.56,.43],[.57,.30],[.57,.19],[.57,.10],[.62,.46],[.63,.35],[.63,.25],[.63,.16],[.67,.56],[.71,.47],[.73,.39],[.73,.32]];
    HAND.forEach(function(p){pts.push({x:p[0]*W,y:p[1]*H,vx:(Math.random()-.5)*.28,vy:(Math.random()-.5)*.28,ax:p[0],ay:p[1],r:2.5,anchor:true,col:'#818cf8'});});
    for(var i=0;i<55;i++) pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.45,vy:(Math.random()-.5)*.45,ax:null,ay:null,r:1.2+Math.random()*1.8,anchor:false,col:['#818cf8','#38bdf8','#34d399','#c4b5fd'][Math.floor(Math.random()*4)]});
  }
  function tick(){
    ctx.clearRect(0,0,W,H);
    pts.forEach(function(n){if(n.anchor){n.x+=(n.ax*W-n.x)*.006+n.vx;n.y+=(n.ay*H-n.y)*.006+n.vy;}else{n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>W)n.vx*=-1;if(n.y<0||n.y>H)n.vy*=-1;}});
    for(var i=0;i<pts.length;i++) for(var j=i+1;j<pts.length;j++){var dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<160){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(129,140,248,'+(1-d/160)*.22+')';ctx.lineWidth=.7;ctx.stroke();}}
    pts.forEach(function(n){ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fillStyle=n.col+(n.anchor?'bb':'55');ctx.fill();});
    if (window.parent.hgRaf) { window.parent.cancelAnimationFrame(window.parent.hgRaf); }
    raf=window.parent.requestAnimationFrame(tick); window.parent.hgRaf = raf;
  }
  resize();init();tick();
})();
</script>
""", height=0, width=0)

top_col, theme_col = st.columns([9.2, 0.8])
with top_col:
    st.markdown("""
    <div style="position:relative;z-index:10;margin-bottom:4px;">
      <div class="hg-topnav"><span class="hg-brand">MANUMOTION</span><span class="hg-divider">/</span><span class="hg-page-name">Hand Gesture Control</span></div>
      <div class="page-heading"><div class="page-heading-title">&#9994; Hand Gesture Control</div><div class="page-heading-sub">Control your OS with natural hand movements &middot; Powered by MediaPipe</div></div>
    </div>
    """, unsafe_allow_html=True)
with theme_col:
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button("☀️" if st.session_state.theme == "dark" else "🌙", key="theme_btn", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

col_back, _ = st.columns([1.5, 8.5])
with col_back:
    if st.button("← Home", key="hg_back", use_container_width=True):
        st.switch_page("app.py")

if not st.session_state.hg_cam_on:
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("""
<div class="perm-card" style="margin: 0 auto; width: 100%;">
  <span class="perm-icon">✋</span>
  <div class="perm-h">Grant Webcam Access</div>
  <p class="perm-p">MANUMOTION tracks your hand landmarks in real-time to control your OS. Everything is processed locally.</p>
</div>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("📷  Open Webcam & Start", key="hg_start", use_container_width=True):
                st.session_state.hg_cam_on = True
                st.rerun()
else:
    st.markdown("""<div class="info-bar"><strong>&#9889; Webcam active.</strong> Gesture control is running.</div>""", unsafe_allow_html=True)

    col_vid, col_side = st.columns([3, 1], gap="medium")

    with col_vid:
        st.markdown("""<div class="vid-panel"><div class="vid-hdr"><div class="live-dot"></div><span class="live-tag">Live Camera Feed</span></div></div>""", unsafe_allow_html=True)
        frame_ph = st.empty()

    with col_side:
        hand_ph = st.empty()
        fps_ph = st.empty()
        gesture_ph = st.empty()
        
        hand_ph.markdown("<div class='sb-card'><div class='sb-lbl'>Hands</div><div class='sb-val'>0</div></div>", unsafe_allow_html=True)
        fps_ph.markdown("<div class='sb-card'><div class='sb-lbl'>FPS</div><div class='sb-val'>0</div></div>", unsafe_allow_html=True)
        gesture_ph.markdown("<div class='sb-card'><div class='sb-lbl'>Active Gesture</div><div class='sb-gesture'>—</div></div>", unsafe_allow_html=True)

        st.markdown("""
<div class="sb-card">
  <div class="sb-lbl">Gesture Guide</div>
  <div class="gleg">
    <div class="gleg-item"><div class="gleg-e">&#9757;&#65039;</div><div class="gleg-n">Move</div></div>
    <div class="gleg-item"><div class="gleg-e">&#128075;</div><div class="gleg-n">Play</div></div>
    <div class="gleg-item"><div class="gleg-e">&#128080;</div><div class="gleg-n">Pause</div></div>
    <div class="gleg-item"><div class="gleg-e">&#129304;</div><div class="gleg-n">L-Click</div></div>
    <div class="gleg-item"><div class="gleg-e">&#129395;</div><div class="gleg-n">R-Click</div></div>
    <div class="gleg-item"><div class="gleg-e">&#128247;</div><div class="gleg-n">Screenshot</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

        stop_btn = st.button("⏹  Stop Webcam", key="hg_stop")

    # ── NATIVE LOCAL CAMERA LOOP (SPEED OPTIMIZED) ──
    if not stop_btn:
        try:
            mp_h, detector, draw_u = load_mp()
            cap = cv2.VideoCapture(0)
            
            # 🚀 SPEED FIX 2: Hardware Resolution Throttling
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Make sure it's not in use.")
                st.session_state.hg_cam_on = False
                st.rerun()

            prev_t = time.time()

            while True:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)

                h, w = frame.shape[:2]
                lml = []
                n_hands = 0
                gesture_text = ""

                if result.multi_hand_landmarks:
                    n_hands = len(result.multi_hand_landmarks)
                    hnd = result.multi_hand_landmarks[0]
                    draw_u.draw_landmarks(
                        frame, hnd, mp_h.HAND_CONNECTIONS,
                        draw_u.DrawingSpec(color=(129, 140, 248), thickness=2, circle_radius=4),
                        draw_u.DrawingSpec(color=(56, 189, 248), thickness=2),
                    )
                    
                    for lm in hnd.landmark:
                        lml.append((int(lm.x * w), int(lm.y * h)))
                        
                    gesture_text = gesture_main.detect_gesture(frame, lml, result)

                now = time.time()
                fps = int(1.0 / max(now - prev_t, 1e-9))
                prev_t = now

                # HUD overlay
                cv2.rectangle(frame, (0, 0), (w, 44), (7, 7, 26), -1)
                cv2.putText(frame, f"MANUMOTION  |  FPS: {fps}  |  Hands: {n_hands}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (129, 140, 248), 2)

                frame_ph.image(frame, channels="BGR", use_container_width=True)
                hand_ph.markdown(f"<div class='sb-card'><div class='sb-lbl'>Hands</div><div class='sb-val'>{n_hands}</div></div>", unsafe_allow_html=True)
                fps_ph.markdown(f"<div class='sb-card'><div class='sb-lbl'>FPS</div><div class='sb-val'>{fps}</div></div>", unsafe_allow_html=True)
                gesture_ph.markdown(f"<div class='sb-card'><div class='sb-lbl'>Active Gesture</div><div class='sb-gesture'>{gesture_text if gesture_text else '—'}</div></div>", unsafe_allow_html=True)

                if st.session_state.get("hg_stop", False):
                    break

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            try: cap.release()
            except: pass
    else:
        st.session_state.hg_cam_on = False
        st.rerun()

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MANUMOTION",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "first_load" not in st.session_state:
    st.session_state.first_load = True

# ─────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, html,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.stApp {
    background: #07071a !important;
    font-family: 'Outfit', sans-serif !important;
    overflow-x: hidden !important;
    color: #e8e8f5 !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
.stDeployButton, footer, #MainMenu { display: none !important; }

.block-container, [data-testid="stAppViewBlockContainer"] {
    padding: 32px 48px 48px !important;
    max-width: 100% !important;
    min-height: 100vh !important;
}

/* ── Canvas ── */
#mm-canvas {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none;
}

/* ── Hero ── */
.hero-wrap {
    text-align: center; padding-top: 20px;
    margin: 0 auto 60px auto;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: #818cf8;
    background: rgba(129,140,248,0.08); border: 1px solid rgba(129,140,248,0.2);
    padding: 8px 20px; border-radius: 99px; margin-bottom: 32px;
}
.hero-title {
    font-size: 6.5rem; font-weight: 900; letter-spacing: -5px; line-height: 1;
    background: linear-gradient(130deg, #ffffff 0%, #c4b5fd 35%, #818cf8 60%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 28px;
}
.hero-tagline {
    font-size: 1.1rem; line-height: 1.8; color: #6b6b8a;
    max-width: 520px; margin: 0 auto;
    text-align: center;
}

/* ── Cards ── */
.mm-card {
    position: relative;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 28px;
    padding: 40px 36px 36px;
    overflow: hidden;
    transition: transform 0.35s ease, border-color 0.35s ease;
    height: 100%;
    min-height: 300px;
}
.mm-card:hover { transform: translateY(-6px); }
.mm-card-purple:hover { border-color: rgba(129,140,248,0.35) !important; }
.mm-card-teal:hover   { border-color: rgba(52,211,153,0.35) !important; }

.card-glow-blob {
    position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px; border-radius: 50%;
    opacity: 0.08; filter: blur(50px); pointer-events: none;
}
.card-icon-wrap {
    width: 72px; height: 72px; border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 24px;
}
.card-title { font-size: 1.5rem; font-weight: 800; color: #f0f0fa; margin-bottom: 14px; }
.card-desc { font-size: 0.9rem; color: #6b6b8a; line-height: 1.7; margin-bottom: 24px; }
.card-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.card-pill {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    padding: 5px 14px; border-radius: 99px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09); color: #9090aa;
}

/* ── Stats ── */
.stat-bar-wrap {
    position: relative; z-index: 10;
    display: flex; justify-content: center;
    margin-top: 60px; padding-bottom: 56px;
}
.stat-bar {
    display: flex; width: 100%; max-width: 720px;
    border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; overflow: hidden;
}
.stat-item { flex: 1; text-align: center; padding: 28px 20px; border-right: 1px solid rgba(255,255,255,0.06); }
.stat-item:last-child { border-right: none; }
.stat-n {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #c4b5fd, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.stat-l { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #44445a; margin-top: 5px; }

/* ── Navigation buttons ── */
div[data-testid="stButton"] {
    display: flex !important; justify-content: center !important; width: 100% !important;
}
div[data-testid="stButton"] > button {
    height: 54px !important;
    white-space: nowrap !important;
    background: linear-gradient(130deg, #818cf8, #38bdf8) !important;
    color: #fff !important; font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    border: none !important; border-radius: 16px !important;
    cursor: pointer !important;
    margin: 0 auto !important;
    box-shadow: 0 8px 30px rgba(129,140,248,0.28) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 40px rgba(129,140,248,0.45) !important;
    filter: brightness(1.08) !important;
}
}
</style>
"""

LIGHT_MODE_CSS = """
<style>
body, html, [data-testid="stApp"], .stApp, [data-testid="stAppViewContainer"] {
    background: #f1f5f9 !important; color: #1e293b !important;
}
.hero-tagline { color: #64748b !important; }
.hero-title { background: linear-gradient(130deg, #334155 0%, #818cf8 40%, #0ea5e9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.mm-card { 
    background: rgba(255,255,255,1) !important; 
    border: 1px solid rgba(0,0,0,0.04) !important; 
    box-shadow: 0 16px 40px -12px rgba(0,0,0,0.12), 0 4px 10px rgba(0,0,0,0.04) !important; 
}
.card-title { color: #0f172a !important; }
.card-desc { color: #64748b !important; }
.stat-box { background: rgba(0,0,0,0.03) !important; box-shadow: inset 0 2px 8px rgba(0,0,0,0.015) !important; }
.stat-v { color: #0f172a !important; }
.stat-l { color: #64748b !important; }
</style>
"""

st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
if st.session_state.theme == "light":
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# ── Landing Page Animation (Loader) ──
if st.session_state.first_load:
    loader_bg = "#f1f5f9" if st.session_state.theme == "light" else "#07071a"
    st.markdown(f"""
<style>
.mm-loader {{
    position: fixed; inset: 0; z-index: 999999;
    background: {loader_bg}; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    animation: mmHideLoader 0.6s cubic-bezier(0.8, 0, 0.2, 1) 2.2s forwards;
}}
.mm-loader-content {{
    display: flex; flex-direction: column; align-items: center; gap: 24px;
    animation: mmPulse 1.5s ease-in-out infinite alternate;
}}
.mm-spinner {{
    width: 60px; height: 60px;
    border: 4px solid rgba(129,140,248,0.2);
    border-top-color: #818cf8; border-bottom-color: #38bdf8;
    border-radius: 50%;
    animation: mmSpin 1.2s cubic-bezier(0.6, 0.2, 0.4, 0.8) infinite;
}}
.mm-loader-text {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem; font-weight: 800; letter-spacing: 4px;
    background: linear-gradient(130deg, #c4b5fd, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
@keyframes mmSpin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
@keyframes mmPulse {{ 0% {{ opacity: 0.7; transform: scale(0.98); }} 100% {{ opacity: 1; transform: scale(1.02); }} }}
@keyframes mmHideLoader {{ 0% {{ opacity: 1; visibility: visible; transform: scale(1); }} 100% {{ opacity: 0; visibility: hidden; transform: scale(1.05); }} }}
</style>
<div class="mm-loader">
    <div class="mm-loader-content">
        <div class="mm-spinner"></div>
        <div class="mm-loader-text">MANUMOTION</div>
    </div>
</div>
""", unsafe_allow_html=True)
    st.session_state.first_load = False

# ── Canvas animation script ──
components.html("""
<script>
(function() {
  var pDoc = window.parent.document;
  ["mm-canvas", "hg-canvas", "isl-canvas"].forEach(function(id) {
    var elem = pDoc.getElementById(id);
    if (elem) elem.remove();
  });

  var cvs = pDoc.createElement('canvas');
  cvs.id = 'mm-canvas';
  pDoc.body.prepend(cvs);

  var ctx=cvs.getContext('2d'),W,H,nodes=[],raf;
  var HAND=[[.50,.72],[.44,.58],[.43,.44],[.43,.32],[.43,.21],
    [.50,.42],[.50,.29],[.50,.18],[.50,.09],
    [.56,.43],[.57,.30],[.57,.19],[.57,.10],
    [.62,.46],[.63,.35],[.63,.25],[.63,.16],
    [.67,.56],[.71,.47],[.73,.39],[.73,.32]];
  var COLS=['#818cf8','#38bdf8','#34d399','#c4b5fd','#f472b6'];
  function resize(){W=cvs.width=window.parent.innerWidth;H=cvs.height=window.parent.innerHeight;}
  function init(){
    nodes=[];
    HAND.forEach(function(p){nodes.push({x:p[0]*W,y:p[1]*H,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,ax:p[0],ay:p[1],r:3+Math.random()*1.5,col:COLS[Math.floor(Math.random()*COLS.length)],anchor:true});});
    for(var i=0;i<65;i++) nodes.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,ax:null,ay:null,r:1.2+Math.random()*2.2,col:COLS[Math.floor(Math.random()*COLS.length)],anchor:false});
  }
  function tick(){
    ctx.clearRect(0,0,W,H);
    nodes.forEach(function(n){if(n.anchor){n.x+=(n.ax*W-n.x)*.007+n.vx;n.y+=(n.ay*H-n.y)*.007+n.vy;}else{n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>W)n.vx*=-1;if(n.y<0||n.y>H)n.vy*=-1;}});
    for(var i=0;i<nodes.length;i++) for(var j=i+1;j<nodes.length;j++){var dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<170){ctx.beginPath();ctx.moveTo(nodes[i].x,nodes[i].y);ctx.lineTo(nodes[j].x,nodes[j].y);ctx.strokeStyle='rgba(129,140,248,'+((1-d/170)*.28)+')';ctx.lineWidth=.8;ctx.stroke();}}
    nodes.forEach(function(n){ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fillStyle=n.anchor?(n.col+'cc'):(n.col+'66');ctx.fill();if(n.anchor){ctx.beginPath();ctx.arc(n.x,n.y,n.r+5,0,Math.PI*2);ctx.strokeStyle=n.col+'33';ctx.lineWidth=1;ctx.stroke();}});
    if (window.parent.appRaf) { window.parent.cancelAnimationFrame(window.parent.appRaf); }
    raf=window.parent.requestAnimationFrame(tick);
    window.parent.appRaf = raf;
  }
  resize();init();tick();

  if (window.parent.appResizeHandler) {
      window.parent.removeEventListener('resize', window.parent.appResizeHandler);
  }
  window.parent.appResizeHandler = function() {
      if (window.parent.appRaf) window.parent.cancelAnimationFrame(window.parent.appRaf);
      resize();init();tick();
  };
  window.parent.addEventListener('resize', window.parent.appResizeHandler);
})();
</script>
""", height=0, width=0)

# ── Hero ──
# Top level header for Theme toggle
top_col, theme_col = st.columns([12, 1])
with theme_col:
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if st.button("☀️" if st.session_state.theme == "dark" else "🌙", key="theme_btn", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

with top_col:
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-badge">&#10022; AI-Powered Gesture Recognition</div>
      <div class="hero-title">MANUMOTION</div>
      <p class="hero-tagline">Control your world with the wave of a hand.<br>Real-time gesture intelligence, powered by computer vision &amp; ML.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Cards — one per column ──
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
<div class="mm-card mm-card-purple">
  <div class="card-glow-blob" style="background:#818cf8;"></div>
  <div class="card-icon-wrap" style="background:rgba(129,140,248,0.12);border:1px solid rgba(129,140,248,0.2);">&#9994;</div>
  <div class="card-title">Hand Gesture Control</div>
  <p class="card-desc">Control mouse, volume, brightness and media playback entirely through hand gestures — no touch needed.</p>
  <div class="card-pills">
    <span class="card-pill">MediaPipe</span>
    <span class="card-pill">Real-time</span>
    <span class="card-pill">OS Control</span>
  </div>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="mm-card mm-card-teal">
  <div class="card-glow-blob" style="background:#34d399;"></div>
  <div class="card-icon-wrap" style="background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.2);">&#129310;</div>
  <div class="card-title">ISL Recognition</div>
  <p class="card-desc">Recognize Indian Sign Language alphabets in real-time with an ensemble of ML models and audio pronunciation.</p>
  <div class="card-pills">
    <span class="card-pill">Ensemble ML</span>
    <span class="card-pill">26 Alphabets</span>
    <span class="card-pill">Audio Feedback</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Buttons (same columns, directly under cards) ──
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
btn1, btn2 = st.columns([1, 1], gap="large")
with btn1:
    if st.button("🚀  Launch Hand Gesture Control", key="go_hg"):
        st.switch_page("pages/1_Hand_Gesture.py")
with btn2:
    if st.button("🤟  Launch ISL Translator", key="go_isl"):
        st.switch_page("pages/2_ISL_Recognition.py")

# ── Stats bar ──
st.markdown("""
<div class="stat-bar-wrap">
  <div class="stat-bar">
    <div class="stat-item"><div class="stat-n">26</div><div class="stat-l">ISL Alphabets</div></div>
    <div class="stat-item"><div class="stat-n">5</div><div class="stat-l">ML Models</div></div>
    <div class="stat-item"><div class="stat-n">Live</div><div class="stat-l">Inference</div></div>
    <div class="stat-item"><div class="stat-n">95%+</div><div class="stat-l">Accuracy</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #6b6b8a; font-size: 0.75rem; padding-top: 40px; font-weight: 500; letter-spacing: 1px;">
  &copy; 2026 MANUMOTION. All rights reserved.
</div>
""", unsafe_allow_html=True)
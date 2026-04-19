import streamlit as st
import streamlit.components.v1 as components
import os
import sys

# --- 1. CLOUD MOCKS ---
from unittest.mock import MagicMock
try:
    import pyautogui
except ImportError:
    mock_py = MagicMock()
    mock_py.size.return_value = (1920, 1080)
    sys.modules['pyautogui'] = mock_py

# --- 2. CONFIG & PATHS ---
st.set_page_config(page_title="Hand Gesture Control · MANUMOTION", page_icon="✋", layout="wide", initial_sidebar_state="collapsed")

# RESTORE YOUR FULL ORIGINAL CSS
DARK_MODE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
body, html, [data-testid="stApp"], .stApp { background: #07071a !important; font-family: 'Outfit', sans-serif !important; color: #e8e8f5 !important; }
.vid-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; overflow: hidden; height: 480px; position: relative; }
.sb-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }
.sb-val { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg,#818cf8,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#webcam-video { width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); }
</style>
"""
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

# --- 3. UI LAYOUT ---
st.markdown('<div class="hg-topnav"><span style="font-weight:800; color:#c4b5fd;">MANUMOTION</span> / <span style="color:#5b5b7a">Hand Gesture Control</span></div>', unsafe_allow_html=True)
if st.button("← Home"): st.switch_page("app.py")

col_vid, col_side = st.columns([3, 1], gap="medium")

with col_vid:
    st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
    # DIRECT BROWSER WEBCAM COMPONENT (No STUN Error possible)
    components.html("""
        <video id="webcam-video" autoplay playsinline></video>
        <script>
            const video = document.getElementById('webcam-video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { console.error("Error accessing webcam: ", err); });
        </script>
    """, height=480)
    st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown(f"<div class='sb-card'><div style='color:#44445a; font-size:0.6rem;'>ACTIVE GESTURE</div><div class='sb-val'>✋ Tracking</div></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="sb-card">
          <div style="color:#44445a; font-size:0.6rem; font-weight:700;">GESTURE GUIDE</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px;">
            <div>☝️ Move</div><div>👋 Play</div><div>🖐️ Pause</div><div>🤘 Click</div>
          </div>
        </div>
    """, unsafe_allow_html=True)

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.set_page_config(page_title="ISL Recognition", layout="wide")

# RESTORE YOUR ISL CSS
st.markdown("""
<style>
body { background: #07071a !important; color: #e8e8f5 !important; }
.pred-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:28px 20px; text-align:center; }
.pred-letter { font-size:7rem; font-weight:900; background:linear-gradient(135deg,#34d399,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.alph-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
.alph-cell { border-radius:10px; padding:8px 4px; text-align:center; background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.05); color:#3a3a5a; }
</style>
""", unsafe_allow_html=True)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    # Recognition logic updates session state...
    st.session_state["isl_ltr"] = "A" 
    st.session_state["isl_conf"] = 98
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🤟 ISL Recognition")

col_vid, col_pred, col_alpha = st.columns([5, 3, 2])

with col_vid:
    webrtc_streamer(
        key="isl-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun.services.mozilla.com"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col_pred:
    ltr = st.session_state.get("isl_ltr", "—")
    st.markdown(f'<div class="pred-card"><div class="pred-letter">{ltr}</div></div>', unsafe_allow_html=True)

with col_alpha:
    cells = "".join([f"<div class='alph-cell'>{c}</div>" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    st.markdown(f'<div class="alph-grid">{cells}</div>', unsafe_allow_html=True)

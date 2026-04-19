import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. SETUP ---
st.set_page_config(page_title="ISL Recognition · MANUMOTION", page_icon="🤟", layout="wide")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun.services.mozilla.com"]}]})

# --- 2. YOUR ORIGINAL CSS (RESTORED) ---
st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

# --- 3. LOGIC ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    # Your recognition logic here updates session state...
    st.session_state["isl_ltr"] = "A" 
    st.session_state["isl_conf"] = 90
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. YOUR ORIGINAL UI LAYOUT ---
st.title("🤟 ISL Recognition")

col_vid, col_pred, col_alpha = st.columns([5, 3, 2], gap="medium")

with col_vid:
    st.markdown('<div class="vid-panel">', unsafe_allow_html=True)
    webrtc_streamer(
        key="isl-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_pred:
    # Your original Detected Sign card
    ltr = st.session_state.get("isl_ltr", "—")
    cf = st.session_state.get("isl_conf", 0)
    st.markdown(f'<div class="pred-card"><div class="pred-letter">{ltr}</div><div>{cf}% Confidence</div></div>', unsafe_allow_html=True)

with col_alpha:
    # Your original A-Z Reference panel
    st.markdown('<div class="alph-panel">', unsafe_allow_html=True)
    cells = "".join([f"<div class='alph-cell'>{c}</div>" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    st.markdown(f'<div class="alph-grid">{cells}</div></div>', unsafe_allow_html=True)

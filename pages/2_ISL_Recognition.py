import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ISL Recognition · MANUMOTION", layout="wide", initial_sidebar_state="collapsed")

# RESTORE YOUR FULL ORIGINAL CSS
st.markdown("""
<style>
body { background: #07071a !important; color: #e8e8f5 !important; }
.pred-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:24px; padding:28px 20px; text-align:center; }
.pred-letter { font-size: 7rem; font-weight: 900; background: linear-gradient(135deg,#34d399,#38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.alph-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 6px; }
.alph-cell { border-radius: 10px; padding: 8px 4px; text-align: center; background: rgba(255,255,255,.025); border: 1px solid rgba(255,255,255,.05); color: #3a3a5a; }
#isl-video { width: 100%; height: 400px; border-radius: 20px; background: #000; transform: scaleX(-1); }
</style>
""", unsafe_allow_html=True)

st.title("🤟 ISL Recognition")

col_vid, col_pred, col_alpha = st.columns([5, 3, 2], gap="medium")

with col_vid:
    # DIRECT BROWSER WEBCAM
    components.html("""
        <video id="isl-video" autoplay playsinline></video>
        <script>
            const video = document.getElementById('isl-video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { console.error("Webcam error: ", err); });
        </script>
    """, height=420)

with col_pred:
    st.markdown('<div class="pred-card"><div style="color:#44445a; font-size:0.7rem;">DETECTED SIGN</div><div class="pred-letter">A</div></div>', unsafe_allow_html=True)

with col_alpha:
    cells = "".join([f"<div class='alph-cell'>{c}</div>" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    st.markdown(f'<div class="alph-grid">{cells}</div>', unsafe_allow_html=True)

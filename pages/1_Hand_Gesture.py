with col_vid:
    st.markdown('<div class="vid-panel">...</div>', unsafe_allow_html=True)
    
    # DELETE THIS: frame_ph = st.empty()
    # DELETE THE ENTIRE: while True: loop and cap.release()
    
    # PASTE THE CODE HERE:
    webrtc_streamer(
        key="manumotion-stream-isl",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        },
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},
                "height": {"ideal": 360},
                "frameRate": {"ideal": 15} 
            },
            "audio": False,
        },
        async_processing=True,
    )

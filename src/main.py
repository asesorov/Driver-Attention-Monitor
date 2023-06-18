import sys
from pathlib import Path

import av
import threading
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

from utils.audio import AudioFrameHandler
from predict.detection import VideoFrameHandler

# to be able to access packages from any script workdir
sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.detection_conf import SystemConfig


MAIN_CONFIG = SystemConfig()


st.set_page_config(
    page_title="DAMS",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
    menu_items={
        "About": "a.sesorov",
    },
)

col1, col2 = st.columns(spec=[6, 2], gap="medium")

with col1:
    st.title("Driver Attention Monitoring System")
    with st.container():
        wait_slider, ear_slider, head_slider = st.columns(spec=[1, 1, 1])
        with wait_slider:
            # The amount of time (in seconds) to wait before sounding the alarm.
            WAIT_TIME = st.slider("Seconds to wait before sounding alarm:",
                                  *MAIN_CONFIG.properties['THRESHOLDS']['WAIT_TIME'])
        with ear_slider:
            # Lowest valid value of Eye Aspect Ratio. Ideal values [0.15, 0.2].
            EAR_THRESH = st.slider("Eye Aspect Ratio threshold:",
                                   *MAIN_CONFIG.properties['THRESHOLDS']['EAR_THRESH'])
        with head_slider:
            # Highest valid value of head position threshold. Ideal value 15 +-3
            HEAD_THRESH = st.slider("Head position threshold:",
                                    *MAIN_CONFIG.properties['THRESHOLDS']['HEAD_THRESH'])

thresholds = {
    "EAR_THRESH": EAR_THRESH,
    "HEAD_THRESH": HEAD_THRESH,
    "WAIT_TIME": WAIT_TIME,
}

# For streamlit-webrtc
root_path = Path(Path(__file__).resolve().parent.parent)
video_handler = VideoFrameHandler(sys_config=MAIN_CONFIG)
audio_handler = AudioFrameHandler(sound_file_path=root_path / MAIN_CONFIG.properties['RESOURCES']['AUDIO_ALARM'])

lock = threading.Lock()  # For thread-safe access & to prevent race-condition.
shared_state = {"play_alarm": False}


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="bgr24")  # Decode and convert frame to RGB

    frame, play_alarm = video_handler.process(frame, thresholds)  # Process frame
    with lock:
        shared_state["play_alarm"] = play_alarm  # Update shared state

    return av.VideoFrame.from_ndarray(frame, format="bgr24")  # Encode and return BGR frame


def audio_frame_callback(frame: av.AudioFrame):
    with lock:  # access the current “play_alarm” state
        play_alarm = shared_state["play_alarm"]

    new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_alarm)
    return new_frame


# conda activate MFDP && streamlit run src/main.py

with col1:
    ctx = webrtc_streamer(
        key=MAIN_CONFIG.properties['WEBRTC']['KEY'],
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # cloud deployment config
        media_stream_constraints={"video": {"height": {"ideal": MAIN_CONFIG.properties['WEBRTC']['VIDEO_HEIGHT']}},
                                  "audio": True},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
    )

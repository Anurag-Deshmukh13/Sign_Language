
import sys
import pickle
from pathlib import Path
from collections import deque
from datetime import datetime
from threading import Lock

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from streamlit_webrtc import webrtc_streamer, WebRtcMode

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import SignLanguageBiLSTM
from preprocessing import preprocess_hand_body_live

CHECKPOINT = ROOT / "models" / "best_sign_language_hand_body_bilstm_v2.pth"
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"
POSE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
LABEL_PATH = ROOT / "models" / "label_encoder_video_mode_nearestfill.pkl"
LOG_PATH = ROOT / "logs" / "recognition_log.csv"

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
)
st.title("🤟 Sign Language Recognition")
st.caption("BiLSTM + MediaPipe Hands and Pose | 9 greeting signs")

required_files = [CHECKPOINT, HAND_MODEL, POSE_MODEL, LABEL_PATH]
missing_files = [str(p) for p in required_files if not p.exists()]
if missing_files:
    st.error("Required files are missing:\n" + "\n".join(missing_files))
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    with open(LABEL_PATH, "rb") as f:
        encoder = pickle.load(f)

    net = SignLanguageBiLSTM(
        input_size=225,
        num_classes=len(encoder.classes_),
        dropout=0.4,
    ).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net, list(encoder.classes_)


model, class_names = load_model()

st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.20,
    max_value=0.95,
    value=0.60,
    step=0.05,
)
st.sidebar.write(f"Compute device: `{DEVICE}`")
st.sidebar.write("Keep your hands and upper body visible; hold each sign for a few seconds.")

if "sentence_words" not in st.session_state:
    st.session_state.sentence_words = []
if "recognition_rows" not in st.session_state:
    st.session_state.recognition_rows = []

state_lock = Lock()
frame_buffer = deque(maxlen=60)

prediction_state = {
    "gesture": "Waiting for webcam",
    "confidence": 0.0,
    "status": "waiting",
    "frames_processed": 0,
    "hands_detected": False,
    "pose_detected": False,
    "error": "",
}

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
RunningMode = Vision.RunningMode


@st.cache_resource
def get_hand_landmarker():
    options = Vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return Vision.HandLandmarker.create_from_options(options)


@st.cache_resource
def get_pose_landmarker():
    options = Vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return Vision.PoseLandmarker.create_from_options(options)


hand_landmarker = get_hand_landmarker()
pose_landmarker = get_pose_landmarker()

timestamp_lock = Lock()
timestamp_ms = {"value": 0}


def extract_features(hand_result, pose_result):
    hands = np.zeros((2, 21, 3), dtype=np.float32)
    body = np.zeros((33, 3), dtype=np.float32)

    # Match the training convention: Left -> slot 0, Right -> slot 1.
    for landmarks, handedness in zip(
        hand_result.hand_landmarks,
        hand_result.handedness,
    ):
        if not handedness:
            continue

        label = handedness[0].category_name.lower()
        if label == "left":
            slot = 0
        elif label == "right":
            slot = 1
        else:
            continue

        # Ignore duplicate labels instead of overwriting the first hand.
        if np.any(hands[slot]):
            continue

        hands[slot] = np.asarray(
            [[p.x, p.y, p.z] for p in landmarks],
            dtype=np.float32,
        )

    if pose_result.pose_landmarks:
        body = np.asarray(
            [[p.x, p.y, p.z] for p in pose_result.pose_landmarks[0]],
            dtype=np.float32,
        )

    return np.concatenate(
        [hands.reshape(-1), body.reshape(-1)]
    ).astype(np.float32)


def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        with timestamp_lock:
            timestamp_ms["value"] += 33
            current_timestamp = timestamp_ms["value"]

        hand_result = hand_landmarker.detect_for_video(
            mp_image, current_timestamp
        )
        pose_result = pose_landmarker.detect_for_video(
            mp_image, current_timestamp
        )

        row = extract_features(hand_result, pose_result)
        hands_detected = bool(np.any(row[:126] != 0))
        pose_detected = bool(np.any(row[126:] != 0))

        with state_lock:
            frame_buffer.append(row)
            prediction_state["frames_processed"] += 1
            prediction_state["hands_detected"] = hands_detected
            prediction_state["pose_detected"] = pose_detected
            prediction_state["error"] = ""
            seq = np.asarray(frame_buffer, dtype=np.float32).copy()

        # Wait for a useful sequence and require at least one detected hand.
        if len(seq) >= 10 and np.any(seq[:, :126] != 0):
            X = preprocess_hand_body_live(seq)
            inp = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                probs = F.softmax(model(inp), dim=1)[0]

            idx = int(probs.argmax().item())
            conf = float(probs[idx].item())

            with state_lock:
                prediction_state["confidence"] = conf
                prediction_state["gesture"] = (
                    str(class_names[idx]) if conf >= threshold else "Uncertain"
                )
                prediction_state["status"] = (
                    "recognized" if conf >= threshold else "uncertain"
                )

        else:
            with state_lock:
                prediction_state["gesture"] = (
                    "Show hands and upper body"
                    if not pose_detected
                    else "Detecting sign..."
                )
                prediction_state["confidence"] = 0.0
                prediction_state["status"] = "waiting"

    except Exception as exc:
        with state_lock:
            prediction_state["gesture"] = "Detection error"
            prediction_state["status"] = "error"
            prediction_state["error"] = str(exc)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


st.subheader("Live webcam")
ctx = webrtc_streamer(
    key="sign-language-webcam",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
st.write("WebRTC playing:", ctx.state.playing)

left, right = st.columns(2)
with left:
    st.subheader("Current prediction")
    state_placeholder = st.empty()
with right:
    st.subheader("Sentence")
    sentence_placeholder = st.empty()


@st.fragment(run_every="500ms")
def show_live_prediction():
    with state_lock:
        current = prediction_state.copy()

    with state_placeholder.container():
        st.metric("Gesture", current["gesture"])
        st.write(f"Confidence: {current['confidence']:.1%}")
        st.write(f"Status: {current['status']}")
        st.caption(f"Frames processed: {current['frames_processed']}")
        st.caption(
            "Hands detected: " + ("Yes" if current["hands_detected"] else "No")
        )
        st.caption(
            "Body pose detected: " + ("Yes" if current["pose_detected"] else "No")
        )
        if current["error"]:
            st.error(current["error"])


show_live_prediction()

sentence_placeholder.write(
    " ".join(st.session_state.sentence_words)
    or "_Your sentence will appear here._"
)

st.divider()
st.subheader("Sentence controls")

selected_word = st.selectbox("Add a word", options=class_names)
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Add selected sign"):
        word = selected_word.split(". ", 1)[-1]
        st.session_state.sentence_words.append(word)
        st.rerun()

with c2:
    if st.button("Add current prediction"):
        with state_lock:
            current = prediction_state.copy()

        if current["status"] == "recognized":
            word = current["gesture"].split(". ", 1)[-1]
            st.session_state.sentence_words.append(word)
            st.session_state.recognition_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "gesture": current["gesture"],
                "confidence": current["confidence"],
                "status": current["status"],
            })
            st.rerun()
        else:
            st.warning("No confident gesture to add yet.")

with c3:
    if st.button("Clear sentence"):
        st.session_state.sentence_words = []
        st.rerun()

st.write(
    "**Current sentence:**",
    " ".join(st.session_state.sentence_words) or "(empty)",
)

st.subheader("Recognition log")
if st.session_state.recognition_rows:
    log_df = pd.DataFrame(st.session_state.recognition_rows)
    st.dataframe(log_df, use_container_width=True)

    st.download_button(
        "Download session CSV",
        log_df.to_csv(index=False).encode("utf-8"),
        file_name="sign_language_session.csv",
        mime="text/csv",
    )

    if st.button("Append session to project CSV"):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not LOG_PATH.exists()
        log_df.to_csv(
            LOG_PATH,
            mode="a",
            header=write_header,
            index=False,
        )
        st.success(f"Saved log to {LOG_PATH}")
else:
    st.info("No gestures have been logged in this session yet.")
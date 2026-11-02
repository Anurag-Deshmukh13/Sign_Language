
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
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import SignLanguageBiLSTM
from preprocessing import preprocess_live_sequence

CHECKPOINT = ROOT / "models" / "best_sign_language_bilstm.pth"
LANDMARK_MODEL = ROOT / "models" / "hand_landmarker.task"
LABEL_PATH = ROOT / "models" / "label_encoder_video_mode_nearestfill.pkl"
LOG_PATH = ROOT / "logs" / "recognition_log.csv"

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
)
st.title("🤟 Sign Language Recognition")
st.caption("BiLSTM + MediaPipe Tasks | 9 greeting signs")

if not all(p.exists() for p in [CHECKPOINT, LANDMARK_MODEL, LABEL_PATH]):
    st.error("A required model or label file is missing. Check the project models folder.")
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    with open(LABEL_PATH, "rb") as f:
        encoder = pickle.load(f)

    net = SignLanguageBiLSTM(num_classes=len(encoder.classes_)).to(DEVICE)
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
    value=0.20,
    step=0.05,
)
st.sidebar.write(f"Compute device: `{DEVICE}`")
st.sidebar.write("Hold a sign clearly in view for a few seconds.")

if "sentence_words" not in st.session_state:
    st.session_state.sentence_words = []

if "recognition_rows" not in st.session_state:
    st.session_state.recognition_rows = []

if "last_added" not in st.session_state:
    st.session_state.last_added = None

# Shared resources used by the webcam callback.
state_lock = Lock()
frame_buffer = deque(maxlen=60)

prediction_state = {
    "gesture": "Waiting for webcam",
    "confidence": 0.0,
    "status": "waiting",
    "sequence": [],
    "frames_processed": 0,
    "hands_detected": False,
    "error": "",
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



def extract_landmarks(result):
    hands = np.zeros((2, 21, 3), dtype=np.float32)

    if not result.hand_landmarks:
        return hands.reshape(-1)

    for i, landmarks in enumerate(result.hand_landmarks):
        if i >= len(result.handedness):
            continue

        if not result.handedness[i]:
            continue

        label = result.handedness[i][0].category_name.lower()

        if label == "left":
            slot = 0
        elif label == "right":
            slot = 1
        else:
            continue

        # If MediaPipe returns duplicate handedness labels,
        # preserve the first detected hand in that slot.
        if np.any(hands[slot]):
            continue

        hands[slot] = np.asarray(
            [[p.x, p.y, p.z] for p in landmarks],
            dtype=np.float32,
        )

    return hands.reshape(-1).astype(np.float32)

@st.cache_resource
def get_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(LANDMARK_MODEL)
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


landmarker = get_landmarker()

# Timestamp is kept monotonic for MediaPipe VIDEO mode.
timestamp_lock = Lock()
timestamp_ms = {"value": 0}


def video_frame_callback(frame):
    print("VIDEO CALLBACK RECEIVED A FRAME")
    image = frame.to_ndarray(format="bgr24")
    print("Webcam frame shape:", image.shape)
    print("Webcam frame dtype:", image.dtype)
    print("Webcam pixel range:", image.min(), image.max())

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        with timestamp_lock:
            timestamp_ms["value"] += 33
            current_timestamp = timestamp_ms["value"]

        result = landmarker.detect_for_video(
            mp_image,
            current_timestamp,
        )

        print("Raw detected hands:", len(result.hand_landmarks))
        print(
            "Hand labels:",
            [
                h[0].category_name
                for h in result.handedness
                if h
            ],
        )

        row = extract_landmarks(result)
        hands_detected = bool(np.any(row != 0))

        with state_lock:
            frame_buffer.append(row)
            prediction_state["frames_processed"] += 1
            prediction_state["hands_detected"] = hands_detected
            prediction_state["error"] = ""

            if len(frame_buffer) >= 10:
                seq = np.array(frame_buffer, dtype=np.float32)
                prediction_state["sequence"] = seq.tolist()
            else:
                seq = None

        print(
            "Frames:", prediction_state["frames_processed"],
            "| Hands detected:", hands_detected,
            "| Landmarks:", np.count_nonzero(row)
        )

        if seq is not None:
            if np.any(seq != 0):
                X = preprocess_live_sequence(seq)
                inp = torch.tensor(
                    X,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                with torch.no_grad():
                    probs = F.softmax(model(inp), dim=1)[0]

                idx = int(probs.argmax().item())
                conf = float(probs[idx].item())

                top_indices = torch.topk(probs, k=3).indices.tolist()
                print("Top predictions:")
                for i in top_indices:
                    print(f"  {class_names[i]}: {probs[i].item():.1%}")

                with state_lock:
                    prediction_state["confidence"] = conf
                    prediction_state["gesture"] = (
                        str(class_names[idx])
                        if conf >= threshold
                        else "Uncertain"
                    )
                    prediction_state["status"] = (
                        "recognized"
                        if conf >= threshold
                        else "uncertain"
                    )

            else:
                with state_lock:
                    prediction_state["gesture"] = "No hands detected"
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
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
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
        current = {
            "gesture": prediction_state["gesture"],
            "confidence": prediction_state["confidence"],
            "status": prediction_state["status"],
            "frames_processed": prediction_state["frames_processed"],
            "hands_detected": prediction_state["hands_detected"],
            "error": prediction_state["error"],
        }

    with state_placeholder.container():
        st.metric("Gesture", current["gesture"])
        st.write(f"Confidence: {current['confidence']:.1%}")
        st.write(f"Status: {current['status']}")
        st.caption(f"Frames processed: {current['frames_processed']}")
        st.caption(
            "Hands detected: "
            + ("Yes" if current["hands_detected"] else "No")
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

selected_word = st.selectbox(
    "Add a word",
    options=class_names,
)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Add selected sign"):
        word = selected_word.split(". ", 1)[-1]
        st.session_state.sentence_words.append(word)
        st.rerun()

with c2:
    if st.button("Add current prediction"):
        if current["status"] == "recognized":
            word = current["gesture"].split(". ", 1)[-1]
            st.session_state.sentence_words.append(word)

            st.session_state.recognition_rows.append({
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
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

    st.dataframe(
        log_df,
        use_container_width=True,
    )

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

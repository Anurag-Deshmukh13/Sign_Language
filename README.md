# Sign Language Recognition System

A real-time sign-language greeting recognition application built using MediaPipe, PyTorch, BiLSTM, and Streamlit.

## Overview

This project recognizes nine sign-language greetings from hand movements captured through a webcam. It extracts hand landmarks from video frames, processes them into fixed-length sequences, and uses a trained deep-learning model to predict the corresponding greeting.

## Features

- Real-time webcam-based recognition interface.
- Hand landmark extraction using MediaPipe Tasks.
- Sequential gesture classification using a Bidirectional LSTM (BiLSTM).
- Recognition confidence scores.
- Configurable confidence threshold.
- Sentence-building interface for recognized signs.
- Prediction history and recognition results.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Landmark Detection:** MediaPipe
- **Computer Vision:** OpenCV
- **Web Interface:** Streamlit and Streamlit-WebRTC
- **Data Processing:** NumPy, Pandas, Scikit-learn

## Recognized Greetings

The model supports nine classes:

1. Hello
2. How are you
3. Alright
4. Good Morning
5. Good afternoon
6. Good evening
7. Good night
8. Thank you
9. Pleased

## Project Structure

```text
T240_Sign_Language/
├── app/
│   ├── app.py
│   └── app_backup.py
├── models/
│   ├── best_sign_language_bilstm.pth
│   ├── best_sign_language_hand_body_bilstm_v2.pth
│   ├── hand_landmarker.task
│   ├── pose_landmarker_lite.task
│   └── label_encoder_video_mode_nearestfill.pkl
├── src/
│   ├── model.py
│   ├── prediction.py
│   └── preprocessing.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anurag-Deshmukh13/Sign_Language.git
cd Sign_Language
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app/app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`, and allow webcam access.

## Model Performance

The selected video-mode, nearest-fill dataset pipeline achieved **92.11% validation accuracy**, corresponding to 35 correct predictions out of 38 validation samples.

This result reflects evaluation on the available validation split and does not guarantee the same accuracy under real-world webcam conditions.

## Limitations

- The model recognizes nine greeting classes rather than a complete sign language.
- Recognition may vary with lighting, camera position, hand visibility, and signing style.
- Webcam functionality depends on browser permissions and WebRTC support.
- The validation dataset is relatively small.

## Future Improvements

- Expand the vocabulary to include more signs.
- Increase dataset size and signer diversity.
- Improve real-time prediction stability.
- Test across different environments and camera conditions.
- Deploy the application to a stable public hosting platform.

## Author

**Anurag Deshmukh**

GitHub: [Anurag-Deshmukh13](https://github.com/Anurag-Deshmukh13)
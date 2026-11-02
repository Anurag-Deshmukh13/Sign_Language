import numpy as np
import torch
import torch.nn.functional as F

from preprocessing import preprocess_live_sequence


def predict_gesture(sequence, model, class_names, device, threshold=0.60):
    """
    sequence: array shaped (frames, 126)
    Returns predicted gesture, confidence, and status.
    """
    if len(sequence) < 2:
        return {
            "gesture": "Collecting frames",
            "confidence": 0.0,
            "status": "uncertain",
        }

    X = preprocess_live_sequence(np.asarray(sequence, dtype=np.float32))
    inputs = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(inputs), dim=1)[0]

    index = int(probabilities.argmax().item())
    confidence = float(probabilities[index].item())

    if confidence < threshold:
        gesture = "Uncertain"
        status = "uncertain"
    else:
        gesture = str(class_names[index])
        status = "recognized"

    return {
        "gesture": gesture,
        "confidence": confidence,
        "status": status,
    }

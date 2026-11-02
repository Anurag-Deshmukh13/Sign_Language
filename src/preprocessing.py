
import numpy as np


def preprocess_hand_body_live(raw, target_len=60):
    """Preprocess 225 hand + body landmark features for the V2 model."""
    raw = np.asarray(raw, dtype=np.float32).copy()

    if raw.ndim != 2 or raw.shape[1] != 225 or len(raw) == 0:
        raise ValueError(f"Expected (frames, 225), got {raw.shape}")

    if not np.isfinite(raw).all():
        raise ValueError("Raw landmarks contain NaN or infinite values.")

    # 1. Fill missing hand and body frames independently.
    for start, end, landmark_count in [
        (0, 126, 42),
        (126, 225, 33),
    ]:
        group = raw[:, start:end].reshape(
            len(raw), landmark_count, 3
        ).copy()

        valid = np.any(group != 0, axis=(1, 2))
        valid_idx = np.where(valid)[0]

        if len(valid_idx):
            for idx in np.where(~valid)[0]:
                nearest = valid_idx[
                    np.argmin(np.abs(valid_idx - idx))
                ]
                group[idx] = group[nearest]

        raw[:, start:end] = group.reshape(len(raw), -1)

    # 2. Resample to the model's 60-frame sequence length.
    if len(raw) == 1:
        seq = np.repeat(raw, target_len, axis=0)
    else:
        old_t = np.linspace(0, 1, len(raw))
        new_t = np.linspace(0, 1, target_len)

        seq = np.stack([
            np.interp(new_t, old_t, raw[:, j])
            for j in range(225)
        ], axis=1).astype(np.float32)

    # 3. Normalize each hand relative to its wrist.
    X = seq[None, :, :].copy()
    hands = X[0, :, :126].reshape(
        target_len, 2, 21, 3
    ).copy()

    for h in range(2):
        hand = hands[:, h]
        detected = np.any(hand != 0, axis=(1, 2))

        wrist = hand[:, 0:1, :].copy()
        hand -= wrist

        scale = np.max(
            np.abs(hand), axis=(1, 2), keepdims=True
        )
        scale[scale < 1e-6] = 1.0

        hand /= scale
        hand[~detected] = 0

    X[0, :, :126] = hands.reshape(target_len, 126)

    # 4. Normalize body landmarks relative to shoulder center.
    body = X[0, :, 126:].reshape(
        target_len, 33, 3
    ).copy()
    detected_body = np.any(body != 0, axis=(1, 2))

    shoulder_center = (
        body[:, 11:12, :] + body[:, 12:13, :]
    ) / 2.0

    body -= shoulder_center

    scale = np.max(
        np.abs(body), axis=(1, 2), keepdims=True
    )
    scale[scale < 1e-6] = 1.0

    body /= scale
    body[~detected_body] = 0

    X[0, :, 126:] = body.reshape(target_len, 99)

    return X.astype(np.float32)

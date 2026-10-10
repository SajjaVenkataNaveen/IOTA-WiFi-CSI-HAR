import argparse
import ast
import time
from collections import deque

import numpy as np
import serial

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
MIN_PACKETS = 80

EMPTY_CAL_SECONDS = 30
EMPTY_MOVE_CAL_SECONDS = 30
EMPTY_POST_CAL_SECONDS = 30
OCCUPIED_CAL_SECONDS = 50

TRANSITION_SECONDS = 12

# Require several consecutive windows before changing state.
ENTER_REQUIRED = 4
EXIT_REQUIRED = 5

# Median filter over recent probabilities.
SMOOTH_COUNT = 5


# ============================================================
# CSI PARSING
# ============================================================

def parse_csi_line(line):

    if not line.startswith("CSI_DATA"):
        return None

    # Receiver format:
    #
    # type,seq,mac,rssi,rate,noise_floor,fft_gain,
    # agc_gain,channel,local_timestamp,sig_len,
    # rx_format,len,first_word,data

    parts = line.split(",", 14)

    if len(parts) != 15:
        return None

    try:
        rssi = float(parts[3])
        fft_gain = float(parts[6])
        agc_gain = float(parts[7])

        text = parts[14].strip()

        if (
            len(text) >= 2
            and text[0] == '"'
            and text[-1] == '"'
        ):
            text = text[1:-1]

        raw = ast.literal_eval(text)

        raw = np.asarray(
            raw,
            dtype=np.float32
        ).reshape(-1)

        if (
            len(raw) < 4
            or len(raw) % 2 != 0
        ):
            return None

        i = raw[0::2]
        q = raw[1::2]

        amplitude = np.sqrt(
            i * i + q * q
        )

        if not np.all(
            np.isfinite(amplitude)
        ):
            return None

        return {
            "amp": amplitude.astype(np.float32),
            "rssi": rssi,
            "fft_gain": fft_gain,
            "agc_gain": agc_gain,
        }

    except Exception:
        return None


# ============================================================
# COUNTDOWN
# ============================================================

def countdown(seconds, message):

    print()
    print(message)

    for remaining in range(
        seconds,
        0,
        -1
    ):
        print(
            f"\rStarting in {remaining:2d}s...",
            end="",
            flush=True
        )

        time.sleep(1)

    print()
    print()


# ============================================================
# PACKET COLLECTION
# ============================================================

def collect_packets(
    ser,
    seconds,
    phase_name
):

    ser.reset_input_buffer()

    records = []

    start = time.monotonic()

    print(
        f"Collecting {phase_name} for "
        f"{seconds} seconds..."
    )

    while True:

        now = time.monotonic()

        if (
            now - start
            >= seconds
        ):
            break

        raw_bytes = ser.readline()

        if not raw_bytes:
            continue

        line = raw_bytes.decode(
            "utf-8",
            errors="ignore"
        ).strip()

        parsed = parse_csi_line(
            line
        )

        if parsed is None:
            continue

        records.append(
            {
                "time": now - start,
                "amp": parsed["amp"],
                "rssi": parsed["rssi"],
                "fft_gain": parsed["fft_gain"],
                "agc_gain": parsed["agc_gain"],
            }
        )

        if (
            len(records) % 500 == 0
        ):
            print(
                f"  {len(records)} "
                f"valid packets"
            )

    print(
        f"{phase_name}: "
        f"{len(records)} valid packets"
    )

    return records


# ============================================================
# ACTIVE SUBCARRIERS
# ============================================================

def determine_active_subcarriers(
    empty_records
):

    matrix = np.stack(
        [
            r["amp"]
            for r in empty_records
        ]
    )

    median_amp = np.median(
        matrix,
        axis=0
    )

    nonzero_fraction = np.mean(
        matrix > 0.5,
        axis=0
    )

    active = (
        (median_amp > 1.0)
        & (nonzero_fraction > 0.90)
    )

    count = int(
        np.sum(active)
    )

    print()
    print(
        f"Active CSI carriers: "
        f"{count}/{matrix.shape[1]}"
    )

    if count < 40:
        raise RuntimeError(
            "Too few active CSI carriers."
        )

    return active


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    records,
    active
):

    if len(records) < MIN_PACKETS:
        return None

    amp = np.stack(
        [
            r["amp"]
            for r in records
        ]
    )[:, active]

    rssi = np.asarray(
        [
            r["rssi"]
            for r in records
        ],
        dtype=np.float32
    )

    agc = np.asarray(
        [
            r["agc_gain"]
            for r in records
        ],
        dtype=np.float32
    )

    fft = np.asarray(
        [
            r["fft_gain"]
            for r in records
        ],
        dtype=np.float32
    )

    eps = 1e-3

    # IMPORTANT:
    # Preserve absolute CSI amplitude.
    db = 20.0 * np.log10(
        amp + eps
    )

    # Static channel profile.
    profile = np.median(
        db,
        axis=0
    )

    # Relative channel shape.
    profile_center = np.median(
        profile
    )

    relative_profile = (
        profile
        - profile_center
    )

    # Temporal fluctuation.
    temporal_std = np.std(
        db,
        axis=0
    )

    # Packet-to-packet motion.
    temporal_diff = np.mean(
        np.abs(
            np.diff(
                db,
                axis=0
            )
        ),
        axis=0
    )

    globals_ = np.array(
        [
            np.mean(profile),
            np.median(profile),
            np.std(profile),

            np.mean(temporal_std),
            np.median(temporal_std),
            np.percentile(
                temporal_std,
                90
            ),

            np.mean(temporal_diff),
            np.median(temporal_diff),
            np.percentile(
                temporal_diff,
                90
            ),

            np.mean(rssi),
            np.std(rssi),

            np.mean(agc),
            np.std(agc),

            np.mean(fft),
            np.std(fft),
        ],
        dtype=np.float32
    )

    features = np.concatenate(
        [
            profile,
            relative_profile,
            temporal_std,
            temporal_diff,
            globals_,
        ]
    ).astype(np.float32)

    if not np.all(
        np.isfinite(features)
    ):
        return None

    return features


# ============================================================
# BUILD TIME WINDOWS
# ============================================================

def build_windows(
    records,
    active
):

    if len(records) == 0:
        return np.empty((0, 0))

    final_time = records[-1]["time"]

    features = []

    start = 0.0

    while (
        start + WINDOW_SECONDS
        <= final_time
    ):

        end = (
            start
            + WINDOW_SECONDS
        )

        window = [
            r
            for r in records
            if (
                r["time"] >= start
                and r["time"] < end
            )
        ]

        feature = extract_features(
            window,
            active
        )

        if feature is not None:
            features.append(
                feature
            )

        start += STRIDE_SECONDS

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# TEMPORAL TRAIN/VALIDATION SPLIT
# ============================================================

def temporal_split(X):

    n = len(X)

    cut = int(
        n * 0.65
    )

    # Gap between train and validation
    # to reduce overlap leakage.
    purge = 4

    train_end = max(
        1,
        cut - purge
    )

    val_start = min(
        n - 1,
        cut + purge
    )

    return (
        X[:train_end],
        X[val_start:]
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_empty,
    X_occupied
):

    empty_train, empty_val = (
        temporal_split(
            X_empty
        )
    )

    occupied_train, occupied_val = (
        temporal_split(
            X_occupied
        )
    )

    X_train = np.concatenate(
        [
            empty_train,
            occupied_train
        ]
    )

    y_train = np.concatenate(
        [
            np.zeros(
                len(empty_train),
                dtype=int
            ),
            np.ones(
                len(occupied_train),
                dtype=int
            )
        ]
    )

    X_val = np.concatenate(
        [
            empty_val,
            occupied_val
        ]
    )

    y_val = np.concatenate(
        [
            np.zeros(
                len(empty_val),
                dtype=int
            ),
            np.ones(
                len(occupied_val),
                dtype=int
            )
        ]
    )

    n_components = min(
        15,
        X_train.shape[0] - 2,
        X_train.shape[1]
    )

    base_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=42
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42
                )
            )
        ]
    )

    base_model.fit(
        X_train,
        y_train
    )

    probabilities = (
        base_model
        .predict_proba(
            X_val
        )[:, 1]
    )

    # --------------------------------------------------------
    # Choose probability threshold from validation data.
    # --------------------------------------------------------

    best_threshold = 0.5
    best_balanced = -1

    for threshold in np.linspace(
        0.10,
        0.90,
        161
    ):

        prediction = (
            probabilities
            >= threshold
        ).astype(int)

        empty_mask = (
            y_val == 0
        )

        occupied_mask = (
            y_val == 1
        )

        empty_accuracy = np.mean(
            prediction[
                empty_mask
            ] == 0
        )

        occupied_accuracy = np.mean(
            prediction[
                occupied_mask
            ] == 1
        )

        balanced = (
            empty_accuracy
            + occupied_accuracy
        ) / 2.0

        if balanced > best_balanced:
            best_balanced = balanced
            best_threshold = float(
                threshold
            )

    validation_prediction = (
        probabilities
        >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_val,
        validation_prediction,
        labels=[0, 1]
    )

    print()
    print("=" * 70)
    print("CALIBRATION VALIDATION")
    print("=" * 70)

    print(
        "Threshold:",
        f"{best_threshold:.3f}"
    )

    print(
        "Balanced accuracy:",
        f"{best_balanced * 100:.2f}%"
    )

    print()
    print(
        "Confusion matrix "
        "[EMPTY, OCCUPIED]:"
    )

    print(cm)

    empty_probs = probabilities[
        y_val == 0
    ]

    occupied_probs = probabilities[
        y_val == 1
    ]

    print()
    print(
        "EMPTY P(OCCUPIED): "
        f"median={np.median(empty_probs):.3f} "
        f"max={np.max(empty_probs):.3f}"
    )

    print(
        "OCCUPIED P(OCCUPIED): "
        f"median={np.median(occupied_probs):.3f} "
        f"min={np.min(occupied_probs):.3f}"
    )

    # --------------------------------------------------------
    # IMPORTANT: use the EXACT model that was validated.
    # Do not refit after choosing the probability threshold.
    # --------------------------------------------------------

    return (
        base_model,
        best_threshold
    )



# ============================================================
# LIVE DETECTOR
# ============================================================

def live_detect(
    ser,
    model,
    threshold,
    active
):

    # Our real tests show true occupancy is normally very
    # strong (~0.9-1.0), while small operator disturbances can
    # produce weaker probabilities.
    enter_threshold = max(0.70, threshold)

    exit_threshold = min(
        0.30,
        max(0.10, threshold * 0.50)
    )

    print()
    print("=" * 70)
    print("LIVE EMPTY / OCCUPIED DETECTION")
    print("=" * 70)

    print(
        f"Entry threshold: "
        f"{enter_threshold:.3f}"
    )

    print(
        f"Exit threshold : "
        f"{exit_threshold:.3f}"
    )

    print()
    print(
        "Do NOT move either ESP32."
    )

    print(
        "Keep the operator in the same "
        "position used during EMPTY calibration."
    )

    print()

    ser.reset_input_buffer()

    buffer = deque()

    state = "EMPTY"

    enter_count = 0
    exit_count = 0

    prediction_number = 0

    prob_history = deque(
        maxlen=SMOOTH_COUNT
    )

    last_prediction = (
        time.monotonic()
    )

    try:
        while True:

            raw_bytes = ser.readline()

            if not raw_bytes:
                continue

            line = raw_bytes.decode(
                "utf-8",
                errors="ignore"
            ).strip()

            parsed = parse_csi_line(
                line
            )

            if parsed is None:
                continue

            now = time.monotonic()

            buffer.append(
                {
                    "time": now,
                    "amp": parsed["amp"],
                    "rssi": parsed["rssi"],
                    "fft_gain": parsed["fft_gain"],
                    "agc_gain": parsed["agc_gain"],
                }
            )

            cutoff = (
                now
                - WINDOW_SECONDS
                - 0.5
            )

            while (
                buffer
                and buffer[0]["time"]
                < cutoff
            ):
                buffer.popleft()

            if (
                now - last_prediction
                < STRIDE_SECONDS
            ):
                continue

            window_start = (
                now
                - WINDOW_SECONDS
            )

            current = [
                r
                for r in buffer
                if r["time"]
                >= window_start
            ]

            feature = extract_features(
                current,
                active
            )

            if feature is None:
                continue

            raw_probability = float(
                model.predict_proba(
                    feature.reshape(
                        1,
                        -1
                    )
                )[0, 1]
            )

            prob_history.append(
                raw_probability
            )

            probability = float(
                np.median(
                    list(prob_history)
                )
            )

            if state == "EMPTY":

                if (
                    probability
                    >= enter_threshold
                ):
                    enter_count += 1

                else:
                    enter_count = 0

                if (
                    enter_count
                    >= ENTER_REQUIRED
                ):
                    state = "OCCUPIED"

                    enter_count = 0
                    exit_count = 0

            else:

                if (
                    probability
                    <= exit_threshold
                ):
                    exit_count += 1

                else:
                    exit_count = 0

                if (
                    exit_count
                    >= EXIT_REQUIRED
                ):
                    state = "EMPTY"

                    exit_count = 0
                    enter_count = 0

            prediction_number += 1

            marker = ""

            if (
                state == "EMPTY"
                and enter_count
            ):
                marker = (
                    f" entering?"
                    f" {enter_count}/"
                    f"{ENTER_REQUIRED}"
                )

            elif (
                state == "OCCUPIED"
                and exit_count
            ):
                marker = (
                    f" leaving?"
                    f" {exit_count}/"
                    f"{EXIT_REQUIRED}"
                )

            print(
                f"[{prediction_number:04d}] "
                f"{state:<8} "
                f"P(OCCUPIED)="
                f"{probability:.3f} "
                f"raw={raw_probability:.3f}"
                f"{marker}"
            )

            last_prediction = now

    except KeyboardInterrupt:
        print()
        print("Stopped.")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="COM9"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ESP32-C6 SAME-ROOM PRESENCE DEMO")
    print("=" * 70)

    print()
    print(
        "Goal: detect whether the sensing "
        "area between the two ESP32s is "
        "EMPTY or OCCUPIED."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "During EMPTY calibration, stay in "
        "the same operator position you will "
        "use during the final demo."
    )

    print(
        "Only the sensing area between the "
        "two ESP32s should change."
    )

    print()

    ser = serial.Serial(
        args.port,
        args.baud,
        timeout=1
    )

    try:

        # ----------------------------------------------------
        # EMPTY CALIBRATION
        # ----------------------------------------------------

        countdown(
            TRANSITION_SECONDS,
            (
                "PHASE 1: EMPTY calibration.\n"
                "Make the area between the "
                "ESP32s EMPTY.\n"
                "Return to your normal operator "
                "position near the laptop and "
                "STAY STILL."
            )
        )

        empty_records = collect_packets(
            ser,
            EMPTY_CAL_SECONDS,
            "EMPTY"
        )

        if len(empty_records) < 1000:
            raise RuntimeError(
                "Too few EMPTY packets."
            )

        active = determine_active_subcarriers(
            empty_records
        )

        # ----------------------------------------------------
        # EMPTY WITH NORMAL OPERATOR MOVEMENT
        #
        # Friend remains OUTSIDE sensing region.
        # Operator performs only normal laptop-side movement:
        # typing, mouse use, posture changes, turning slightly.
        # ----------------------------------------------------

        countdown(
            8,
            (
                "PHASE 1B: EMPTY with NORMAL OPERATOR MOVEMENT.\n"
                "Friend: stay STILL at the EMPTY waiting position.\n"
                "You: remain near the laptop, but make normal "
                "small demo movements: type, use mouse, shift "
                "posture, turn slightly.\n"
                "Do NOT walk into the sensing region."
            )
        )

        empty_move_records = collect_packets(
            ser,
            EMPTY_MOVE_CAL_SECONDS,
            "EMPTY-MOVEMENT"
        )

        if len(empty_move_records) < 700:
            raise RuntimeError(
                "Too few EMPTY-MOVEMENT packets."
            )

        # ----------------------------------------------------
        # OCCUPIED CALIBRATION
        # ----------------------------------------------------

        countdown(
            TRANSITION_SECONDS,
            (
                "PHASE 2: OCCUPIED calibration.\n"
                "Enter the sensing area between "
                "the ESP32s.\n"
                "During collection, spend time "
                "still and move to a few positions."
            )
        )

        occupied_records = collect_packets(
            ser,
            OCCUPIED_CAL_SECONDS,
            "OCCUPIED"
        )

        if len(occupied_records) < 1000:
            raise RuntimeError(
                "Too few OCCUPIED packets."
            )

        # ----------------------------------------------------
        # POST-OCCUPANCY EMPTY
        #
        # This teaches the model what the channel looks like
        # after the person leaves the sensing region.
        # ----------------------------------------------------

        countdown(
            TRANSITION_SECONDS,
            (
                "PHASE 3: POST-OCCUPANCY EMPTY.\n"
                "Friend: leave the sensing region and return "
                "to the EXACT EMPTY waiting position.\n"
                "You: return to your normal laptop position.\n"
                "Remain mostly still during this calibration."
            )
        )

        empty_post_records = collect_packets(
            ser,
            EMPTY_POST_CAL_SECONDS,
            "EMPTY-POST"
        )

        if len(empty_post_records) < 700:
            raise RuntimeError(
                "Too few EMPTY-POST packets."
            )

        # ----------------------------------------------------
        # WINDOWS
        # ----------------------------------------------------

        print()
        print("Building calibration windows...")

        X_empty_still = build_windows(
            empty_records,
            active
        )

        X_empty_move = build_windows(
            empty_move_records,
            active
        )

        X_occupied = build_windows(
            occupied_records,
            active
        )

        X_empty_post = build_windows(
            empty_post_records,
            active
        )

        X_empty = np.concatenate(
            [
                X_empty_still,
                X_empty_move,
                X_empty_post
            ],
            axis=0
        )

        print(
            "EMPTY-STILL windows:",
            len(X_empty_still)
        )

        print(
            "EMPTY-MOVEMENT windows:",
            len(X_empty_move)
        )

        print(
            "EMPTY-POST windows:",
            len(X_empty_post)
        )

        print(
            "TOTAL EMPTY windows:",
            len(X_empty)
        )

        print(
            "OCCUPIED windows:",
            len(X_occupied)
        )

        if (
            len(X_empty_still) < 20
            or len(X_empty_move) < 20
            or len(X_empty_post) < 20
            or len(X_occupied) < 20
        ):
            raise RuntimeError(
                "Not enough calibration windows."
            )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model, threshold = train_model(
            X_empty,
            X_occupied
        )

        # ----------------------------------------------------
        # RETURN TO EMPTY CONFIGURATION
        # ----------------------------------------------------

        countdown(
            5,
            (
                "CALIBRATION FINISHED.\n"
                "Keep friend at the EMPTY waiting position.\n"
                "You may use the laptop normally, but do not "
                "walk into the sensing region.\n"
                "LIVE DEMO starts now."
            )
        )

        live_detect(
            ser,
            model,
            threshold,
            active
        )

    finally:

        ser.close()


if __name__ == "__main__":
    main()
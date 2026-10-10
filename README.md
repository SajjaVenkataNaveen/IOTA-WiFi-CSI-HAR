# IOTA — Wi-Fi CSI-Based Human Activity Recognition

A real-time Wi-Fi sensing prototype that uses Channel State Information (CSI) and machine learning to detect human presence without cameras or wearable sensors.

## Overview

IOTA uses two ESP32-C6 boards to transmit and receive Wi-Fi signals. The receiver captures CSI data, which is processed to extract signal features and classify room occupancy.

## Key Features

- CSI packet parsing and signal processing
- Window-based feature extraction
- Machine learning model training using a scikit-learn pipeline
- Empty-room and occupied-room calibration
- Probability-based classification and temporal smoothing
- Real-time occupancy detection through serial data

## Hardware

- ESP32-C6 transmitter
- ESP32-C6 receiver
- Computer running Python

## Project Structure

```text
IOTA-WiFi-CSI-HAR/
├── src/
│   └── csi_har.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Install dependencies

Make sure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 2. Connect the hardware

Connect the ESP32-C6 receiver to your computer and identify its serial port.

### 3. Check available options

Run:

```bash
python src/csi_har.py --help
```

Follow the script's prompts to perform calibration and start real-time detection.

## How It Works

1. The receiver collects Wi-Fi CSI packets.
2. The program parses signal measurements and extracts features over time windows.
3. A machine learning classifier is trained using calibration data.
4. Consecutive predictions are smoothed to reduce unstable state changes.
5. The system reports whether the sensing area is empty or occupied.

## Notes

Detection performance depends on sensor placement, environmental conditions, and calibration quality. This repository contains the Python software; hardware firmware and raw datasets are not included.

## Technologies

Python · NumPy · scikit-learn · PySerial · ESP32-C6

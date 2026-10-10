\# IOTA — Wi-Fi CSI-Based Human Activity Recognition



A Wi-Fi sensing prototype that uses Channel State Information (CSI) and machine learning to detect human presence.



\## Overview



The system processes CSI data received from an ESP32-C6 receiver, extracts signal features, trains a classifier, and performs real-time occupancy detection.



\## Features



\- CSI packet parsing and signal processing

\- Window-based feature extraction

\- Machine learning model training

\- Empty-room and occupied-room calibration

\- Probability-based classification and temporal smoothing

\- Real-time serial data processing



\## Hardware



\- ESP32-C6 transmitter

\- ESP32-C6 receiver

\- Computer running Python



\## Project Structure



```text

IOTA-WiFi-CSI-HAR/

├── README.md

├── requirements.txt

├── .gitignore

└── src/

&#x20;   └── csi\_har.py

```



\## Installation



Install the dependencies:



```bash

python -m pip install -r requirements.txt

```



\## Usage



Connect the ESP32-C6 receiver and identify its serial port. Check the program's supported arguments:



```bash

python src/csi\_har.py --help

```



Follow the program's calibration prompts before starting real-time detection.



\## Notes



Detection performance depends on sensor placement, environmental conditions, and calibration. This repository contains the software implementation; hardware firmware and raw datasets are not included.


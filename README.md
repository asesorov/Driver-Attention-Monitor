# Driver Attention Monitor
## Introduction
This is a repository for MFDP course (Emotion recognition task). The developed pipeline should be able to recognize driver's drowsiness using real-time video of the driver's face and trigger an alert if necessary. Additional functionality may allow to collect driver's state data from the whole workday, marking the the most drowsy or distracted states on timeline.
Application is run using Streamlit library.

## Usage
For docker build&run instruction, please refer to the Wiki: [Run app using Dockerfile](https://github.com/asesorov/Driver-Attention-Monitor/wiki/Docker-run)

## Repo structure
- `assets` directory contains visual/audio resources.
- `configs` directory contains configuration files and necessary validation/processing scripts.
	- `samples` directory contains sample configs with pre-calculated default values.
	- `schemas` directory contains JSONShema config validation files.
	- `detection_conf.py` is a main configuration file singleton handler.
	- `logger_conf.py` is a logging configuration file.
- `models` directory contains ML/DL models resources.
  - `head-pose-estimation-adas-0001` contains OpenVINO IR model for head position determination (used in eyes-on-road module).
  - `mobilenetv3` contains OpenVINO IR model for drowsiness detection.
- `sample_images` is used for debug and contains sample images for models testing.
- `src` is a main project source directory containing scripts for video processing.
  - `predict` directory contains pipeline used for drowsiness prediction.
  - `train` directory contains pipeline used for models training.
  - `utils` directory contains auxiliary scripts.
- `requirements.txt` file contains the list of necessary libraries to build&run the application.
- `Dockerfile` is a file for automatic building&running containerized application.

## How it works
Application implements attention monitor system pipeline using 3 metrics:
- Eye Aspect Ratio (EAR) calculated for blink/closed eyes detection
- Head Position is calculated for distraction detection (e.g. driver is looking at their phone)
- Drowsiness is detected using DL model MobileNet-v3 trained on Drowsiness Prediction Dataset

Based on these metrics and pre-determined thresholds form configuration, application monitors driver's state and triggers an alarm when necessary.
The application's MVP version is run as a web-service based on Streamlit library.
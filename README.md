# Multi-Object Detection and Tracking

## Overview
This project implements a computer vision pipeline for detecting and tracking multiple objects in a video with persistent IDs.

## Approach
- Detection: YOLOv8
- Tracking: ByteTrack

## Pipeline
Video → Object Detection → Tracking → ID Assignment → Output Video

## Features
- Real-time object detection
- Persistent ID tracking
- Handles occlusion and motion blur

## Installation
pip install ultralytics opencv-python numpy

## Usage
python main.py

## Input
Place video file in:
input/video.mp4

## Output
Processed video will be saved in:
output/output.mp4

## Assumptions
- Objects are visible in most frames
- Moderate camera movement

## Limitations
- ID switching may occur in heavy occlusion
- Small objects may not be detected

## Improvements
- DeepSORT integration
- Trajectory visualization
- Speed estimation

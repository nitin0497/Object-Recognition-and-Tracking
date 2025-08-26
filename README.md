# Bubble-Tracking---Computer-Vision
# Real time image processing
# Circle Tracking Consistency using OpenCV

This project implements a robust object tracking method using radius, distance, and LAB color consistency to handle ID switching between frames. The target use case is tracking similar-sized colored circles in a video using computer vision.

## Features
- Radius + Distance + Color similarity-based cost function
- Occlusion handling using radius matching
- Inline visualization and video output (for Colab and local environments)

## Project Structure
- `main.py`: Main tracking pipeline
- `input/`: Sample input video
- `output/`: Output result of the tracking
- `notebook/`: Colab notebook used during development

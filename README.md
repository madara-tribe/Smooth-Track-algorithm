## System Overview
This repository provides a real-time object tracking system based on YOLO object detection, designed for **more smoother and stable tracking** performance than previous core sysytem. 

The core functionality is built around object detection (YOLO) and custom logic for bounding box tracking. 
Unlike conventional approaches that cause bounding boxes (BBoxes) to flicker or disappear when detection is out side of frame, this system maintains **continuous, smooth tracking behavior**.

The system supports both **single-object** and **group-object tracking** modes.

## Features

### 1. Single Tracking Mode
- Focuses on tracking a single target object.
- only one target object is to be prioritized.

### 2. Smooth Bounding Box Adjustment
- Automatically adjusts the bounding box size based on camera perspective and object position.

### 3. Robustness to Partial Occlusion or Frame Loss
- Even when the object temporarily moves outside the image frame, the system **keeps the bounding box active**, avoiding disappearance.

### 4. Group Tracking Mode
- detects and surrounds multiple detected objects with a group bounding box.
- Also useful for customization to track only group bbox.

### 5. Motion-Based Tracking Update
- Tracks object movement based on the **difference between previous and current bounding box positions**.
- Adapts to directional changes (left, right, up, down), and moves to new positions smoothly, reducing detection noise.

### single mode

<img src="https://github.com/user-attachments/assets/0b0213ec-8a7a-416e-92e9-e59c822d5eae" width="500px" height="300px">


### group mode
![Image](https://github.com/user-attachments/assets/6fce4f9d-33be-4d09-a254-0f96c9a575ab)

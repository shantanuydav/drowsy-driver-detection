# drowsy-driver-detection
Real-time drowsy driver detection system to improve road safety using Python, OpenCV, MediaPipe, Pygame, and computer vision project that detects driver drowsiness by monitoring eye closure through a webcam. The system uses facial landmarks to analyze the driver's eyes and triggers an audio alarm when the eyes remain closed for a predefined duration.

## 📌 Features

- Real-time webcam monitoring
- Face and eye landmark detection
- Eye-closure based drowsiness detection
- Configurable eye-closure threshold
- Audio alert using Pygame
- Real-time video processing

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- Pygame
- Numpy
- Time

## ⚙️ How It Works

1. The webcam captures the driver's video.
2. OpenCV processes each video frame.
3. MediaPipe detects facial and eye landmarks.
4. The system monitors the driver's eye state.
5. If the eyes remain closed beyond the defined threshold, the system considers the driver potentially drowsy.
6. Pygame plays an audio alarm to alert the driver.

## 📂 Project Structure

```text
drowsy-driver-detection/
│
├── main.py
├── alarm.wav
├── requirements.txt
├── README.md
└── screenshots/

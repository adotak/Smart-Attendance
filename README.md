# FaceSentinel AI 🚀

A high-performance, real-time Smart Attendance System built with Python, OpenCV, and DeepFace. FaceSentinel AI automatically recognizes faces, tracks attendance, logs it to a CSV file, and provides a sleek live visual overlay—including real-time emotion sensing!

## Features ✨
- **Blazing Fast Face Recognition:** Automatically detects and identifies users from a live webcam feed.
- **Real-Time Visual Status:** 
  - 🟩 **[JUST MARKED!]**: Green bounding box feedback when attendance is successfully logged.
  - 🟧 **[LOGGED]**: Orange bounding box feedback for users who have already checked in today.
- **Emotion Sensing (Built-in Option):** Seamless integration with `deepface` to display the user's dominant emotion (e.g., happy, neutral, sad) directly on the live feed.
- **Optimized Memory Caching:** Avoids repetitive disk I/O by caching today's attendees in memory, preventing lag and stuttering in the video feed.
- **Live FPS Counter:** Monitor system performance in real-time.

## Setup & Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adotak/Smart-Attendance.git
   cd Smart-Attendance
   ```

2. **Install the required dependencies:**
   ```bash
   pip install opencv-python face-recognition dlib numpy cmake
   ```
   *(Optional) For Emotion Sensing, install deepface (requires Python 3.8-3.11 with tensorflow support):*
   ```bash
   pip install deepface tf-keras
   ```

3. **Add Face Data:**
   - Create a folder named `dataset` in the root directory.
   - Add images of the people you want to track inside the `dataset` folder. 
   - Name the image files exactly as you want the person's name to appear (e.g., `John_Doe.jpg`).

## Usage 🚀

Run the main application:
```bash
python FaceSentinel_AI.py
```

- The system will load the faces from the `dataset` directory and start your webcam.
- Attendance logs are automatically saved to `Attendance.csv` with timestamps.
- **Press `Enter`** while focused on the webcam window to safely exit the application.

## Configuration ⚙️

You can easily toggle the emotion sensing feature directly inside `FaceSentinel_AI.py`:
```python
DETECT_EMOTIONS = True # Set to False to instantly disable emotion detection
```
## Real-time Face and Hand Tracking (Python, OpenCV, MediaPipe)

This Python program uses OpenCV, MediaPipe, and a touch of image processing magic to provide real-time face and hand tracking with smooth zoom and cropping. 

### Features

- **Face and Hand Tracking:** Detects and tracks facial landmarks and hand keypoints using MediaPipe's powerful machine learning models.
- **Wireframe Visualization:** Displays a wireframe representation of your face and hands on a black background.
- **Smooth Zoom and Cropping:** Allows you to zoom in and out of the video feed while maintaining the correct aspect ratio. The window automatically adjusts to your desired size.

### Requirements

- **Python 3.x** (tested with Python 3.9)
- **OpenCV (`cv2`)**
- **MediaPipe**
- **NumPy**

You can install the required libraries using pip:
```bash
pip install opencv-python mediapipe numpy 
```

### How to Run

1. **Save the code:** Copy the provided code and save it as a Python file (e.g., `facial_tracking.py`).
2. **Run the script:** Open your terminal or command prompt and navigate to the directory where you saved the file. Run the script using:

   ```bash
   python facial_tracking.py 
   ```
3. **Select Camera:** The program will print a list of available cameras. Update the `selected_camera_index` variable in the code to choose the correct index for your built-in camera.
4. **Zoom and Control:**
   - Use the `z` key to zoom in.
   - Use the `x` key to zoom out.
   - Resize the window using your mouse for the desired aspect ratio and zoom level.
   - Press the `Esc` key to exit.

### Troubleshooting

- **Camera Selection Issues:**  If the program tries to use an external camera instead of your built-in camera, carefully review the camera selection section of the code (look for `selected_camera_index`). You might need to experiment with different camera indices.

### Potential Features for the Future

- **Facial Expression Recognition:**  Extend the code to recognize and display basic facial expressions (e.g., happy, sad, surprised).
- **Gesture Control:** Use hand tracking data to control external applications or games.
- **Augmented Reality (AR) Effects:** Add fun AR overlays based on face and hand positions. 



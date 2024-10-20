import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

try:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )

except TypeError:
    print("Warning: model_complexity not supported, using default models.")
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))

# Capture video input 
# Get a list of available cameras
available_cameras = []
for i in range(10):  # Check a few indices (adjust as needed)
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

# Print the indices of available cameras
print("Available cameras:", available_cameras)

# Select the desired camera index (usually 0 for built-in)
selected_camera_index = 0  # Change this if needed

# Capture video input from the selected camera
cap = cv2.VideoCapture(selected_camera_index) 
cap.set(cv2.CAP_PROP_FPS, 60)

# Initial zoom and window size
zoom_factor = 1.5
window_width = 640
window_height = 480

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.resize(image, (640, 480))

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    face_results = face_mesh.process(image)
    hand_results = hands.process(image)

    black_image = np.zeros_like(image)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec,
            )

            right_eye_center_x = int((face_landmarks.landmark[133].x + face_landmarks.landmark[33].x) / 2 * image.shape[1])
            right_eye_center_y = int((face_landmarks.landmark[133].y + face_landmarks.landmark[33].y) / 2 * image.shape[0])

            left_eye_center_x = int((face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2 * image.shape[1])
            left_eye_center_y = int((face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2 * image.shape[0])

            cv2.circle(black_image, (right_eye_center_x, right_eye_center_y), 4, (255, 0, 255), -1)
            cv2.circle(black_image, (left_eye_center_x, left_eye_center_y), 4, (0, 255, 0), -1)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec, drawing_spec
            )

            finger_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0), (128, 0, 128)]
            for i, color in zip([4, 8, 12, 16, 20], finger_colors):
                x = int(hand_landmarks.landmark[i].x * image.shape[1])
                y = int(hand_landmarks.landmark[i].y * image.shape[0])
                cv2.circle(black_image, (x, y), 8, color, -1)

    # --- Zoom and Display ---
    # Prevent zero scaling factors
    width_scale = max(zoom_factor * window_width / image.shape[1], 0.001)
    height_scale = max(zoom_factor * window_height / image.shape[0], 0.001)

    scaled_image = cv2.resize(black_image, None, fx=width_scale, fy=height_scale)
    cropped_image = scaled_image[0:window_height, 0:window_width]

    cv2.imshow('Wireframe Face & Hands', cropped_image)
    # --- End Zoom and Display ---

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('z'): 
        zoom_factor += 0.1
    elif key == ord('x'): 
        zoom_factor -= 0.1
        zoom_factor = max(zoom_factor, 1.0)  

    elif cv2.getWindowProperty('Wireframe Face & Hands', cv2.WND_PROP_VISIBLE) < 1:
        break
    else:
        window_width = int(cv2.getWindowProperty('Wireframe Face & Hands', cv2.WND_PROP_AUTOSIZE))
        window_height = int(cv2.getWindowProperty('Wireframe Face & Hands', cv2.WND_PROP_AUTOSIZE))

cap.release()
cv2.destroyAllWindows()
import time
import winsound as winsound
import numpy as np
import cv2 as cv
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

# -----------------------------
# MediaPipe Face Landmarker
# -----------------------------

model_path = "face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

# Store the latest MediaPipe result
latest_result = None


# MediaPipe calls this function whenever a result is ready
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


#eyes landmarks
LEFT_EYE = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
RIGHT_EYE = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE

# Landmark points used for EAR
RIGHT_EYE_POINTS = [33, 159, 158, 133, 153, 145]
LEFT_EYE_POINTS = [362, 386, 385, 263, 380, 374]

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=print_result
)


# -----------------------------
# Open webcam
# -----------------------------

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

def calculate_ear(landmarks, eye_points):

    p1 = np.array([
        landmarks[eye_points[0]].x,
        landmarks[eye_points[0]].y
    ])

    p2 = np.array([
        landmarks[eye_points[1]].x,
        landmarks[eye_points[1]].y
    ])

    p3 = np.array([
        landmarks[eye_points[2]].x,
        landmarks[eye_points[2]].y
    ])

    p4 = np.array([
        landmarks[eye_points[3]].x,
        landmarks[eye_points[3]].y
    ])

    p5 = np.array([
        landmarks[eye_points[4]].x,
        landmarks[eye_points[4]].y
    ])

    p6 = np.array([
        landmarks[eye_points[5]].x,
        landmarks[eye_points[5]].y
    ])

    horizontal = np.linalg.norm(p1 - p4)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    ear = (vertical_1 + vertical_2) / (2 * horizontal)

    return ear
# Create the Face Landmarker
with FaceLandmarker.create_from_options(options) as landmarker:

    eyes_closed_start = None
    alarm_on = False
    CLOSED_DURATION_THRESHOLD = 1.5 #seconds
    alarm_playing = False

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # OpenCV uses BGR.
        # Convert it to RGB for MediaPipe.
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the NumPy image into a MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Get timestamp in milliseconds
        timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

        # Send frame to MediaPipe
        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )
                # --------------------------------
        # Draw facial landmarks
        # --------------------------------

        if latest_result is not None:

            for face_landmarks in latest_result.face_landmarks:

                for landmark in face_landmarks:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )
                eye_indices = set()
                for connection in LEFT_EYE:
                    eye_indices.add(connection.start)
                    eye_indices.add(connection.end)

                for connection in RIGHT_EYE:
                    eye_indices.add(connection.start)
                    eye_indices.add(connection.end)
                for index in eye_indices:
                    landmark = face_landmarks[index]

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv.circle(
                        frame,
                        (x, y),
                        2,
                        (0, 0, 255),
                        -1
                    )
                # Draw the 6 EAR points for each eye

                for index in RIGHT_EYE_POINTS:
                    landmark = face_landmarks[index]

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv.circle(
                        frame,
                        (x, y),
                        4,
                        (255, 0, 0),
                        -1
                    )


                for index in LEFT_EYE_POINTS:
                    landmark = face_landmarks[index]

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv.circle(
                        frame,
                        (x, y),
                        4,
                        (255, 0, 0),
                        -1
                    )   

                right_ear = calculate_ear(
                    face_landmarks,
                    RIGHT_EYE_POINTS
                )

                left_ear = calculate_ear(
                    face_landmarks,
                    LEFT_EYE_POINTS
                ) 

                EAR_THRESHOLD = 0.27

                ear = (left_ear + right_ear) / 2

                if ear < EAR_THRESHOLD:
                    eye_state = "CLOSED"
                else:
                    eye_state = "OPEN"


                if eye_state == "CLOSED":

                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()

                    closed_duration = time.time() - eyes_closed_start

                    if closed_duration >= CLOSED_DURATION_THRESHOLD:
                        alarm_on = True

                else:
                    eyes_closed_start = None
                    closed_duration = 0
                    alarm_on = False
                if alarm_on and not alarm_playing:
                    winsound.PlaySound(
                        "alarmSound.wav",
                        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                    )
                    alarm_playing = True

                elif not alarm_on and alarm_playing:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    alarm_playing = False
                cv.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (30, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    f"EYES: {eye_state}",
                    (30, 90),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

        # Display the original frame
        cv.imshow("frame", frame)

        # Press q to quit
        if cv.waitKey(1) == ord("q"):
            break


# -----------------------------
# Clean up
# -----------------------------

cap.release()
cv.destroyAllWindows()
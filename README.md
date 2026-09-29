# Real-Time Drowsiness Detection System

A learning-focused computer vision project that uses a laptop webcam to detect prolonged eye closure and trigger an audio alarm.

## Project Overview

The goal of this project is to build a simple real-time drowsiness detector without external hardware or a separately trained machine-learning model.

The system observes the user's eyes through a webcam, measures how open they are using the **Eye Aspect Ratio (EAR)**, and combines that measurement with time. A short eye closure, such as a normal blink, should not trigger the alarm. If the eyes remain closed continuously for at least **1.5 seconds**, the system starts an alarm and keeps it playing until the eyes open again.

### Final pipeline

```text
Laptop Webcam
      ↓
OpenCV frame capture
      ↓
MediaPipe Face Landmarker
      ↓
Facial landmarks
      ↓
Eye landmarks
      ↓
Eye Aspect Ratio (EAR)
      ↓
OPEN / CLOSED classification
      ↓
Closed continuously ≥ 1.5 seconds?
      ↓
Audio alarm
```

---

## Features

- Real-time webcam processing
- Face landmark detection with MediaPipe
- Eye landmark selection
- Eye Aspect Ratio calculation
- Open/closed eye classification
- Temporal detection to distinguish blinking from prolonged eye closure
- Audio alarm using Windows `winsound`
- Visual display of facial landmarks and EAR value
- Simple architecture suitable for learning and experimentation

---

## Project Scope

This version intentionally focuses on one problem:

> **Detect prolonged eye closure using a laptop webcam.**

It does **not** currently include:

- Head-pose estimation
- Yawning detection
- Phone/distraction detection
- Facial-expression analysis
- External sensors or hardware
- A custom-trained neural network
- A production-grade safety system

The purpose is to understand the computer-vision pipeline step by step before adding more complex signals.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam capture and image display |
| MediaPipe Face Landmarker | Facial landmark detection |
| NumPy | Mathematical calculations |
| `time` | Measuring continuous eye-closure duration |
| `winsound` | Playing the alarm on Windows |
| `face_landmarker.task` | Pre-trained MediaPipe model |

---

## Project Structure

```text
drowsinessAlert/
│
├── main.py
├── face_landmarker.task
├── alarmSound.wav
└── README.md
```

### Files

**`main.py`**

Contains the complete detection pipeline.

**`face_landmarker.task`**

The MediaPipe Face Landmarker model used to obtain facial landmarks.

**`alarmSound.wav`**

The WAV audio file played when prolonged eye closure is detected.

**`README.md`**

This documentation.

---

# How the System Works

## 1. Webcam Capture

OpenCV is used to access the laptop camera:

```python
cap = cv.VideoCapture(0)
```

Frames are continuously captured from the webcam.

OpenCV normally represents images in **BGR** format, while the MediaPipe image is prepared as RGB:

```python
rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
```

The resulting image is converted into a MediaPipe image:

```python
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
)
```

---

## 2. MediaPipe Face Landmarker

MediaPipe Face Landmarker detects facial landmarks from each frame.

The project uses the **LIVE_STREAM** mode:

```python
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=print_result
)
```

Because LIVE_STREAM processing is asynchronous, MediaPipe returns its result through a callback:

```python
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
```

The latest result is then used by the main webcam loop.

---

# 3. Selecting the Eyes

MediaPipe provides predefined eye connections:

```python
LEFT_EYE = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
RIGHT_EYE = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
```

These connections identify which facial landmarks belong to the eye regions.

For EAR, six points are selected for each eye.

### Right eye

```python
RIGHT_EYE_POINTS = [33, 159, 158, 133, 153, 145]
```

### Left eye

```python
LEFT_EYE_POINTS = [362, 386, 385, 263, 380, 374]
```

These six points are arranged so that they represent:

```text
             p2       p3
              ●───────●
              ↕       ↕
              ↕       ↕
              ●───────●
             p6       p5

        p1 ●───────────● p4
```

The points are used to measure:

- `p1 → p4`: horizontal eye width
- `p2 → p6`: one vertical eye measurement
- `p3 → p5`: another vertical eye measurement

The six points are not a special MediaPipe `EAR_POINTS` list. They are selected for the geometry required by the EAR calculation.

---

# 4. Eye Aspect Ratio

The Eye Aspect Ratio is used as a numerical measurement of eye openness.

The formula is:

```text
                 |p2-p6| + |p3-p5|
EAR = -----------------------------------
                    2 × |p1-p4|
```

In Python:

```python
horizontal = np.linalg.norm(p1 - p4)
vertical_1 = np.linalg.norm(p2 - p6)
vertical_2 = np.linalg.norm(p3 - p5)

ear = (vertical_1 + vertical_2) / (2 * horizontal)
```

`np.linalg.norm()` calculates the Euclidean distance between two points.

### Why use a ratio?

The raw eye height changes when the user moves closer to or farther from the camera.

A ratio is more useful because both the numerator and denominator change with scale.

For example:

```text
5 / 20  = 0.25
15 / 60 = 0.25
```

Even though the physical measurements are different, the ratio remains the same.

---

# 5. Combining Both Eyes

The EAR is calculated independently for the left and right eyes:

```python
right_ear = calculate_ear(
    face_landmarks,
    RIGHT_EYE_POINTS
)

left_ear = calculate_ear(
    face_landmarks,
    LEFT_EYE_POINTS
)
```

The final EAR is their average:

```python
ear = (left_ear + right_ear) / 2
```

This gives one value representing the current eye openness.

---

# 6. Detecting OPEN and CLOSED Eyes

A threshold is used to classify the current eye state.

The current experimental threshold is:

```python
EAR_THRESHOLD = 0.27
```

The rule is:

```text
EAR < 0.27  → CLOSED
EAR ≥ 0.27  → OPEN
```

In code:

```python
if ear < EAR_THRESHOLD:
    eye_state = "CLOSED"
else:
    eye_state = "OPEN"
```

### Important

`0.27` is an **empirically chosen threshold for this project**, not a universal value that will work for every person, camera, lighting condition, or setup.

The threshold was selected by observing the EAR values produced by the system during testing.

---

# 7. Why EAR Alone Is Not Enough

A major problem is that:

> A blink and drowsiness can both produce a CLOSED eye state.

The difference is often the **duration**.

For example:

```text
Normal blink:
OPEN → CLOSED → OPEN
        ↑
     very short
```

Possible prolonged closure:

```text
OPEN → CLOSED ───────────────→ OPEN
             ↑
        1.5+ seconds
```

Therefore, the system needs a time component.

---

# 8. Measuring Eye-Closure Duration

The system records the moment when the eyes first become closed:

```python
if eyes_closed_start is None:
    eyes_closed_start = time.time()
```

The current duration is then calculated:

```python
closed_duration = time.time() - eyes_closed_start
```

The current threshold is:

```python
CLOSED_DURATION_THRESHOLD = 1.5
```

Therefore:

```text
Eyes closed < 1.5 seconds
        ↓
No alarm

Eyes closed ≥ 1.5 seconds
        ↓
Alarm
```

When the eyes open, the timer is reset:

```python
eyes_closed_start = None
closed_duration = 0
alarm_on = False
```

---

# 9. Alarm System

The alarm uses Python's built-in Windows `winsound` module.

The project uses:

```python
winsound.PlaySound(
    "alarmSound.wav",
    winsound.SND_FILENAME |
    winsound.SND_ASYNC |
    winsound.SND_LOOP
)
```

The flags mean:

- `SND_FILENAME` — play the WAV file
- `SND_ASYNC` — play without blocking the webcam loop
- `SND_LOOP` — continuously repeat the sound

The alarm should continue while the eyes remain closed and stop when the eyes open.

---

# 10. The Alarm State Problem

One important debugging problem occurred during development.

The first implementation effectively did this:

```python
if alarm_on:
    winsound.PlaySound(...)
```

But the webcam loop runs many times per second.

Therefore, the program kept calling `PlaySound()` repeatedly:

```text
Frame 1 → PlaySound()
Frame 2 → PlaySound()
Frame 3 → PlaySound()
Frame 4 → PlaySound()
...
```

This caused the alarm to behave incorrectly.

The solution was to introduce a second state:

```python
alarm_playing = False
```

The final logic is:

```python
if alarm_on and not alarm_playing:
    winsound.PlaySound(
        "alarmSound.wav",
        winsound.SND_FILENAME |
        winsound.SND_ASYNC |
        winsound.SND_LOOP
    )
    alarm_playing = True

elif not alarm_on and alarm_playing:
    winsound.PlaySound(None, winsound.SND_PURGE)
    alarm_playing = False
```

This separates two concepts:

```text
alarm_on
    ↓
Should the alarm be active?

alarm_playing
    ↓
Is the sound already playing?
```

This was an important lesson about **state management in real-time applications**.

---

# 11. Final Detection Logic

The complete decision process can be summarized as:

```text
                  Webcam frame
                       ↓
               Face detected?
                       ↓
               Facial landmarks
                       ↓
                  Eye points
                       ↓
                     EAR
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
          EAR ≥ 0.27        EAR < 0.27
              ↓                 ↓
             OPEN             CLOSED
                                ↓
                         Start timer
                                ↓
                      Closed ≥ 1.5 sec?
                         ↙           ↘
                       NO             YES
                       ↓               ↓
                  No alarm        Alarm ON
                                       ↓
                                Keep looping sound
                                       ↓
                                  Eyes OPEN
                                       ↓
                                   Alarm OFF
```

---

# 12. Visual Interface

The current interface displays:

- Facial landmarks
- Eye landmarks
- Six EAR points for each eye
- Current EAR value
- Current eye state

Example information displayed:

```text
EAR: 0.31
EYES: OPEN
```

When the eyes close:

```text
EAR: 0.18
EYES: CLOSED
```

The visual output is mainly useful for debugging and understanding what the computer vision system is doing.

---

# 13. Development Problems and Lessons

| Problem | What happened | Lesson |
|---|---|---|
| OpenCV window behavior | The camera window did not initially behave as expected | Real-time applications depend on the event loop and `waitKey()` |
| `ear` not defined | EAR was only created inside a conditional block | Variables used by the interface should have predictable initialization |
| Choosing eye points | It was initially unclear which landmarks should be used | EAR requires a specific geometric arrangement of points |
| EAR threshold | Eye values changed between open and closed states | Thresholds should be tested rather than blindly copied |
| Blink vs. drowsiness | A blink also produces a CLOSED state | A temporal condition is necessary |
| Pygame installation | Pygame installation failed in the Python 3.14 environment | A simpler built-in Windows audio solution was appropriate |
| Alarm restarting | `PlaySound()` was called every frame | Real-time systems need state management |
| Alarm integration | The sound worked independently but not correctly in the interface | Audio should be triggered by state changes rather than every frame |

---

# 14. Testing

The main behavioral tests are:

| Test | Expected behavior |
|---|---|
| Eyes open | No alarm |
| Normal blink | No alarm |
| Eyes closed for less than 1.5 s | No alarm |
| Eyes closed for at least 1.5 s | Alarm starts |
| Eyes remain closed | Alarm continues |
| Eyes open again | Alarm stops |

These tests verify the core behavior of the current prototype.

---

# 15. Current Limitations

This is a learning prototype rather than a certified driver-safety system.

### Threshold sensitivity

The `0.27` EAR threshold may need adjustment for another user or camera.

### Lighting

Poor lighting can affect facial landmark detection.

### Camera position

Extreme camera angles or large head movements can affect the eye measurements.

### Glasses and occlusion

Objects covering parts of the eyes can make landmark detection less reliable.

### Single signal

The current system uses eye closure only. Real drowsiness detection can involve other signals such as:

- blink frequency
- PERCLOS
- yawning
- head pose
- gaze direction

### No large-scale validation

The current system has been tested as a personal prototype and has not been validated on a large dataset.

---

# 16. Possible Future Improvements

Possible future versions could include:

1. Adaptive EAR threshold
2. Blink-rate analysis
3. PERCLOS-based drowsiness estimation
4. Head-pose detection
5. Yawning detection
6. Better handling when no face is detected
7. Improved behavior under different lighting conditions
8. Event logging
9. A graphical user interface
10. Configurable thresholds
11. More systematic testing with different users and conditions

These improvements should be added gradually so that each new signal can be understood and tested independently.

---

# 17. What I Learned

This project was not only about building a drowsiness detector. It was an introduction to building a real-time computer-vision pipeline.

The main concepts learned were:

- Webcam frame capture with OpenCV
- BGR and RGB image formats
- MediaPipe Face Landmarker
- Facial landmark coordinates
- Eye landmark topology
- Selecting geometric features from landmarks
- Euclidean distance
- Eye Aspect Ratio
- Threshold-based classification
- Temporal state tracking
- Asynchronous processing
- Audio integration
- Debugging real-time applications
- State management

One of the most important lessons was that a computer-vision system does not always need a new trained model for every problem. In this project, a pre-trained facial landmark model provides the information needed to build a higher-level measurement such as EAR.

---

# 18. References

### MediaPipe

Google AI Edge — Face Landmarker for Python:

https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

General Face Landmarker documentation:

https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker

MediaPipe Face Landmarker Python source:

https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py

MediaPipe face mesh connections:

https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

### Eye Aspect Ratio

Soukupová, T. & Čech, J. — *Real-Time Eye Blink Detection using Facial Landmarks*.

The paper describes the Eye Aspect Ratio approach for detecting eye blinks from facial landmarks.

---

# 19. Final Architecture

```text
┌─────────────────────┐
│    Laptop Camera    │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│       OpenCV        │
│   Frame Capture     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│      MediaPipe      │
│   Face Landmarker   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   Facial Landmarks  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│    Eye Landmarks    │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│        EAR          │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  OPEN / CLOSED      │
│     Threshold       │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Closure Duration   │
│      ≥ 1.5 sec      │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│    Alarm State      │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   winsound Alarm    │
└─────────────────────┘
```

---

## Conclusion

The final prototype successfully combines webcam-based computer vision, facial landmark detection, geometric analysis, temporal reasoning, and audio feedback into a single real-time system.

The most important part of the project was not simply making the alarm work. It was understanding how each layer contributes to the final behavior:

```text
Pixels
  ↓
Landmarks
  ↓
Geometry
  ↓
EAR
  ↓
Eye State
  ↓
Time
  ↓
Decision
  ↓
Audio Feedback
```

This provides a foundation for exploring more advanced drowsiness-detection techniques in future versions.

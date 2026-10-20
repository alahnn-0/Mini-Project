
import cv2
import mediapipe as mp
import serial
import numpy as np
import time
import os
import urllib.request

# Change COM9 if your ESP32 uses another port
ser = serial.Serial("COM9", 115200, timeout=0.02)
time.sleep(2)
print("ESP32 connected")

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
Options = mp_tasks.vision.HandLandmarkerOptions
RunningMode = mp_tasks.vision.RunningMode

latest_result = None


def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


options = Options(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=on_result
)


class Kalman:
    def __init__(self, q=0.3, r=6.0):
        self.Q = q
        self.R = r
        self.P = 1.0
        self.x = 0.0
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return int(round(self.x))

        self.P += self.Q
        k = self.P / (self.P + self.R)
        self.x += k * (measurement - self.x)
        self.P *= (1 - k)
        return int(round(np.clip(self.x, 0, 180)))


# filters = [Kalman() for _ in range(5)]
filters = [
    Kalman(q=2.0, r=2.0),  # Thumb: faster response
    Kalman(),              # Index
    Kalman(),              # Middle
    Kalman(),              # Ring
    Kalman()               # Little
]


def get_angle(lm, a, b, c):
    v1 = np.array([lm[a].x - lm[b].x,
                   lm[a].y - lm[b].y])
    v2 = np.array([lm[c].x - lm[b].x,
                   lm[c].y - lm[b].y])

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator < 1e-6:
        return 90.0

    cos_a = np.dot(v1, v2) / denominator
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


# Thumb, index, middle, ring, little
FINGER_POINTS = [
    (1, 2, 4),
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20)
]

FINGER_NAMES = [
    "THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"
]

FINGER_COLORS = [
    (255, 200, 0),
    (0, 200, 255),
    (255, 100, 0),
    (255, 0, 255),
    (0, 255, 100)
]

# Adjust after testing your finger mechanism.
SERVO_OPEN = 0
SERVO_CLOSED = 180

last_angles = None
timestamp = 0


cap = None

for camera_index in [0, 1, 2, 3]:
    test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if test_cap.isOpened():
        ret, test_frame = test_cap.read()

        if ret and test_frame is not None:
            cap = test_cap
            print(f"Camera found at index {camera_index}")
            break

    test_cap.release()

if cap is None:
    ser.close()
    raise RuntimeError(
        "No camera found. Check camera permissions and connection."
    )


print("Show your hand to control all five fingers.")
print("Press Q to quit.")

with HandLandmarker.create_from_options(options) as detector:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp += 1
            detector.detect_async(mp_image, timestamp)

            if latest_result and latest_result.hand_landmarks:
                lm = latest_result.hand_landmarks[0]
                h, w = frame.shape[:2]

                # Draw the hand landmarks
                for point in lm:
                    cv2.circle(
                        frame,
                        (int(point.x * w), int(point.y * h)),
                        3, (0, 255, 0), -1
                    )

                angles = []

                for i, points in enumerate(FINGER_POINTS):
                    raw_angle = get_angle(lm, *points)

                    # Map finger bend to servo angle.
                    # Calibrate this range for your hand and mechanism.
                    target = int(np.interp(
                        raw_angle,
                        [60, 160],
                        [SERVO_CLOSED, SERVO_OPEN]
                    ))

                    target = int(np.clip(target, 0, 180))
                    servo_angle = filters[i].update(target)
                    angles.append(servo_angle)

                    cv2.putText(
                        frame,
                        f"{FINGER_NAMES[i]}: {servo_angle}",
                        (10, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        FINGER_COLORS[i],
                        2
                    )

                if (
                    last_angles is None
                    or any(abs(a - b) > 1
                           for a, b in zip(angles, last_angles))
                ):
                    command = ",".join(map(str, angles)) + "\n"
                    ser.write(command.encode("ascii"))
                    last_angles = angles.copy()

            else:
                cv2.putText(
                    frame, "No hand detected",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2
                )

                # Open all fingers when the hand disappears.
                if last_angles != [0, 0, 0, 0, 0]:
                    ser.write(b"0,0,0,0,0\n")
                    last_angles = [0, 0, 0, 0, 0]

            cv2.imshow("Five Finger Robot Hand", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        ser.close()
        cv2.destroyAllWindows()

print("Stopped.")

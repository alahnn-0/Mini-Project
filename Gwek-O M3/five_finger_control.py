"""
GWEK-O  -  host application (Python 3.11)

Webcam -> MediaPipe HandLandmarker -> 6 joint angles -> Kalman filter
       -> ramp limiter + dead band -> USB serial "a0,a1,a2,a3,a4,a5\\n" -> ESP32

Install :  py -3.11 -m pip install opencv-python mediapipe numpy pyserial
Run     :  py -3.11 five_finger_control.py --port COM9
Test    :  py -3.11 five_finger_control.py --no-serial          (camera only, no robot)
Log     :  py -3.11 five_finger_control.py --port COM9 --log angle_log.csv   (for Fig 5.1)
Keys    :  q / Esc = quit,  s = save screenshot (shot_N.png)
Close the Arduino Serial Monitor first - it blocks the COM port.
"""
import argparse, csv, os, time, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------- settings
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"
BAUD = 115200

CHANNELS = ["Index", "Middle", "Ring", "Pinky", "ThumbCurl", "ThumbCMC"]

# (CURLED, STRAIGHT) measured value for each channel.
# Fingers/thumb curl: joint angle in degrees.  Thumb CMC: tip-to-index-MCP distance / hand size.
CAPTURE = [(20, 155), (20, 155), (20, 155), (20, 155), (10, 125), (0.25, 0.95)]

# Servo angle sent when the finger is CURLED and when it is STRAIGHT.
# If a finger moves the wrong way, swap the two numbers for that channel.
SERVO_CURLED   = [180, 180, 180, 180, 180, 180]
SERVO_STRAIGHT = [  0,   0,   0,   0,   0,   0]

KALMAN_Q, KALMAN_R = 0.5, 4.0      # process / measurement noise
RAMP_LIMIT = 25                    # max change per frame (degrees)
DEAD_BAND  = 2                     # ignore changes smaller than this (degrees)

# Hand skeleton for drawing
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
# (MCP/base, PIP/middle, TIP) landmark triplets, angle measured at the middle point
TRIPLETS = [(5,6,8), (9,10,12), (13,14,16), (17,18,20), (2,3,4)]

# ----------------------------------------------------------------- helpers
class Kalman:
    """One-dimensional constant-position Kalman filter (one per servo channel)."""
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        self.q, self.r, self.x, self.p = q, r, None, 1.0
    def update(self, z):
        if self.x is None:
            self.x = z
            return z
        self.p += self.q                 # predict
        k = self.p / (self.p + self.r)   # gain
        self.x += k * (z - self.x)       # update
        self.p *= (1 - k)
        return self.x

def joint_angle(a, b, c):
    """Angle at b (degrees) between vectors b->a and b->c."""
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def scale(value, curled, straight, s_curled, s_straight):
    """Linear map of a measured value to a servo angle, clamped to the capture range."""
    t = (value - curled) / (straight - curled)       # 0 = curled, 1 = straight
    t = min(max(t, 0.0), 1.0)
    return s_curled + t * (s_straight - s_curled), t

def classify(ext, pinch_ratio):
    """ext = 0 (curled) .. 1 (straight) for [index, middle, ring, pinky, thumb]."""
    e = [x > 0.6 for x in ext]
    c = [x < 0.4 for x in ext]
    if pinch_ratio < 0.25 and e[1] and e[2]:         return "Pinch"
    if all(e):                                       return "Open hand"
    if all(c):                                       return "Power grip"
    if e[0] and all(c[1:4]):                         return "Pointing"
    if e[0] and e[1] and c[2] and c[3]:              return "Peace sign"
    return "Transitional"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM9", help="ESP32 serial port, e.g. COM9")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-serial", action="store_true", help="run without the robot")
    ap.add_argument("--log", default=None, help="CSV file for raw/filtered log")
    ap.add_argument("--log-channel", type=int, default=0, help="channel to log (0=Index)")
    args = ap.parse_args()

    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = vision.HandLandmarker.create_from_options(options)

    ser = None
    if not args.no_serial:
        import serial
        ser = serial.Serial(args.port, BAUD, timeout=0)
        time.sleep(2.0)                      # ESP32 resets when the port opens

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    filters = [Kalman() for _ in range(6)]
    last_cmd = [None] * 6
    log_f = log_w = None
    if args.log:
        log_f = open(args.log, "w", newline=""); log_w = csv.writer(log_f)
        log_w.writerow(["t", "raw", "filtered"])
    t0, last_ts, shots, fps_t, fps = time.time(), 0, 0, time.time(), 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                       # mirror view
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int(time.time() * 1000), last_ts + 1); last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        label = "No hand"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            P = np.array([[p.x * w, p.y * h, p.z * w] for p in lm])   # pixel-scale 3-D points
            hand_size = np.linalg.norm(P[0] - P[9]) + 1e-9

            # six measured values: 4 finger angles, thumb curl angle, thumb CMC ratio
            measured = [joint_angle(P[a], P[b], P[c]) for a, b, c in TRIPLETS]
            measured.append(np.linalg.norm(P[4] - P[5]) / hand_size)

            cmd, ext, raw_servo = [], [], []
            for i in range(6):
                lo, hi = CAPTURE[i]
                target, t = scale(measured[i], lo, hi, SERVO_CURLED[i], SERVO_STRAIGHT[i])
                ext.append(t); raw_servo.append(target)
                smooth = filters[i].update(target)                       # Kalman
                if last_cmd[i] is None:
                    out = smooth
                else:
                    out = last_cmd[i] + max(-RAMP_LIMIT, min(RAMP_LIMIT, smooth - last_cmd[i]))  # ramp
                    if abs(out - last_cmd[i]) < DEAD_BAND:               # dead band
                        out = last_cmd[i]
                out = int(round(min(max(out, 0), 180)))
                cmd.append(out); last_cmd[i] = out

            pinch_ratio = np.linalg.norm(P[4] - P[8]) / hand_size
            label = classify([ext[0], ext[1], ext[2], ext[3], ext[4]], pinch_ratio)

            if ser:
                ser.write((",".join(map(str, cmd)) + "\n").encode())     # one packet per frame
            if log_w:
                ch = args.log_channel
                log_w.writerow([round(time.time() - t0, 3), round(raw_servo[ch], 2), cmd[ch]])

            for a, b in CONNECTIONS:
                cv2.line(frame, tuple(P[a][:2].astype(int)), tuple(P[b][:2].astype(int)), (80, 200, 120), 2)
            for p in P:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 120, 255), -1)
            for i, v in enumerate(cmd):                                  # angle bars
                y = 90 + i * 22
                cv2.rectangle(frame, (10, y), (10 + int(v / 180 * 150), y + 14), (0, 180, 255), -1)
                cv2.putText(frame, f"{CHANNELS[i]} {v}", (170, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # (if no hand is found nothing is sent, so the servos hold their last position)

        now = time.time(); fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6); fps_t = now
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS {fps:.0f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("GWEK-O", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            shots += 1; cv2.imwrite(f"shot_{shots}.png", frame); print("saved", f"shot_{shots}.png")

    cap.release(); cv2.destroyAllWindows()
    if ser: ser.close()
    if log_f: log_f.close()

if __name__ == "__main__":
    main()

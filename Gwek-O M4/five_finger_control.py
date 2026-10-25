"""
Camera-Controlled Bionic Hand — Full 5 Fingers + Working Thumb
==============================================================
Stack : Python 3.11 | OpenCV | MediaPipe 0.10.33 (Tasks API) | PySerial | NumPy
Hardware: ESP32 DevKit V1 → PCA9685 → 6 servos (SG90/MG90S)
Channels: 0=Index  1=Middle  2=Ring  3=Pinky  4=Thumb-curl  5=Thumb-CMC

TIGHT ELASTIC PROTECTION (built in):
  - RAMP_SPEED    : servo moves max N degrees per frame (no slamming)
  - DEAD_BAND     : ignore changes smaller than N degrees (no hunting)
  - Kalman R=8.0  : heavier smoothing = less servo chatter against elastic

Run:
    py -3.11 five_finger_control.py

First run auto-downloads hand_landmarker.task (~15 MB).
Change COM_PORT below to match your ESP32 (Device Manager → Ports).
"""

import cv2
import numpy as np
import serial
import time
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ══════════════════════════════════════════════════════════════
#  CONFIG  — edit these to match your build
# ══════════════════════════════════════════════════════════════
COM_PORT   = "COM9"       # ← your ESP32 COM port
BAUD_RATE  = 115200

# Servo travel — adjust if your servos are mounted reversed
SERVO_OPEN    = 10    # degrees when finger fully open
SERVO_CLOSED  = 170   # degrees when finger fully closed (tight elastic: keep ≤170)

# Thumb
THUMB_CURL_OPEN    = 10
THUMB_CURL_CLOSED  = 160
THUMB_CMC_OPEN     = 80   # abducted (thumb away from palm)
THUMB_CMC_CLOSED   = 10   # adducted / opposing

# ── Tight elastic protection ──────────────────────────────────
RAMP_SPEED  = 25    # max degrees per frame the servo is allowed to move
                    # 25 = fast/natural feel  |  lower if elastic strains
DEAD_BAND   = 2     # ignore angle changes smaller than this (degrees)
                    # 2 = responsive  |  raise to 4-5 if servo chatters

# MediaPipe model
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ══════════════════════════════════════════════════════════════
#  KALMAN FILTER  — one per channel, tuned for tight elastic
# ══════════════════════════════════════════════════════════════
class Kalman1D:
    """
    1D scalar Kalman filter.
    Q=0.5  — faster tracking, follows hand movement quickly
    R=4.0  — less smoothing = more responsive, slight jitter ok
    """
    def __init__(self, q: float = 0.5, r: float = 4.0, init: float = 90.0):
        self.Q = q
        self.R = r
        self.P = 1.0
        self.x = float(init)

    def update(self, z: float) -> float:
        self.P += self.Q
        K       = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1.0 - K)
        return float(np.clip(self.x, 0, 180))


# ══════════════════════════════════════════════════════════════
#  RAMP CONTROLLER  — per channel, limits speed + dead band
# ══════════════════════════════════════════════════════════════
class RampController:
    """
    Moves the servo toward target at max RAMP_SPEED deg/frame.
    Ignores changes within DEAD_BAND to prevent elastic hunting.
    """
    def __init__(self, init: float = 90.0):
        self.current = float(init)

    def step(self, target: float) -> int:
        diff = target - self.current
        # Dead band — don't move for tiny fluctuations
        if abs(diff) < DEAD_BAND:
            return int(round(self.current))
        # Ramp — clamp movement to RAMP_SPEED per frame
        move = max(-RAMP_SPEED, min(RAMP_SPEED, diff))
        self.current += move
        self.current  = float(np.clip(self.current, 0, 180))
        return int(round(self.current))


# One Kalman + one Ramp per servo channel (5 total — no CMC)
kalman  = [Kalman1D() for _ in range(5)]
ramp    = [RampController() for _ in range(5)]


# ══════════════════════════════════════════════════════════════
#  ANGLE MATHS
# ══════════════════════════════════════════════════════════════
def _vec(lm, a: int, b: int) -> np.ndarray:
    return np.array([lm[b].x - lm[a].x,
                     lm[b].y - lm[a].y,
                     lm[b].z - lm[a].z], dtype=float)

def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def finger_angle(lm, mcp: int, pip: int, tip: int) -> float:
    """
    Multi-joint weighted average with full Z correction.

    MCP angle — base knuckle lift (weight 0.4)
      vectors originate at MCP: MCP→wrist(0) and MCP→PIP

    PIP angle — mid-segment bend (weight 0.6)
      vectors originate at PIP: PIP→MCP and PIP→TIP

    Blending:  angle = 0.4×MCP + 0.6×PIP
    Z included in all vectors so hand tilt doesn't distort the reading.
    Result: ~170° open → ~10° fully closed.
    """
    # PIP joint angle (mid bend)
    pip_v1 = _vec(lm, pip, mcp)
    pip_v2 = _vec(lm, pip, tip)
    pip_angle = _angle_deg(pip_v1, pip_v2)

    # MCP joint angle (base lift) — uses wrist (landmark 0) as anchor
    mcp_v1 = _vec(lm, mcp, 0)      # MCP → wrist
    mcp_v2 = _vec(lm, mcp, pip)    # MCP → PIP
    mcp_angle = _angle_deg(mcp_v1, mcp_v2)

    # Weighted blend — PIP contributes more to closure feel
    return 0.4 * mcp_angle + 0.6 * pip_angle

def thumb_curl(lm) -> float:
    """
    Thumb multi-joint blend with Z correction.
    MCP angle: landmark 2 using wrist(0) as anchor  (weight 0.4)
    IP angle:  landmark 3 between landmarks 2 and 4  (weight 0.6)
    """
    # IP joint (tip bend)
    ip_v1 = _vec(lm, 3, 2)
    ip_v2 = _vec(lm, 3, 4)
    ip_angle = _angle_deg(ip_v1, ip_v2)

    # MCP joint (base lift)
    mcp_v1 = _vec(lm, 2, 0)    # thumb MCP → wrist
    mcp_v2 = _vec(lm, 2, 3)    # thumb MCP → IP
    mcp_angle = _angle_deg(mcp_v1, mcp_v2)

    return 0.4 * mcp_angle + 0.6 * ip_angle

def thumb_cmc(lm) -> float:
    """
    Thumb opposition: normalised distance between thumb tip (4)
    and index MCP (5).  Large = open/abducted, small = pinching.
    Returns a servo angle directly.
    """
    dx = lm[4].x - lm[5].x
    dy = lm[4].y - lm[5].y
    dz = lm[4].z - lm[5].z     # Z included — fixes when thumb points toward camera
    dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))
    scale = float(np.sqrt((lm[9].x-lm[0].x)**2
                         +(lm[9].y-lm[0].y)**2
                         +(lm[9].z-lm[0].z)**2)) + 1e-8
    norm  = float(np.clip(dist / scale, 0.0, 1.5))
    return float(np.interp(norm, [0.1, 0.9],
                            [THUMB_CMC_CLOSED, THUMB_CMC_OPEN]))


# Finger landmark triples: (MCP, PIP, TIP)
FINGERS = [
    (5,  6,  8 ),   # ch 0 — Index
    (9,  10, 12),   # ch 1 — Middle
    (13, 14, 16),   # ch 2 — Ring
    (17, 18, 20),   # ch 3 — Pinky
]
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky", "Thumb curl"]


def compute_angles(lm) -> list[int]:
    """
    Full pipeline: raw angle → Kalman filter → ramp limiter → int servo angle.
    Returns 6 servo angles [idx, mid, rng, pnk, t_curl, t_cmc].
    """
    raw = []

    # Fingers 0-3: MediaPipe angle ~170=open → ~10=closed
    # Map to SERVO_OPEN…SERVO_CLOSED
    for mcp, pip, tip in FINGERS:
        a = finger_angle(lm, mcp, pip, tip)
        s = float(np.interp(a,
                             [25.0, 145.0],
                             [SERVO_CLOSED, SERVO_OPEN]))
        raw.append(s)

    # Thumb curl (ch 4)
    tc = thumb_curl(lm)
    raw.append(float(np.interp(tc,
                                [25.0, 145.0],
                                [THUMB_CURL_CLOSED, THUMB_CURL_OPEN])))

    # Kalman → ramp  (5 channels only)
    out = []
    for i in range(5):
        smooth  = kalman[i].update(raw[i])
        clamped = ramp[i].step(smooth)
        out.append(clamped)

    return out


# ══════════════════════════════════════════════════════════════
#  GRASP CLASSIFIER  (display only)
# ══════════════════════════════════════════════════════════════
def classify(angles: list[int]) -> str:
    idx, mid, rng, pnk, tc = angles
    curled = [a > 110 for a in [idx, mid, rng, pnk]]
    open_  = [a <  60 for a in [idx, mid, rng, pnk]]

    if tc > 120 and idx > 110:           return "Pinch"
    if all(curled) and tc > 100:         return "Power Grip"
    if not curled[0] and all(curled[1:]): return "Pointing"
    if all(open_):                        return "Open Hand"
    if not curled[0] and not curled[1] and all(curled[2:]): return "Peace"
    return "Transitional"


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
TIP_IDS = {4, 8, 12, 16, 20}

def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 1)
    for i, (x, y) in enumerate(pts):
        col = (0, 100, 255) if i in TIP_IDS else (255, 255, 255)
        cv2.circle(frame, (x, y), 4, col, -1)

def draw_overlay(frame, angles: list[int], grasp: str, fps: float):
    h, w = frame.shape[:2]
    BAR = 150

    # FPS top-left
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Grasp label
    cv2.putText(frame, grasp, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 120), 2)

    # Per-channel bars on right
    for i, (name, ang) in enumerate(zip(FINGER_NAMES, angles)):
        y   = 12 + i * 34
        x0  = w - BAR - 90
        fil = int(ang / 180 * BAR)

        # Background bar
        cv2.rectangle(frame, (x0, y), (x0 + BAR, y + 18), (50, 50, 50), -1)
        # Filled portion — green when open, orange when closed
        color = (0, 200, 80) if ang < 90 else (0, 140, 255)
        cv2.rectangle(frame, (x0, y), (x0 + fil, y + 18), color, -1)
        # Label
        cv2.putText(frame, f"{name[:6]}: {ang:3d}",
                    (x0, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    # Tight elastic warning strip at bottom
    cv2.rectangle(frame, (0, h-22), (w, h), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Ramp:{RAMP_SPEED}  Dead:{DEAD_BAND}  Kalman R:{kalman[0].R}",
                (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)


# ══════════════════════════════════════════════════════════════
#  SERIAL
# ══════════════════════════════════════════════════════════════
def open_serial(port: str):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        print(f"[OK] Serial on {port}")
        return ser
    except Exception as e:
        print(f"[WARN] Serial failed ({e}) — DEMO mode, angles printed only.")
        return None

def send(ser, angles: list[int]):
    line = ",".join(map(str, angles)) + "\n"
    if ser and ser.is_open:
        ser.write(line.encode())
    else:
        print(f"  → {line.strip()}")


# ══════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[DL] Downloading {MODEL_PATH} (~15 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[DL] Done.")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    ensure_model()

    # MediaPipe setup
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    ser = open_serial(COM_PORT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_ts   = 0
    FRAME_MS   = 33
    angles     = [SERVO_OPEN] * 4 + [THUMB_CURL_OPEN]
    prev_time  = time.time()
    fps        = 0.0

    print("Running — press Q to quit, +/- to adjust RAMP_SPEED live.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result  = detector.detect_for_video(mp_img, frame_ts)
        frame_ts += FRAME_MS

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_hand(frame, lm)
            angles = compute_angles(lm)
            send(ser, angles)

        grasp = classify(angles)

        # FPS
        now     = time.time()
        fps     = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        draw_overlay(frame, angles, grasp, fps)
        cv2.imshow("Bionic Hand — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):           # increase ramp speed
            global RAMP_SPEED
            RAMP_SPEED = min(RAMP_SPEED + 1, 20)
            print(f"RAMP_SPEED → {RAMP_SPEED}")
        elif key == ord('-'):           # decrease ramp speed
            RAMP_SPEED = max(RAMP_SPEED - 1, 1)
            print(f"RAMP_SPEED → {RAMP_SPEED}")

    cap.release()
    detector.close()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
import os
import json
import subprocess
import threading
import time
import urllib.request

import cv2
import numpy as np

from mqtt_comm import publish_control

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import tkinter as tk
except Exception:
    tk = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CMD_PATH = os.path.join(BASE_DIR, "head_tracking_cmd.txt")
BOUNDS_PATH = os.path.join(BASE_DIR, "head_tracking_bounds.txt")
STATUS_PATH = os.path.join(BASE_DIR, "head_tracking_status.txt")
LBF_MODEL_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")
LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
NTFY_TOPIC = "upsurt-help-12345"
NTFY_SERVER_BASE = "https://ntfy.sh"
NTFY_PRIORITY = "urgent"

# ----------------------------
# SETTINGS
# ----------------------------
MOUSE_CONTROL_ENABLED = True
SHOW_CAMERA_PREVIEW = False

SMOOTHING_ALPHA = 0.75
MOVE_INTERVAL_SEC = 0.03
MIN_MOVE_PIXELS = 1
RESET_AFTER_LOST_FACE_SEC = 0.6

INVERT_X = False
INVERT_Y = False

HEAD_DIRECTION_CONTROL_ENABLED = True
YAW_DEADZONE_NORM = 0.0015
PITCH_DEADZONE_NORM = 0.0015
ORIENTATION_PIXEL_SCALE_X = 3000
ORIENTATION_PIXEL_SCALE_Y = 3000
MAX_STEP_PIXELS_X = 5
MAX_STEP_PIXELS_Y = 5
# Directional X tuning (after optional invert): helps compensate harder left movement.
X_DEADZONE_NEG_NORM = 0.0012
X_DEADZONE_POS_NORM = 0.0015
X_SCALE_NEG_MULTIPLIER = 1.28
X_SCALE_POS_MULTIPLIER = 1.00
Y_DEADZONE_NEG_NORM = 0.0011
Y_DEADZONE_POS_NORM = 0.0015
Y_SCALE_NEG_MULTIPLIER = 1.30
Y_SCALE_POS_MULTIPLIER = 1.00

# Replaced by dwell/hover click inside the UI (see light_menu_view.py).
BLINK_DOUBLE_CLICK_ENABLED = False
BLINK_RATIO_CLOSED = 0.27
BLINK_RATIO_OPEN = 0.29
BLINK_CLOSED_MIN_SEC = 0.012
DOUBLE_BLINK_WINDOW_SEC = 2.4
CLICK_COOLDOWN_SEC = 0.45
BLINK_ADAPTIVE_ENABLED = True
BLINK_INTER_BLINK_MIN_SEC = 0.00
CAMERA_RETRY_DELAY_SEC = 1.0
BLINK_EVENT_OPEN_CONFIRM_SEC = 0.0
BLINK_REQUIRED_OPEN_GAP_SEC = 0.070
POST_CLICK_BLINK_LOCK_SEC = 0.35
BLINK_SCORE_SMOOTH_ALPHA = 0.60
BLINK_MIN_CLOSED_FRAMES = 1
BLINK_MIN_OPEN_FRAMES = 1
BLINK_MAX_EYE_ASYMMETRY = 0.22

# Dwell/hover click (cursor stays still over a control).
DWELL_CLICK_ENABLED = True
DWELL_CLICK_SEC = 2.0
DWELL_MOVE_TOL_PX = 6
DWELL_COOLDOWN_SEC = 0.80

# Keeps motion continuous for brief tracking dips.
# (Does NOT change stopping method; it only extends direction "inertia".)
MOTION_HOLD_FRAMES = 10
MOTION_HOLD_DECAY_ENABLED = True

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CUSTOM_CURSOR_ENABLED = True
CUSTOM_CURSOR_COLOR = "#00E5FF"
CUSTOM_CURSOR_SIZE = 24
CUSTOM_CURSOR_THICKNESS = 3

# Climate LED2 brightness control (same +/-50 step as http_test.py).
led2_brightness = 255
led1_brightness = 255
led1_blink_interval_sec = 0.8
led1_blink_enabled = False
led1_power_on = True
led1_blink_stop_event = threading.Event()
led1_blink_thread = None

# 68-point facial landmark mapping (dlib/OpenCV LBF)
LEFT_EYE_INDICES = {36, 37, 38, 39, 40, 41}
RIGHT_EYE_INDICES = {42, 43, 44, 45, 46, 47}
NOSE_TIP_INDEX = 30

LEFT_EYE_CONNECTIONS = [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]
RIGHT_EYE_CONNECTIONS = [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]
NOSE_CONNECTIONS = [
    (27, 28),
    (28, 29),
    (29, 30),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
]


os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")


def resolve_haar_cascade_path() -> str | None:
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    candidates.extend(
        [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]
    )

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def ensure_lbf_model() -> bool:
    if os.path.exists(LBF_MODEL_PATH):
        return True
    print(f"Downloading landmark model to {LBF_MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(LBF_MODEL_URL, LBF_MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error downloading LBF model: {e}")
        return False


def draw_connections(frame, landmarks_norm, connections, color=(0, 255, 0), thickness=1) -> None:
    h, w, _ = frame.shape
    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks_norm) or end_idx >= len(landmarks_norm):
            continue
        x1, y1 = landmarks_norm[start_idx]
        x2, y2 = landmarks_norm[end_idx]
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        if not (0 <= px1 < w and 0 <= py1 < h and 0 <= px2 < w and 0 <= py2 < h):
            continue
        cv2.line(frame, (px1, py1), (px2, py2), color, thickness)


def eye_open_ratio(landmarks_norm, eye_indices) -> float:
    xs = []
    ys = []
    for idx in eye_indices:
        if idx >= len(landmarks_norm):
            continue
        x, y = landmarks_norm[idx]
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    if len(xs) < 2 or len(ys) < 2:
        return 1.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    eye_width = max(max_x - min_x, 1e-6)
    eye_height = max_y - min_y
    return eye_height / eye_width


def mean_xy_for_indices(landmarks_norm, indices):
    xs = []
    ys = []
    for idx in indices:
        if idx >= len(landmarks_norm):
            continue
        x, y = landmarks_norm[idx]
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def head_yaw_pitch(landmarks_norm):
    if NOSE_TIP_INDEX >= len(landmarks_norm):
        return None, None
    nose_x, nose_y = landmarks_norm[NOSE_TIP_INDEX]
    if nose_x is None or nose_y is None:
        return None, None

    left_eye = mean_xy_for_indices(landmarks_norm, LEFT_EYE_INDICES)
    right_eye = mean_xy_for_indices(landmarks_norm, RIGHT_EYE_INDICES)
    if left_eye is None or right_eye is None:
        return None, None

    left_x, left_y = left_eye
    right_x, right_y = right_eye
    eye_mid_x = (left_x + right_x) / 2.0
    eye_mid_y = (left_y + right_y) / 2.0

    eye_dx = right_x - left_x
    denom = max(abs(eye_dx), 1e-6)

    yaw = (float(nose_x) - eye_mid_x) / denom
    pitch = (float(nose_y) - eye_mid_y) / denom
    return yaw, pitch


def safe_run(cmd):
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        print(f"Command failed: {cmd} -> {e}")
        return False


def send_led2_brightness(brightness: int) -> bool:
    payload = {"device": "led2", "action": "on", "speed": int(brightness)}
    ok = publish_control(payload)
    if ok:
        print(f"LED2 brightness set to {brightness}")
    else:
        print("LED2 brightness update failed: MQTT publish failed")
    return ok


def _send_esp_payload(payload: dict, timeout_sec: float = 1.0) -> bool:
    _ = timeout_sec
    ok = publish_control(payload)
    if not ok:
        print(f"ESP command failed: {payload} -> MQTT publish failed")
    return ok


def send_help_alert() -> bool:
    """Send emergency help alert to mobile via ntfy push."""
    topic = (NTFY_TOPIC or "").strip().strip("/")
    if not topic:
        return False

    url = f"{NTFY_SERVER_BASE.rstrip('/')}/{topic}"
    payload = {
        "event": "help_request",
        "source": "raspberry_pi",
        "message": "Need assistance now",
        "ts": time.time(),
    }
    headers = {
        "Content-Type": "application/json",
        "Priority": NTFY_PRIORITY,
        "Title": "UPSURT HELP",
        "Tags": "warning,sos,rotating_light",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            _ = resp.read()
        return True
    except Exception as e:
        print(f"Help alert failed: {e}")
        return False


def send_led1_brightness(brightness: int) -> bool:
    payload = {"device": "led1", "action": "on", "speed": int(brightness)}
    ok = _send_esp_payload(payload)
    if ok:
        print(f"LED1 brightness set to {brightness}")
    return ok


def send_led1_on() -> bool:
    return _send_esp_payload({"device": "led1", "action": "on"})


def send_led1_off() -> bool:
    return _send_esp_payload({"device": "led1", "action": "off"})


def _led1_blink_loop() -> None:
    led_on = False
    while not led1_blink_stop_event.is_set():
        if not led1_blink_enabled or not led1_power_on:
            time.sleep(0.05)
            continue
        if led_on:
            send_led1_off()
        else:
            send_led1_brightness(led1_brightness)
        led_on = not led_on
        time.sleep(max(0.1, float(led1_blink_interval_sec)))


def print_help_menu() -> None:
    print("")
    print("=== Head Tracking Help ===")
    print("Core commands:")
    print("  toggle            - enable/disable mouse control")
    print("  calibrate         - set current head pose as neutral")
    print("  exit              - stop tracking loop")
    print("")
    print("External actions:")
    print("  keyboard          - open on-screen keyboard")
    print("  light             - open light menu/action")
    print("  light_on/off      - light power control")
    print("  climate           - open climate menu/action")
    print("  climate_temp_up   - LED2 +50 brightness step")
    print("  climate_temp_down - LED2 -50 brightness step")
    print("  tv_volume_up/down - LED1 brightness +/-50")
    print("  tv_channel_up/down- LED1 blink speed +/-0.1s")
    print("")
    print("Blink click settings:")
    print(f"  enabled={BLINK_DOUBLE_CLICK_ENABLED} window={DOUBLE_BLINK_WINDOW_SEC:.2f}s cooldown={CLICK_COOLDOWN_SEC:.2f}s")
    print(f"  closed/open ratios={BLINK_RATIO_CLOSED:.3f}/{BLINK_RATIO_OPEN:.3f} adaptive={BLINK_ADAPTIVE_ENABLED}")
    print("")
    print("Direction tuning:")
    print(f"  X deadzone(-/+)= {X_DEADZONE_NEG_NORM:.4f}/{X_DEADZONE_POS_NORM:.4f} scale(-/+)= {X_SCALE_NEG_MULTIPLIER:.2f}/{X_SCALE_POS_MULTIPLIER:.2f}")
    print(f"  Y deadzone(-/+)= {Y_DEADZONE_NEG_NORM:.4f}/{Y_DEADZONE_POS_NORM:.4f} scale(-/+)= {Y_SCALE_NEG_MULTIPLIER:.2f}/{Y_SCALE_POS_MULTIPLIER:.2f}")
    print("==========================")
    print("")


def handle_external_action(cmd: str):
    global led2_brightness, led1_brightness, led1_blink_interval_sec, led1_blink_enabled, led1_power_on
    if cmd == "keyboard":
        if safe_run(["wvkbd"]):
            print("Opened keyboard with wvkbd.")
            return
        if safe_run(["matchbox-keyboard"]):
            print("Opened keyboard with matchbox-keyboard.")
            return
        if safe_run(["onboard"]):
            print("Opened keyboard with onboard.")
            return
        print("No on-screen keyboard found.")
        return

    if cmd == "light":
        print("LIGHT command received.")
        return
    if cmd == "light_on":
        print("LIGHT ON command received.")
        return
    if cmd == "light_off":
        print("LIGHT OFF command received.")
        return

    if cmd == "climate":
        print("CLIMATE command received.")
        return
    if cmd == "climate_temp_up":
        led2_brightness = max(0, led2_brightness + 50)
        send_led2_brightness(led2_brightness)
        return
    if cmd == "climate_temp_down":
        led2_brightness = max(0, led2_brightness - 50)
        send_led2_brightness(led2_brightness)
        return
    if cmd == "tv_volume_up":
        led1_brightness = max(0, led1_brightness + 50)
        send_led1_brightness(led1_brightness)
        return
    if cmd == "tv_volume_down":
        led1_brightness = max(0, led1_brightness - 50)
        send_led1_brightness(led1_brightness)
        return
    if cmd == "tv_channel_up":
        led1_power_on = True
        led1_blink_enabled = True
        led1_blink_interval_sec = max(0.1, led1_blink_interval_sec - 0.1)
        print(f"LED1 blink faster: interval {led1_blink_interval_sec:.2f}s")
        return
    if cmd == "tv_channel_down":
        led1_power_on = True
        led1_blink_enabled = True
        led1_blink_interval_sec = min(1.5, led1_blink_interval_sec + 0.1)
        print(f"LED1 blink slower: interval {led1_blink_interval_sec:.2f}s")
        return
    if cmd == "tv_power_off":
        led1_power_on = False
        led1_blink_enabled = False
        send_led1_off()
        print("TV power off: LED1 forced OFF, blinking disabled.")
        return
    if cmd == "tv_power_on":
        led1_power_on = True
        print("TV power on.")
        return

    if cmd == "help":
        print_help_menu()
        if send_help_alert():
            print("Help alert sent to mobile.")
        else:
            print("Help alert not sent (set NTFY_TOPIC and internet access).")
        publish_status("help")
        return


def reset_motion_state():
    return {
        "smoothed_x": None,
        "smoothed_y": None,
        "prev_smoothed_x": None,
        "prev_smoothed_y": None,
        "last_dx_pix": 0,
        "last_dy_pix": 0,
        "hold_x_frames": 0,
        "hold_y_frames": 0,
        "last_seen_time": None,
    }


def detect_face_landmarks(gray, face_detector, facemark):
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    ok, landmarks = facemark.fit(gray, np.array([[x, y, w, h]], dtype=np.int32))
    if not ok or len(landmarks) == 0:
        return None

    pts = landmarks[0][0]
    ih, iw = gray.shape[:2]
    landmarks_norm = []
    for px, py in pts:
        nx = float(px) / float(max(iw, 1))
        ny = float(py) / float(max(ih, 1))
        landmarks_norm.append((nx, ny))
    return landmarks_norm


def try_open_camera_source(source, backend):
    cap = cv2.VideoCapture(source, backend)
    if cap is None or not cap.isOpened():
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def open_camera():
    candidates = []

    # Prefer explicit V4L2 device paths on Linux/Raspberry Pi.
    video_indices = []
    try:
        for name in os.listdir("/dev"):
            if name.startswith("video"):
                suffix = name[5:]
                if suffix.isdigit():
                    video_indices.append(int(suffix))
    except Exception:
        pass

    video_indices = sorted(set(video_indices))
    for idx in video_indices:
        dev = f"/dev/video{idx}"
        candidates.append((dev, cv2.CAP_V4L2))
        candidates.append((dev, cv2.CAP_ANY))

    # Fallback to numeric indexes.
    max_probe = max(video_indices) if video_indices else 5
    for i in range(max_probe + 1):
        candidates.append((i, cv2.CAP_V4L2))
        candidates.append((i, cv2.CAP_ANY))

    seen = set()
    for source, backend in candidates:
        key = (str(source), int(backend))
        if key in seen:
            continue
        seen.add(key)
        cap = try_open_camera_source(source, backend)
        if cap is not None:
            return cap, source, backend

    return None, None, None


def read_gui_bounds():
    if not os.path.exists(BOUNDS_PATH):
        return None
    try:
        with open(BOUNDS_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        parts = [int(p) for p in raw.split(",")]
        if len(parts) != 4:
            return None
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h
    except Exception:
        return None


def move_cursor_to_center():
    if pyautogui is None:
        return
    try:
        bounds = read_gui_bounds()
        if bounds is not None:
            bx, by, bw, bh = bounds
            cx = int(bx + (bw / 2))
            cy = int(by + (bh / 2))
            pyautogui.moveTo(cx, cy, duration=0)
            return
        sw, sh = pyautogui.size()
        pyautogui.moveTo(int(sw / 2), int(sh / 2), duration=0)
    except Exception:
        pass


def publish_status(event: str) -> None:
    try:
        with open(STATUS_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f}|{event}\n")
    except Exception:
        pass


class CursorOverlay:
    def __init__(self, enabled: bool):
        self.enabled = enabled and tk is not None and pyautogui is not None
        self.root = None
        self._visible = False
        self._last_create_attempt = 0.0
        if not self.enabled:
            return
        self._create_window()

    def _create_window(self):
        self._last_create_attempt = time.time()
        try:
            self.root = tk.Tk()
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            self.root.configure(bg="black")
            self.root.geometry(f"{CUSTOM_CURSOR_SIZE}x{CUSTOM_CURSOR_SIZE}+0+0")

            canvas = tk.Canvas(
                self.root,
                width=CUSTOM_CURSOR_SIZE,
                height=CUSTOM_CURSOR_SIZE,
                bg="black",
                highlightthickness=0,
                bd=0,
            )
            canvas.pack()
            pad = max(2, CUSTOM_CURSOR_THICKNESS)
            canvas.create_oval(
                pad,
                pad,
                CUSTOM_CURSOR_SIZE - pad,
                CUSTOM_CURSOR_SIZE - pad,
                outline=CUSTOM_CURSOR_COLOR,
                width=CUSTOM_CURSOR_THICKNESS,
            )
            canvas.create_oval(
                CUSTOM_CURSOR_SIZE // 2 - 2,
                CUSTOM_CURSOR_SIZE // 2 - 2,
                CUSTOM_CURSOR_SIZE // 2 + 2,
                CUSTOM_CURSOR_SIZE // 2 + 2,
                fill=CUSTOM_CURSOR_COLOR,
                outline=CUSTOM_CURSOR_COLOR,
            )
            self.root.withdraw()
            self._visible = False
            self.root.update_idletasks()
        except Exception as e:
            print(f"Warning: custom cursor overlay unavailable: {e}")
            self.root = None

    def update(self, show: bool):
        if not self.enabled:
            return
        if self.root is None:
            # Keep retrying so the custom cursor can recover.
            if (time.time() - self._last_create_attempt) >= 1.0:
                self._create_window()
            return
        try:
            if not show:
                if self._visible:
                    self.root.withdraw()
                    self._visible = False
                self.root.update_idletasks()
                return

            x, y = pyautogui.position()
            offset = CUSTOM_CURSOR_SIZE // 2
            self.root.geometry(f"+{int(x) - offset}+{int(y) - offset}")
            if not self._visible:
                self.root.deiconify()
                self._visible = True
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
            self._visible = False

    def close(self):
        if self.root is None:
            return
        try:
            self.root.withdraw()
            self.root.update_idletasks()
            self.root.destroy()
        except Exception:
            pass
        self.root = None
        self.enabled = False

    def hide_temporarily(self):
        if not self.enabled or self.root is None:
            return
        try:
            self.root.withdraw()
            self._visible = False
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            pass


def main() -> None:
    has_display = os.name == "nt" or "DISPLAY" in os.environ
    show_preview = SHOW_CAMERA_PREVIEW and has_display
    if not has_display:
        print("Warning: no DISPLAY found. GUI preview disabled (plain SSH).")

    if not ensure_lbf_model():
        return

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
        print("Error: OpenCV contrib module is required (cv2.face.createFacemarkLBF missing).")
        print("Install package that includes opencv-contrib-python.")
        return

    cascade_path = resolve_haar_cascade_path()
    if cascade_path is None:
        print("Error: could not find haarcascade_frontalface_default.xml")
        print("Install OpenCV haarcascade data package for your system.")
        return

    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        print("Error: could not load Haar cascade face detector.")
        return

    facemark = cv2.face.createFacemarkLBF()
    try:
        facemark.loadModel(LBF_MODEL_PATH)
    except Exception as e:
        print(f"Error loading landmark model: {e}")
        return

    cap, cam_source, cam_backend = open_camera()
    if cap is None:
        print("Error: camera could not be opened.")
        print("Checked all discovered /dev/video* devices and matching indexes with V4L2/CAP_ANY.")
        return
    print(f"Camera opened: source={cam_source}, backend={cam_backend}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    mouse_enabled = MOUSE_CONTROL_ENABLED
    if pyautogui is None and mouse_enabled:
        print("Warning: pyautogui unavailable, mouse movement disabled.")
        mouse_enabled = False
    elif mouse_enabled:
        try:
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0
        except Exception:
            mouse_enabled = False
            print("Warning: pyautogui loaded but mouse setup failed.")

    state = reset_motion_state()
    last_move_time = 0.0

    blink_is_closed = False
    blink_closed_start_time = 0.0
    blink_reopen_start_time = 0.0
    blink_count = 0
    first_blink_time = 0.0
    last_valid_blink_time = 0.0
    last_click_time = 0.0
    last_blink_score = None
    blink_open_baseline = None
    blink_score_smooth = None
    blink_closed_frames = 0
    blink_open_frames = 0
    cursor_overlay = CursorOverlay(CUSTOM_CURSOR_ENABLED and has_display)

    # Dwell click state (cursor-stationary click).
    dwell_anchor_xy = None  # (x,y)
    dwell_anchor_t = 0.0
    last_dwell_click_t = 0.0

    # Start from a known good position.
    if mouse_enabled and pyautogui is not None:
        move_cursor_to_center()

    print("Camera running. Press q to quit preview if preview is enabled.")

    global led1_blink_thread
    if led1_blink_thread is None:
        led1_blink_stop_event.clear()
        led1_blink_thread = threading.Thread(target=_led1_blink_loop, daemon=True)
        led1_blink_thread.start()

    try:
        while True:
            # Keep the custom cursor visible even when no face is detected.
            cursor_overlay.update(True)
            if os.path.exists(CMD_PATH):
                try:
                    with open(CMD_PATH, "r", encoding="utf-8") as f:
                        cmd = f.read().strip().lower()
                except Exception:
                    cmd = ""

                try:
                    os.remove(CMD_PATH)
                except Exception:
                    pass

                if cmd == "exit":
                    break
                elif cmd == "toggle":
                    mouse_enabled = not mouse_enabled
                    state = reset_motion_state()
                    print(f"Mouse control: {mouse_enabled}")
                elif cmd == "calibrate":
                    if state["smoothed_x"] is not None and state["smoothed_y"] is not None:
                        state["prev_smoothed_x"] = state["smoothed_x"]
                        state["prev_smoothed_y"] = state["smoothed_y"]
                        move_cursor_to_center()
                        print("Neutral recalibrated.")
                elif cmd in (
                    "climate",
                    "climate_temp_up",
                    "climate_temp_down",
                    "tv_power_on",
                    "tv_power_off",
                    "tv_volume_up",
                    "tv_volume_down",
                    "tv_channel_up",
                    "tv_channel_down",
                    "light",
                    "light_on",
                    "light_off",
                    "keyboard",
                    "help",
                ):
                    handle_external_action(cmd)

            ret, frame = cap.read()
            if not ret:
                print("Warning: camera frame read failed, retrying...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(CAMERA_RETRY_DELAY_SEC)
                cap, cam_source, cam_backend = open_camera()
                if cap is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    print(f"Camera reopened: source={cam_source}, backend={cam_backend}")
                    continue
                print("Warning: camera reopen failed, will retry...")
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks_norm = detect_face_landmarks(gray, face_detector, facemark)

            if landmarks_norm is not None and NOSE_TIP_INDEX < len(landmarks_norm):
                h, w, _ = frame.shape
                nose_x_norm, nose_y_norm = landmarks_norm[NOSE_TIP_INDEX]

                if nose_x_norm is None or nose_y_norm is None:
                    # Keep loop alive for blink handling even if nose point is temporarily unstable.
                    nose_x_norm = state["smoothed_x"] if state["smoothed_x"] is not None else 0.5
                    nose_y_norm = state["smoothed_y"] if state["smoothed_y"] is not None else 0.5

                raw_x = float(nose_x_norm)
                raw_y = float(nose_y_norm)
                now = time.time()

                if (
                    state["last_seen_time"] is not None
                    and (now - state["last_seen_time"]) > RESET_AFTER_LOST_FACE_SEC
                ):
                    state = reset_motion_state()

                state["last_seen_time"] = now

                if state["smoothed_x"] is None or state["smoothed_y"] is None:
                    state["smoothed_x"] = raw_x
                    state["smoothed_y"] = raw_y
                    state["prev_smoothed_x"] = raw_x
                    state["prev_smoothed_y"] = raw_y
                    state["last_dx_pix"] = 0
                    state["last_dy_pix"] = 0
                    state["hold_x_frames"] = 0
                    state["hold_y_frames"] = 0
                else:
                    state["smoothed_x"] = SMOOTHING_ALPHA * state["smoothed_x"] + (1.0 - SMOOTHING_ALPHA) * raw_x
                    state["smoothed_y"] = SMOOTHING_ALPHA * state["smoothed_y"] + (1.0 - SMOOTHING_ALPHA) * raw_y

                dx_pix = 0
                dy_pix = 0

                if mouse_enabled and HEAD_DIRECTION_CONTROL_ENABLED:
                    delta_x = state["smoothed_x"] - state["prev_smoothed_x"]
                    delta_y = state["smoothed_y"] - state["prev_smoothed_y"]
                    state["prev_smoothed_x"] = state["smoothed_x"]
                    state["prev_smoothed_y"] = state["smoothed_y"]

                    x_sign = -1.0 if INVERT_X else 1.0
                    delta_x_signed = delta_x * x_sign
                    x_deadzone = X_DEADZONE_NEG_NORM if delta_x_signed < 0 else X_DEADZONE_POS_NORM
                    x_scale_mult = X_SCALE_NEG_MULTIPLIER if delta_x_signed < 0 else X_SCALE_POS_MULTIPLIER
                    if abs(delta_x_signed) >= x_deadzone:
                        dx_pix = int(delta_x_signed * ORIENTATION_PIXEL_SCALE_X * x_scale_mult)
                        if dx_pix == 0:
                            dx_pix = -1 if delta_x_signed < 0 else 1
                        dx_pix = max(-MAX_STEP_PIXELS_X, min(MAX_STEP_PIXELS_X, dx_pix))
                        state["last_dx_pix"] = dx_pix
                        state["hold_x_frames"] = MOTION_HOLD_FRAMES
                    elif state.get("hold_x_frames", 0) > 0:
                        dx_pix = int(state.get("last_dx_pix", 0))
                        if MOTION_HOLD_DECAY_ENABLED:
                            hf = int(state.get("hold_x_frames", 0))
                            denom = float(max(1, MOTION_HOLD_FRAMES))
                            scale = float(hf) / denom
                            dx_pix = int(round(dx_pix * scale))
                            if abs(dx_pix) < 1:
                                dx_pix = 0
                        state["hold_x_frames"] = int(state["hold_x_frames"]) - 1

                    y_sign = -1.0 if INVERT_Y else 1.0
                    delta_y_signed = delta_y * y_sign
                    y_deadzone = Y_DEADZONE_NEG_NORM if delta_y_signed < 0 else Y_DEADZONE_POS_NORM
                    y_scale_mult = Y_SCALE_NEG_MULTIPLIER if delta_y_signed < 0 else Y_SCALE_POS_MULTIPLIER
                    if abs(delta_y_signed) >= y_deadzone:
                        dy_pix = int(delta_y_signed * ORIENTATION_PIXEL_SCALE_Y * y_scale_mult)
                        if dy_pix == 0:
                            dy_pix = -1 if delta_y_signed < 0 else 1
                        dy_pix = max(-MAX_STEP_PIXELS_Y, min(MAX_STEP_PIXELS_Y, dy_pix))
                        state["last_dy_pix"] = dy_pix
                        state["hold_y_frames"] = MOTION_HOLD_FRAMES
                    elif state.get("hold_y_frames", 0) > 0:
                        dy_pix = int(state.get("last_dy_pix", 0))
                        if MOTION_HOLD_DECAY_ENABLED:
                            hf = int(state.get("hold_y_frames", 0))
                            denom = float(max(1, MOTION_HOLD_FRAMES))
                            scale = float(hf) / denom
                            dy_pix = int(round(dy_pix * scale))
                            if abs(dy_pix) < 1:
                                dy_pix = 0
                        state["hold_y_frames"] = int(state["hold_y_frames"]) - 1

                now = time.time()
                if mouse_enabled and pyautogui is not None and (now - last_move_time >= MOVE_INTERVAL_SEC):
                    if abs(dx_pix) >= MIN_MOVE_PIXELS or abs(dy_pix) >= MIN_MOVE_PIXELS:
                        try:
                            bounds = read_gui_bounds()
                            if bounds is not None:
                                bx, by, bw, bh = bounds
                                min_x = bx + 1
                                min_y = by + 1
                                max_x = bx + bw - 2
                                max_y = by + bh - 2
                                cur_x, cur_y = pyautogui.position()
                                target_x = max(min_x, min(max_x, int(cur_x) + int(dx_pix)))
                                target_y = max(min_y, min(max_y, int(cur_y) + int(dy_pix)))
                                pyautogui.moveTo(target_x, target_y, duration=0)
                            else:
                                pyautogui.moveRel(dx_pix, dy_pix, duration=0)
                        except Exception:
                            pass
                    last_move_time = now

                # Dwell/hover click: if the cursor stays in place over the GUI for DWELL_CLICK_SEC,
                # fire a single click (with cooldown). This is independent of blink click.
                if DWELL_CLICK_ENABLED and mouse_enabled and pyautogui is not None:
                    try:
                        bounds = read_gui_bounds()
                        if bounds is not None:
                            bx, by, bw, bh = bounds
                            cur_x, cur_y = pyautogui.position()
                            inside_gui = (bx <= cur_x <= (bx + bw)) and (by <= cur_y <= (by + bh))
                        else:
                            inside_gui = True
                            cur_x, cur_y = pyautogui.position()

                        now_dwell = time.time()
                        if not inside_gui:
                            dwell_anchor_xy = None
                            dwell_anchor_t = 0.0
                        else:
                            if dwell_anchor_xy is None:
                                dwell_anchor_xy = (int(cur_x), int(cur_y))
                                dwell_anchor_t = now_dwell
                            else:
                                ax, ay = dwell_anchor_xy
                                if abs(int(cur_x) - ax) > int(DWELL_MOVE_TOL_PX) or abs(int(cur_y) - ay) > int(DWELL_MOVE_TOL_PX):
                                    dwell_anchor_xy = (int(cur_x), int(cur_y))
                                    dwell_anchor_t = now_dwell
                                else:
                                    # Stable within tolerance; check dwell time and cooldown.
                                    if (now_dwell - dwell_anchor_t) >= float(DWELL_CLICK_SEC) and (now_dwell - last_dwell_click_t) >= float(DWELL_COOLDOWN_SEC):
                                        try:
                                            cursor_overlay.hide_temporarily()
                                            pyautogui.click()
                                            publish_status("dwell_click")
                                        except Exception:
                                            pass
                                        last_dwell_click_t = now_dwell
                                        # Re-anchor to prevent repeated clicking while staying still.
                                        dwell_anchor_xy = (int(cur_x), int(cur_y))
                                        dwell_anchor_t = now_dwell
                    except Exception:
                        dwell_anchor_xy = None
                        dwell_anchor_t = 0.0

                if BLINK_DOUBLE_CLICK_ENABLED and pyautogui is not None:
                    # Ignore blink events briefly after a click to avoid duplicate click firing.
                    if (time.time() - last_click_time) < POST_CLICK_BLINK_LOCK_SEC:
                        blink_is_closed = False
                        blink_reopen_start_time = 0.0
                        blink_count = 0
                        first_blink_time = 0.0
                        blink_closed_frames = 0
                        blink_open_frames = 0
                        continue

                    left_ratio = eye_open_ratio(landmarks_norm, LEFT_EYE_INDICES)
                    right_ratio = eye_open_ratio(landmarks_norm, RIGHT_EYE_INDICES)
                    blink_score = (left_ratio + right_ratio) / 2.0
                    eye_asymmetry = abs(left_ratio - right_ratio)

                    if blink_score_smooth is None:
                        blink_score_smooth = blink_score
                    else:
                        blink_score_smooth = (
                            BLINK_SCORE_SMOOTH_ALPHA * blink_score_smooth
                            + (1.0 - BLINK_SCORE_SMOOTH_ALPHA) * blink_score
                        )
                    last_blink_score = blink_score_smooth

                    closed_threshold = BLINK_RATIO_CLOSED
                    open_threshold = BLINK_RATIO_OPEN
                    if BLINK_ADAPTIVE_ENABLED:
                        if blink_open_baseline is None:
                            blink_open_baseline = blink_score_smooth
                        # Track natural open-eye ratio slowly for robust thresholds.
                        if blink_score_smooth > blink_open_baseline:
                            blink_open_baseline = (0.92 * blink_open_baseline) + (0.08 * blink_score_smooth)
                        else:
                            blink_open_baseline = (0.995 * blink_open_baseline) + (0.005 * blink_score_smooth)
                        closed_threshold = max(0.14, blink_open_baseline * 0.82)
                        open_threshold = max(closed_threshold + 0.005, blink_open_baseline * 0.84)

                    # Balanced tolerance for natural/asymmetric blinks without false positives.
                    closed_threshold = max(closed_threshold, BLINK_RATIO_CLOSED + 0.01)
                    open_threshold = min(open_threshold, BLINK_RATIO_OPEN * 0.98)

                    # Sequence timeout: if second blink does not arrive in time, reset sequence.
                    if blink_count == 1 and first_blink_time > 0.0:
                        if (time.time() - first_blink_time) > DOUBLE_BLINK_WINDOW_SEC:
                            blink_count = 0
                            first_blink_time = 0.0

                    # Keep detection tolerant to natural left/right eye mismatch.
                    both_eyes_closed = (
                        left_ratio < (closed_threshold * 1.12)
                        and right_ratio < (closed_threshold * 1.12)
                    )
                    one_eye_strongly_closed = (
                        min(left_ratio, right_ratio) < (closed_threshold * 0.90)
                        and blink_score_smooth < (closed_threshold * 1.03)
                        and eye_asymmetry <= (BLINK_MAX_EYE_ASYMMETRY * 1.6)
                    )
                    is_closed_frame = both_eyes_closed or one_eye_strongly_closed

                    is_open_frame = (
                        blink_score_smooth > (open_threshold * 0.96)
                        and left_ratio > (open_threshold * 0.88)
                        and right_ratio > (open_threshold * 0.88)
                        and eye_asymmetry <= (BLINK_MAX_EYE_ASYMMETRY * 2.0)
                    )

                    now_blink = time.time()
                    if is_closed_frame:
                        blink_closed_frames += 1
                        blink_open_frames = 0
                        if not blink_is_closed and blink_closed_frames >= BLINK_MIN_CLOSED_FRAMES:
                            blink_is_closed = True
                            blink_closed_start_time = now_blink
                        blink_reopen_start_time = 0.0
                    else:
                        if is_open_frame:
                            blink_open_frames += 1
                        else:
                            blink_open_frames = 0
                            if not blink_is_closed:
                                blink_closed_frames = 0
                        # Eye appears open; require a short stable-open confirmation.
                        if blink_is_closed and blink_open_frames >= BLINK_MIN_OPEN_FRAMES:
                            if blink_reopen_start_time <= 0.0:
                                blink_reopen_start_time = now_blink

                            if (now_blink - blink_reopen_start_time) >= BLINK_EVENT_OPEN_CONFIRM_SEC:
                                closed_dur = now_blink - blink_closed_start_time
                                if closed_dur >= BLINK_CLOSED_MIN_SEC:
                                    if (
                                        (now_blink - last_valid_blink_time) >= BLINK_INTER_BLINK_MIN_SEC
                                        and (
                                            last_valid_blink_time <= 0.0
                                            or (now_blink - last_valid_blink_time) >= BLINK_REQUIRED_OPEN_GAP_SEC
                                        )
                                    ):
                                        last_valid_blink_time = now_blink
                                        blink_count += 1
                                        publish_status(f"blink_{blink_count}")
                                        if blink_count == 1:
                                            first_blink_time = now_blink
                                        elif blink_count == 2:
                                            if (
                                                (now_blink - first_blink_time) <= DOUBLE_BLINK_WINDOW_SEC
                                                and (now_blink - last_click_time) >= CLICK_COOLDOWN_SEC
                                            ):
                                                try:
                                                    # Ensure click reaches widget under cursor.
                                                    cursor_overlay.hide_temporarily()
                                                    pyautogui.click()
                                                    publish_status("blink_click")
                                                except Exception:
                                                    pass
                                                last_click_time = now_blink
                                                # Hard reset blink sequence after click.
                                                blink_is_closed = False
                                                blink_reopen_start_time = 0.0

                                            blink_count = 0
                                            first_blink_time = 0.0
                                        else:
                                            # More than two blinks: restart from current blink.
                                            blink_count = 1
                                            first_blink_time = now_blink

                                blink_is_closed = False
                                blink_reopen_start_time = 0.0
                                blink_closed_frames = 0
                                blink_open_frames = 0
                        else:
                            # Not confidently open yet; keep waiting.
                            blink_reopen_start_time = 0.0

                    if show_preview and last_blink_score is not None:
                        text = f"Blink score: {last_blink_score:.3f}"
                        if BLINK_ADAPTIVE_ENABLED and blink_open_baseline is not None:
                            text += f" th:{closed_threshold:.3f}/{open_threshold:.3f}"
                        if blink_count > 0:
                            text += f" count: {blink_count}"
                        cv2.putText(
                            frame,
                            text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

                nose_x = int(nose_x_norm * w)
                nose_y = int(nose_y_norm * h)

                if show_preview:
                    cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "Nose",
                        (nose_x + 10, nose_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                if show_preview:
                    draw_connections(frame, landmarks_norm, NOSE_CONNECTIONS, color=(0, 255, 0), thickness=1)
                    draw_connections(frame, landmarks_norm, LEFT_EYE_CONNECTIONS, color=(255, 0, 255), thickness=1)
                    draw_connections(frame, landmarks_norm, RIGHT_EYE_CONNECTIONS, color=(255, 0, 255), thickness=1)
            else:
                if mouse_enabled:
                    if state["last_seen_time"] is not None and (time.time() - state["last_seen_time"]) > RESET_AFTER_LOST_FACE_SEC:
                        state = reset_motion_state()

                if blink_is_closed:
                    blink_is_closed = False
                blink_reopen_start_time = 0.0
                blink_count = 0
                first_blink_time = 0.0
                blink_closed_frames = 0
                blink_open_frames = 0

            if show_preview:
                cv2.imshow("Face Points", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    if state["smoothed_x"] is not None and state["smoothed_y"] is not None:
                        state["prev_smoothed_x"] = state["smoothed_x"]
                        state["prev_smoothed_y"] = state["smoothed_y"]
                        print("Neutral recalibrated.")
                elif key == ord("m"):
                    mouse_enabled = not mouse_enabled
                    state = reset_motion_state()
                    print(f"Mouse control: {mouse_enabled}")
    finally:
        led1_blink_stop_event.set()
        cursor_overlay.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import os
import subprocess
import sys
import time

import cv2
import mediapipe as mp

try:
    import pyautogui
except Exception as e:
    pyautogui = None
    _pyautogui_import_error = e


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CMD_PATH = os.path.join(BASE_DIR, "head_tracking_cmd.txt")

# ----------------------------
# SETTINGS
# ----------------------------
MOUSE_CONTROL_ENABLED = True
SHOW_CAMERA_PREVIEW = True

SMOOTHING_ALPHA = 0.8
MOVE_INTERVAL_SEC = 0.03
MIN_MOVE_PIXELS = 1
RESET_AFTER_LOST_FACE_SEC = 0.6

INVERT_X = False
INVERT_Y = False

HEAD_DIRECTION_CONTROL_ENABLED = True
YAW_DEADZONE_NORM = 0.015
PITCH_DEADZONE_NORM = 0.015
YAW_STOP_DEADZONE_NORM = 0.10
PITCH_STOP_DEADZONE_NORM = 0.10
STOP_CONFIRM_FRAMES = 5

ORIENTATION_PIXEL_SCALE_X = 300
ORIENTATION_PIXEL_SCALE_Y = 300
MAX_STEP_PIXELS_X = 5
MAX_STEP_PIXELS_Y = 5

BLINK_DOUBLE_CLICK_ENABLED = True
BLINK_RATIO_CLOSED = 0.20
BLINK_RATIO_OPEN = 0.26
BLINK_CLOSED_MIN_SEC = 0.07
DOUBLE_BLINK_WINDOW_SEC = 1.0
CLICK_COOLDOWN_SEC = 1.0

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FACE_MESH = mp.solutions.face_mesh
LEFT_EYE_CONNECTIONS = FACE_MESH.FACEMESH_LEFT_EYE
RIGHT_EYE_CONNECTIONS = FACE_MESH.FACEMESH_RIGHT_EYE
NOSE_CONNECTIONS = FACE_MESH.FACEMESH_NOSE

LEFT_EYE_INDICES = {c[0] for c in LEFT_EYE_CONNECTIONS} | {c[1] for c in LEFT_EYE_CONNECTIONS}
RIGHT_EYE_INDICES = {c[0] for c in RIGHT_EYE_CONNECTIONS} | {c[1] for c in RIGHT_EYE_CONNECTIONS}


def draw_connections(frame, landmarks, connections, color=(0, 255, 0), thickness=1) -> None:
    h, w, _ = frame.shape
    for conn in connections:
        start_idx, end_idx = conn
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        if start.x is None or start.y is None or end.x is None or end.y is None:
            continue

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)

        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def eye_open_ratio(face_landmarks, eye_connections) -> float:
    xs = []
    ys = []
    indices = set()
    for c in eye_connections:
        indices.add(c[0])
        indices.add(c[1])

    for idx in indices:
        lm = face_landmarks[idx]
        if lm.x is None or lm.y is None:
            continue
        xs.append(float(lm.x))
        ys.append(float(lm.y))

    if len(xs) < 2 or len(ys) < 2:
        return 1.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    eye_width = max(max_x - min_x, 1e-6)
    eye_height = max_y - min_y
    return eye_height / eye_width


def mean_xy_for_indices(face_landmarks, indices):
    xs = []
    ys = []
    for idx in indices:
        lm = face_landmarks[idx]
        if lm.x is None or lm.y is None:
            continue
        xs.append(float(lm.x))
        ys.append(float(lm.y))
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def head_yaw_pitch(face_landmarks):
    nose = face_landmarks[1]
    if nose.x is None or nose.y is None:
        return None, None

    left_eye = mean_xy_for_indices(face_landmarks, LEFT_EYE_INDICES)
    right_eye = mean_xy_for_indices(face_landmarks, RIGHT_EYE_INDICES)
    if left_eye is None or right_eye is None:
        return None, None

    left_x, left_y = left_eye
    right_x, right_y = right_eye
    eye_mid_x = (left_x + right_x) / 2.0
    eye_mid_y = (left_y + right_y) / 2.0

    eye_dx = right_x - left_x
    denom = max(abs(eye_dx), 1e-6)

    yaw = (float(nose.x) - eye_mid_x) / denom
    pitch = (float(nose.y) - eye_mid_y) / denom
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


def handle_external_action(cmd: str):
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

    if cmd == "climate":
        print("CLIMATE command received.")
        return

    if cmd == "help":
        print("HELP command received.")
        return


def reset_motion_state():
    return {
        "smoothed_x": None,
        "smoothed_y": None,
        "prev_smoothed_x": None,
        "prev_smoothed_y": None,
        "last_seen_time": None,
        "neutral_yaw": None,
        "neutral_pitch": None,
        "smoothed_yaw": None,
        "smoothed_pitch": None,
        "stop_streak": 0,
    }


def main() -> None:
    if os.name != "nt" and "DISPLAY" not in os.environ:
        print("Warning: no DISPLAY found. GUI/mouse control may not work from plain SSH.")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: camera could not be opened.")
        return

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
    blink_count = 0
    first_blink_time = 0.0
    last_click_time = 0.0
    last_blink_score = None

    with FACE_MESH.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("Camera running. Press q to quit preview if preview is enabled.")

        while True:
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
                    if state["smoothed_yaw"] is not None and state["smoothed_pitch"] is not None:
                        state["neutral_yaw"] = state["smoothed_yaw"]
                        state["neutral_pitch"] = state["smoothed_pitch"]
                        print("Neutral recalibrated.")

                elif cmd in ("climate", "light", "keyboard", "help"):
                    handle_external_action(cmd)

            ret, frame = cap.read()
            if not ret:
                print("Error: could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                face_landmarks = result.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                nose_tip = face_landmarks[1]

                if nose_tip.x is not None and nose_tip.y is not None:
                    raw_x = float(nose_tip.x)
                    raw_y = float(nose_tip.y)
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
                    else:
                        state["smoothed_x"] = SMOOTHING_ALPHA * state["smoothed_x"] + (1.0 - SMOOTHING_ALPHA) * raw_x
                        state["smoothed_y"] = SMOOTHING_ALPHA * state["smoothed_y"] + (1.0 - SMOOTHING_ALPHA) * raw_y

                    dx_pix = 0
                    dy_pix = 0

                    if mouse_enabled and HEAD_DIRECTION_CONTROL_ENABLED:
                        yaw_raw, pitch_raw = head_yaw_pitch(face_landmarks)
                        if yaw_raw is not None and pitch_raw is not None:
                            if state["neutral_yaw"] is None or state["neutral_pitch"] is None:
                                state["neutral_yaw"] = yaw_raw
                                state["neutral_pitch"] = pitch_raw
                                state["smoothed_yaw"] = yaw_raw
                                state["smoothed_pitch"] = pitch_raw
                            else:
                                if state["smoothed_yaw"] is None or state["smoothed_pitch"] is None:
                                    state["smoothed_yaw"] = yaw_raw
                                    state["smoothed_pitch"] = pitch_raw
                                else:
                                    state["smoothed_yaw"] = SMOOTHING_ALPHA * state["smoothed_yaw"] + (1.0 - SMOOTHING_ALPHA) * yaw_raw
                                    state["smoothed_pitch"] = SMOOTHING_ALPHA * state["smoothed_pitch"] + (1.0 - SMOOTHING_ALPHA) * pitch_raw

                            rel_yaw = state["smoothed_yaw"] - state["neutral_yaw"]
                            rel_pitch = state["smoothed_pitch"] - state["neutral_pitch"]

                            in_stop_band = (
                                abs(rel_yaw) < YAW_STOP_DEADZONE_NORM
                                and abs(rel_pitch) < PITCH_STOP_DEADZONE_NORM
                            )

                            if in_stop_band:
                                state["stop_streak"] += 1
                            else:
                                state["stop_streak"] = 0

                            if abs(rel_yaw) >= YAW_DEADZONE_NORM:
                                dx_pix = int(rel_yaw * ORIENTATION_PIXEL_SCALE_X * (-1 if INVERT_X else 1))
                                dx_pix = max(-MAX_STEP_PIXELS_X, min(MAX_STEP_PIXELS_X, dx_pix))

                            if abs(rel_pitch) >= PITCH_DEADZONE_NORM:
                                dy_pix = int(rel_pitch * ORIENTATION_PIXEL_SCALE_Y * (-1 if INVERT_Y else 1))
                                dy_pix = max(-MAX_STEP_PIXELS_Y, min(MAX_STEP_PIXELS_Y, dy_pix))

                            if state["stop_streak"] >= STOP_CONFIRM_FRAMES:
                                dx_pix = 0
                                dy_pix = 0

                    now = time.time()
                    if mouse_enabled and pyautogui is not None and (now - last_move_time >= MOVE_INTERVAL_SEC):
                        if abs(dx_pix) >= MIN_MOVE_PIXELS or abs(dy_pix) >= MIN_MOVE_PIXELS:
                            try:
                                pyautogui.moveRel(dx_pix, dy_pix, duration=0)
                            except Exception:
                                pass
                        last_move_time = now

                    if BLINK_DOUBLE_CLICK_ENABLED and pyautogui is not None:
                        left_ratio = eye_open_ratio(
                            face_landmarks, LEFT_EYE_CONNECTIONS
                        )
                        right_ratio = eye_open_ratio(
                            face_landmarks, RIGHT_EYE_CONNECTIONS
                        )
                        blink_score = (left_ratio + right_ratio) / 2.0
                        last_blink_score = blink_score

                        if blink_score < BLINK_RATIO_CLOSED:
                            if not blink_is_closed:
                                blink_is_closed = True
                                blink_closed_start_time = time.time()
                        else:
                            if blink_is_closed and blink_score > BLINK_RATIO_OPEN:
                                closed_dur = time.time() - blink_closed_start_time
                                if closed_dur >= BLINK_CLOSED_MIN_SEC:
                                    now_blink = time.time()

                                    if blink_count > 0 and (now_blink - first_blink_time) > DOUBLE_BLINK_WINDOW_SEC:
                                        blink_count = 0

                                    blink_count += 1
                                    if blink_count == 1:
                                        first_blink_time = now_blink
                                    elif blink_count == 2:
                                        if (
                                            (now_blink - first_blink_time) <= DOUBLE_BLINK_WINDOW_SEC
                                            and (now_blink - last_click_time) >= CLICK_COOLDOWN_SEC
                                        ):
                                            try:
                                                pyautogui.click()
                                            except Exception:
                                                pass
                                            last_click_time = now_blink

                                        blink_count = 0
                                        first_blink_time = 0.0

                                blink_is_closed = False

                        if SHOW_CAMERA_PREVIEW and last_blink_score is not None:
                            text = f"Blink score: {last_blink_score:.3f}"
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

                    nose_x = int(nose_tip.x * w)
                    nose_y = int(nose_tip.y * h)

                    if SHOW_CAMERA_PREVIEW:
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

                if SHOW_CAMERA_PREVIEW:
                    draw_connections(
                        frame,
                        face_landmarks,
                        NOSE_CONNECTIONS,
                        color=(0, 255, 0),
                        thickness=1,
                    )
            else:
                if mouse_enabled:
                    if state["last_seen_time"] is not None and (time.time() - state["last_seen_time"]) > RESET_AFTER_LOST_FACE_SEC:
                        state = reset_motion_state()

                if blink_is_closed:
                    blink_is_closed = False
                blink_count = 0
                first_blink_time = 0.0

            if SHOW_CAMERA_PREVIEW:
                cv2.imshow("Face Points", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    if state["smoothed_yaw"] is not None and state["smoothed_pitch"] is not None:
                        state["neutral_yaw"] = state["smoothed_yaw"]
                        state["neutral_pitch"] = state["smoothed_pitch"]
                        print("Neutral recalibrated.")
                elif key == ord("m"):
                    mouse_enabled = not mouse_enabled
                    state = reset_motion_state()
                    print(f"Mouse control: {mouse_enabled}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
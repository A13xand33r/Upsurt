import os
import subprocess
import sys
import urllib.request
import time

import cv2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
)

try:
    import pyautogui
except Exception as e:
    pyautogui = None
    _pyautogui_import_error = e

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
CMD_PATH = os.path.join(os.path.dirname(__file__), "head_tracking_cmd.txt")

# Настройки за управление на курсора
MOUSE_CONTROL_ENABLED = True
MOVE_SENSITIVITY_X = 1800  # пиксела за 1.0 промяна в normalized X
MOVE_SENSITIVITY_Y = 1800  # пиксела за 1.0 промяна в normalized Y
SMOOTHING_ALPHA = 0.8  # 0..1 (по-голямо = повече изглаждане)
DEADZONE_NORM = 0.003  # игнорира много малки движения (normalized)
MOVE_INTERVAL_SEC = 0.03  # честота на преместване
MIN_MOVE_PIXELS = 1
RESET_AFTER_LOST_FACE_SEC = 0.6
INVERT_Y = False  # ако движиш глава нагоре и искаш курсора да отива нагоре

# Ако е False, няма да се отваря прозорец с камерата (само GUI-то трябва да се вижда).
SHOW_CAMERA_PREVIEW = False

# Управление на курсора по посока (докато главата е завъртяна)
HEAD_DIRECTION_CONTROL_ENABLED = True
YAW_DEADZONE_NORM = 0.015
PITCH_DEADZONE_NORM = 0.015
INVERT_X = False  # ако наляво/надясно са обърнати - сложи True
ORIENTATION_PIXEL_SCALE_X = 300  # колко пиксела за единица yaw (relative)
ORIENTATION_PIXEL_SCALE_Y = 300  # колко пиксела за единица pitch (relative)
MAX_STEP_PIXELS_X = 5
MAX_STEP_PIXELS_Y = 5

# Хистерезис за "стоп при гледане напред" (anti-micro-drift)
YAW_STOP_DEADZONE_NORM = 0.10  # по-голям обхват, но стопът изисква "задържане"
PITCH_STOP_DEADZONE_NORM = 0.10
STOP_CONFIRM_FRAMES = 5  # нужни последователни кадри в "напред" за реално спиране

LEFT_EYE_INDICES = {c.start for c in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE} | {
    c.end for c in FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
}
RIGHT_EYE_INDICES = {c.start for c in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE} | {
    c.end for c in FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
}

# Настройки за "двойно мигане = клик"
BLINK_DOUBLE_CLICK_ENABLED = True
BLINK_RATIO_CLOSED = 0.20  # ratio = eye_height / eye_width; по-ниско = по-затворено око
BLINK_RATIO_OPEN = 0.26  # хистерезис (за да не трепти броенето)
BLINK_CLOSED_MIN_SEC = 0.07  # минимално време "затворено", за валидно мигане
DOUBLE_BLINK_WINDOW_SEC = 1.0  # в този прозорец второто мигане тригърва клик
CLICK_COOLDOWN_SEC = 1.0  # анти-спам след клик


def ensure_model_downloaded() -> None:
    if os.path.exists(MODEL_PATH):
        return

    print("Изтеглям face_landmarker.task (може да отнеме минута)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Неуспешно изтегляне на модела от:\n{MODEL_URL}\nПричина: {e}"
        ) from e
    print("Моделът е изтеглен.")


def draw_connections(frame, landmarks, connections, color=(0, 255, 0), thickness=1) -> None:
    h, w, _ = frame.shape
    for conn in connections:
        start = landmarks[conn.start]
        end = landmarks[conn.end]
        if start.x is None or start.y is None or end.x is None or end.y is None:
            continue

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)

        # Skip obviously invalid points (sometimes landmarks can be out of bounds).
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def eye_open_ratio(face_landmarks, eye_connections) -> float:
    """
    Връща метрика за "откритост" на окото.
    ratio = eye_height / eye_width (в normalized координати).
    При мигане (затваряне) ratio намалява.
    """
    xs = []
    ys = []
    indices = set()
    for c in eye_connections:
        indices.add(c.start)
        indices.add(c.end)

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
    """
    yaw/pitch спрямо очната линия.
    yaw: ляво/дясно (нос спрямо средата на очите)
    pitch: нагоре/надолу (нос спрямо средната хоризонтална линия на очите)
    """
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


def main() -> None:
    # Стартира GUI автоматично (ако не сме "gui-only").
    if os.environ.get("HEAD_TRACKING_NO_GUI") != "1":
        try:
            gui_path = os.path.join(os.path.dirname(__file__), "head_control_gui.py")
            env = dict(os.environ)
            env["HEAD_TRACKING_NO_GUI"] = "1"
            subprocess.Popen(
                [sys.executable, gui_path, "--gui-only"],
                cwd=os.path.dirname(__file__),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except Exception:
            # GUI не е критично за работата на сензора.
            pass

    ensure_model_downloaded()

    # Камера
    cap = cv2.VideoCapture(0)

    # Проверка дали камерата е отворена
    if not cap.isOpened():
        print("Грешка: камерата не може да се отвори.")
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Камерата работи. Натисни 'q' за изход.")

        mouse_enabled = MOUSE_CONTROL_ENABLED
        if pyautogui is None and mouse_enabled:
            print("Внимание: pyautogui не е наличен, курсора няма да се мести.")
            mouse_enabled = False
        elif mouse_enabled:
            pyautogui.FAILSAFE = False  # не спира при движение към ъгъл
            pyautogui.PAUSE = 0  # минимално забавяне

        smoothed_x = None
        smoothed_y = None
        prev_smoothed_x = None
        prev_smoothed_y = None
        last_seen_time = None
        last_move_time = 0.0

        neutral_yaw = None
        neutral_pitch = None
        smoothed_yaw = None
        smoothed_pitch = None
        stop_streak = 0

        # Blink state machine
        blink_is_closed = False
        blink_closed_start_time = 0.0
        blink_count = 0
        first_blink_time = 0.0
        last_click_time = 0.0
        last_blink_score = None

        print("Mouse control е включен. Натисни 'm' за toggle.")

        while True:
            # Обработка на команди от GUI (ако има).
            # GUI пише команда в head_tracking_cmd.txt: toggle / calibrate / exit
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
                if cmd == "toggle":
                    mouse_enabled = not mouse_enabled
                    if mouse_enabled:
                        smoothed_x = None
                        smoothed_y = None
                        prev_smoothed_x = None
                        prev_smoothed_y = None
                        last_seen_time = None
                        neutral_yaw = None
                        neutral_pitch = None
                        smoothed_yaw = None
                        smoothed_pitch = None
                        stop_streak = 0
                    print(f"Mouse control: {mouse_enabled}")
                if cmd == "calibrate":
                    if smoothed_yaw is not None and smoothed_pitch is not None:
                        neutral_yaw = smoothed_yaw
                        neutral_pitch = smoothed_pitch
                        print("Neutral recalibrated (GUI).")

            ret, frame = cap.read()
            if not ret:
                print("Грешка: не може да се прочете кадър от камерата.")
                break

            # Огледално обръщане за по-естествено усещане
            frame = cv2.flip(frame, 1)

            # OpenCV е BGR, MediaPipe очаква RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect(mp_image)

            # Ако има намерено лице
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                # Пример: взимаме координатите на върха на носа
                h, w, _ = frame.shape
                nose_tip = face_landmarks[1]

                if nose_tip.x is not None and nose_tip.y is not None:
                    # normalized координати в диапазон ~[0..1] спрямо входната картина
                    raw_x = float(nose_tip.x)
                    raw_y = float(nose_tip.y)
                    now = time.time()
                    # Ако дълго не е имало лице (загубена детекция) -> ресет, за да няма внезапен “скок”.
                    if (
                        last_seen_time is not None
                        and (now - last_seen_time) > RESET_AFTER_LOST_FACE_SEC
                    ):
                        smoothed_x = None
                        smoothed_y = None
                        prev_smoothed_x = None
                        prev_smoothed_y = None
                    last_seen_time = now

                    if smoothed_x is None or smoothed_y is None:
                        smoothed_x = raw_x
                        smoothed_y = raw_y
                        prev_smoothed_x = raw_x
                        prev_smoothed_y = raw_y
                    else:
                        smoothed_x = SMOOTHING_ALPHA * smoothed_x + (1.0 - SMOOTHING_ALPHA) * raw_x
                        smoothed_y = SMOOTHING_ALPHA * smoothed_y + (1.0 - SMOOTHING_ALPHA) * raw_y

                    # Управление на курсора по посока (yaw/pitch спрямо "гледай напред")
                    if mouse_enabled:
                        dx_pix = 0
                        dy_pix = 0

                        if HEAD_DIRECTION_CONTROL_ENABLED:
                            yaw_raw, pitch_raw = head_yaw_pitch(face_landmarks)
                            if yaw_raw is not None and pitch_raw is not None:
                                # Автокалибрация: първия стабилен "напред" става неутралата.
                                if neutral_yaw is None or neutral_pitch is None:
                                    neutral_yaw = yaw_raw
                                    neutral_pitch = pitch_raw
                                    smoothed_yaw = yaw_raw
                                    smoothed_pitch = pitch_raw
                                else:
                                    if smoothed_yaw is None or smoothed_pitch is None:
                                        smoothed_yaw = yaw_raw
                                        smoothed_pitch = pitch_raw
                                    else:
                                        smoothed_yaw = SMOOTHING_ALPHA * smoothed_yaw + (1.0 - SMOOTHING_ALPHA) * yaw_raw
                                        smoothed_pitch = SMOOTHING_ALPHA * smoothed_pitch + (1.0 - SMOOTHING_ALPHA) * pitch_raw

                                rel_yaw = smoothed_yaw - neutral_yaw if smoothed_yaw is not None else 0.0
                                rel_pitch = smoothed_pitch - neutral_pitch if smoothed_pitch is not None else 0.0

                                # Stop gating (anti-micro-drift):
                                # cursor stops only after staying in "forward" band for several frames.
                                in_stop_band = (
                                    abs(rel_yaw) < YAW_STOP_DEADZONE_NORM
                                    and abs(rel_pitch) < PITCH_STOP_DEADZONE_NORM
                                )
                                if in_stop_band:
                                    stop_streak += 1
                                else:
                                    stop_streak = 0

                                # Movement: use the smaller deadzones to avoid noise.
                                if abs(rel_yaw) >= YAW_DEADZONE_NORM:
                                    dx_pix = int(rel_yaw * ORIENTATION_PIXEL_SCALE_X * (-1 if INVERT_X else 1))
                                    dx_pix = max(-MAX_STEP_PIXELS_X, min(MAX_STEP_PIXELS_X, dx_pix))
                                else:
                                    dx_pix = 0

                                if abs(rel_pitch) >= PITCH_DEADZONE_NORM:
                                    dy_pix = int(rel_pitch * ORIENTATION_PIXEL_SCALE_Y * (-1 if INVERT_Y else 1))
                                    dy_pix = max(-MAX_STEP_PIXELS_Y, min(MAX_STEP_PIXELS_Y, dy_pix))
                                else:
                                    dy_pix = 0

                                # "Real stop": only when user held head forward long enough.
                                if stop_streak >= STOP_CONFIRM_FRAMES:
                                    dx_pix = 0
                                    dy_pix = 0

                        now = time.time()
                        if now - last_move_time >= MOVE_INTERVAL_SEC:
                            if abs(dx_pix) >= MIN_MOVE_PIXELS or abs(dy_pix) >= MIN_MOVE_PIXELS:
                                try:
                                    pyautogui.moveRel(dx_pix, dy_pix, duration=0)
                                except Exception:
                                    # Ако Windows/pyautogui блокира, нека не срине целия процес
                                    pass
                            last_move_time = now

                    # Blink -> click
                    if BLINK_DOUBLE_CLICK_ENABLED and pyautogui is not None:
                        left_ratio = eye_open_ratio(
                            face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
                        )
                        right_ratio = eye_open_ratio(
                            face_landmarks, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
                        )
                        blink_score = (left_ratio + right_ratio) / 2.0
                        last_blink_score = blink_score

                        # Ако сме "затворени" -> чакаме да се отворим (transition броим)
                        if blink_score < BLINK_RATIO_CLOSED:
                            if not blink_is_closed:
                                blink_is_closed = True
                                blink_closed_start_time = time.time()
                        else:
                            # transition: затворено -> отворено
                            if blink_is_closed and blink_score > BLINK_RATIO_OPEN:
                                closed_dur = time.time() - blink_closed_start_time
                                if closed_dur >= BLINK_CLOSED_MIN_SEC:
                                    now_blink = time.time()
                                    # Ако прозорецът се е минал -> ресет брояча
                                    if (
                                        blink_count > 0
                                        and (now_blink - first_blink_time) > DOUBLE_BLINK_WINDOW_SEC
                                    ):
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

                        # Визуализация (за настройка на праговете)
                        if blink_count > 0:
                            cv2.putText(
                                frame,
                                f"Blink score: {last_blink_score:.3f} count: {blink_count}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                            )
                        else:
                            cv2.putText(
                                frame,
                                f"Blink score: {last_blink_score:.3f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                            )

                    nose_x = int(nose_tip.x * w)
                    nose_y = int(nose_tip.y * h)

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

                # Начертай само връзките около носа (по-леко от full face mesh).
                draw_connections(
                    frame,
                    face_landmarks,
                    FaceLandmarksConnections.FACE_LANDMARKS_NOSE,
                    color=(0, 255, 0),
                    thickness=1,
                )
            else:
                # Ако няма лице - позволяваме ресет след загуба на детекция
                if mouse_enabled:
                    if last_seen_time is not None and (time.time() - last_seen_time) > RESET_AFTER_LOST_FACE_SEC:
                        smoothed_x = None
                        smoothed_y = None
                        prev_smoothed_x = None
                        prev_smoothed_y = None
                        neutral_yaw = None
                        neutral_pitch = None
                        smoothed_yaw = None
                        smoothed_pitch = None
                        stop_streak = 0
                if blink_is_closed:
                    blink_is_closed = False
                blink_count = 0
                first_blink_time = 0.0

            if SHOW_CAMERA_PREVIEW:
                cv2.imshow("Face Points", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    # Калибриране на "гледай напред" (неутралата) към текущата глава.
                    if smoothed_yaw is not None and smoothed_pitch is not None:
                        neutral_yaw = smoothed_yaw
                        neutral_pitch = smoothed_pitch
                        print("Neutral recalibrated.")
                if key == ord("m"):
                    mouse_enabled = not mouse_enabled
                    if mouse_enabled:
                        smoothed_x = None
                        smoothed_y = None
                        prev_smoothed_x = None
                        prev_smoothed_y = None
                        last_seen_time = None
                        neutral_yaw = None
                        neutral_pitch = None
                        smoothed_yaw = None
                        smoothed_pitch = None
                        stop_streak = 0
                    print(f"Mouse control: {mouse_enabled}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
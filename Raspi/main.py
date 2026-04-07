import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEAD_TRACKING_PATH = os.path.join(BASE_DIR, "head_tracking_opencv.py")
GUI_PATH = os.path.join(BASE_DIR, "head_control_gui.py")


def main():
    env = dict(os.environ)
    env["HEAD_TRACKING_NO_GUI"] = "1"

    tracking_proc = subprocess.Popen(
        [sys.executable, HEAD_TRACKING_PATH],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    try:
        subprocess.run(
            [sys.executable, GUI_PATH, "--gui-only"],
            cwd=BASE_DIR,
            env=env,
            check=False,
        )
    finally:
        try:
            if tracking_proc.poll() is None:
                tracking_proc.terminate()
                tracking_proc.wait(timeout=3)
        except Exception:
            try:
                tracking_proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
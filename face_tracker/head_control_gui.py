import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "head_tracking.py")
CMD_PATH = os.path.join(BASE_DIR, "head_tracking_cmd.txt")

GUI_ONLY = ("--gui-only" in sys.argv) or ("--no-start" in sys.argv)


def send_cmd(cmd: str) -> None:
    tmp_path = CMD_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(cmd)
    os.replace(tmp_path, CMD_PATH)


class ActionTile(tk.Frame):
    def __init__(self, parent, title, subtitle, bg_color, command):
        super().__init__(
            parent,
            bg=bg_color,
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )
        self.command = command
        self.bg_color = bg_color
        self.hover_color = self._darken(bg_color, 0.90)

        self.configure(width=180, height=120)
        self.pack_propagate(False)

        self.inner = tk.Frame(self, bg=bg_color, bd=0, highlightthickness=0)
        self.inner.pack(fill="both", expand=True)

        self.title_label = tk.Label(
            self.inner,
            text=title,
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg=bg_color
        )
        self.title_label.pack(anchor="w", padx=16, pady=(18, 2))

        self.subtitle_label = tk.Label(
            self.inner,
            text=subtitle,
            font=("Segoe UI", 10),
            fg="#e8edf7",
            bg=bg_color,
            justify="left",
            wraplength=145
        )
        self.subtitle_label.pack(anchor="w", padx=16)

        self.bind("<Button-1>", self._on_click)
        self.inner.bind("<Button-1>", self._on_click)
        self.title_label.bind("<Button-1>", self._on_click)
        self.subtitle_label.bind("<Button-1>", self._on_click)

        self.bind("<Enter>", self._on_enter)
        self.inner.bind("<Enter>", self._on_enter)
        self.title_label.bind("<Enter>", self._on_enter)
        self.subtitle_label.bind("<Enter>", self._on_enter)

        self.bind("<Leave>", self._on_leave)
        self.inner.bind("<Leave>", self._on_leave)
        self.title_label.bind("<Leave>", self._on_leave)
        self.subtitle_label.bind("<Leave>", self._on_leave)

    def _darken(self, hex_color, factor):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _set_bg(self, color):
        self.configure(bg=color)
        self.inner.configure(bg=color)
        self.title_label.configure(bg=color)
        self.subtitle_label.configure(bg=color)

    def _on_enter(self, event):
        self._set_bg(self.hover_color)

    def _on_leave(self, event):
        self._set_bg(self.bg_color)

    def _on_click(self, event):
        if self.command:
            self.command()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Assistive Control Panel")
        self.geometry("860x560")
        self.minsize(860, 560)
        self.resizable(False, False)
        self.configure(bg="#0b1020")

        self.tracking_proc = None
        self.launch_tracking = not GUI_ONLY

        self.status_var = tk.StringVar(value="System idle")
        self.mode_var = tk.StringVar(value="Waiting")

        self.build_ui()

        if self.launch_tracking:
            self.after(100, self.start_tracking)
        else:
            self.set_status("GUI only mode")
            self.set_mode("head_tracking.py трябва да е пуснат ръчно")

    def build_ui(self):
        # Main container
        main = tk.Frame(self, bg="#0b1020")
        main.pack(fill="both", expand=True, padx=24, pady=24)

        # Header
        header = tk.Frame(main, bg="#0b1020")
        header.pack(fill="x", pady=(0, 18))

        tk.Label(
            header,
            text="Assistive Room Control",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#0b1020"
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Управление на устройства и помощни функции чрез достъпен интерфейс",
            font=("Segoe UI", 11),
            fg="#98a2b3",
            bg="#0b1020"
        ).pack(anchor="w", pady=(4, 0))

        # Top status bar
        status_card = tk.Frame(main, bg="#121a2e", bd=0, highlightthickness=0)
        status_card.pack(fill="x", pady=(0, 20))

        left_status = tk.Frame(status_card, bg="#121a2e")
        left_status.pack(side="left", padx=18, pady=14)

        tk.Label(
            left_status,
            text="STATUS",
            font=("Segoe UI", 9, "bold"),
            fg="#8b95a7",
            bg="#121a2e"
        ).pack(anchor="w")

        self.status_label = tk.Label(
            left_status,
            textvariable=self.status_var,
            font=("Segoe UI", 14, "bold"),
            fg="#7ee787",
            bg="#121a2e"
        )
        self.status_label.pack(anchor="w", pady=(2, 0))

        right_status = tk.Frame(status_card, bg="#121a2e")
        right_status.pack(side="right", padx=18, pady=14)

        tk.Label(
            right_status,
            text="MODE",
            font=("Segoe UI", 9, "bold"),
            fg="#8b95a7",
            bg="#121a2e"
        ).pack(anchor="e")

        self.mode_label = tk.Label(
            right_status,
            textvariable=self.mode_var,
            font=("Segoe UI", 14, "bold"),
            fg="#8ab4ff",
            bg="#121a2e"
        )
        self.mode_label.pack(anchor="e", pady=(2, 0))

        # Grid area
        grid_wrap = tk.Frame(main, bg="#0b1020")
        grid_wrap.pack(fill="both", expand=True)

        grid_wrap.grid_columnconfigure(0, weight=1)
        grid_wrap.grid_columnconfigure(1, weight=1)
        grid_wrap.grid_rowconfigure(0, weight=1)
        grid_wrap.grid_rowconfigure(1, weight=1)

        tile1 = ActionTile(
            grid_wrap,
            title="Климатик",
            subtitle="Включване и управление на климатика",
            bg_color="#2563eb",
            command=self.on_climate
        )
        tile1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        tile2 = ActionTile(
            grid_wrap,
            title="Лампа",
            subtitle="Включване и изключване на осветлението",
            bg_color="#f59e0b",
            command=self.on_light
        )
        tile2.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        tile3 = ActionTile(
            grid_wrap,
            title="Клавиатура",
            subtitle="Отваряне на екранна клавиатура за писане",
            bg_color="#10b981",
            command=self.on_keyboard
        )
        tile3.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        tile4 = ActionTile(
            grid_wrap,
            title="Помощ",
            subtitle="Извикване на помощ или авариен сигнал",
            bg_color="#ef4444",
            command=self.on_help
        )
        tile4.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        # Bottom controls
        bottom = tk.Frame(main, bg="#0b1020")
        bottom.pack(fill="x", pady=(18, 0))

        controls = tk.Frame(bottom, bg="#0b1020")
        controls.pack(side="left")

        self.make_small_button(controls, "Toggle Mouse", "#1f2937", self.on_toggle).pack(side="left", padx=(0, 10))
        self.make_small_button(controls, "Calibrate", "#374151", self.on_calibrate).pack(side="left")

        self.make_small_button(bottom, "Exit", "#7f1d1d", self.on_exit).pack(side="right")

    def make_small_button(self, parent, text, bg, command):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg=bg,
            activebackground=bg,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2"
        )
        return btn

    def set_status(self, text, color="#7ee787"):
        self.status_var.set(text)
        self.status_label.configure(fg=color)

    def set_mode(self, text, color="#8ab4ff"):
        self.mode_var.set(text)
        self.mode_label.configure(fg=color)

    def start_tracking(self):
        if self.tracking_proc is not None:
            return

        if not os.path.exists(SCRIPT_PATH):
            self.set_status("head_tracking.py not found", "#ff6b6b")
            self.set_mode("Startup failed", "#ff6b6b")
            return

        try:
            env = dict(os.environ)
            env["HEAD_TRACKING_NO_GUI"] = "1"

            self.tracking_proc = subprocess.Popen(
                [sys.executable, SCRIPT_PATH],
                cwd=BASE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            self.set_status("Tracking active", "#7ee787")
            self.set_mode("Connected", "#8ab4ff")

        except Exception as e:
            self.set_status(f"Failed to start: {e}", "#ff6b6b")
            self.set_mode("Error", "#ff6b6b")

    def on_toggle(self):
        try:
            send_cmd("toggle")
            self.set_status("Mouse toggle sent", "#8ab4ff")
        except Exception as e:
            self.set_status(f"Toggle failed: {e}", "#ff6b6b")

    def on_calibrate(self):
        try:
            send_cmd("calibrate")
            self.set_status("Calibration command sent", "#fbbf24")
        except Exception as e:
            self.set_status(f"Calibration failed: {e}", "#ff6b6b")

    def on_climate(self):
        try:
            send_cmd("climate")
            self.set_status("Климатик: команда изпратена", "#60a5fa")
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_light(self):
        try:
            send_cmd("light")
            self.set_status("Лампа: команда изпратена", "#fbbf24")
        except Exception as e:
            self.set_status(f"Light failed: {e}", "#ff6b6b")

    def on_keyboard(self):
        try:
            send_cmd("keyboard")
            self.set_status("Клавиатура: команда изпратена", "#34d399")
        except Exception as e:
            self.set_status(f"Keyboard failed: {e}", "#ff6b6b")

    def on_help(self):
        try:
            send_cmd("help")
            self.set_status("ПОМОЩ: команда изпратена", "#f87171")
            self.set_mode("Emergency / Help", "#f87171")
        except Exception as e:
            self.set_status(f"Help failed: {e}", "#ff6b6b")

    def on_exit(self):
        try:
            send_cmd("exit")
            self.set_status("Stopping system...", "#fbbf24")
            self.set_mode("Disconnecting", "#fbbf24")
        except Exception:
            pass

        self.after(500, self.try_close_proc)

    def try_close_proc(self):
        if self.tracking_proc is None:
            self.destroy()
            return

        try:
            if self.tracking_proc.poll() is None:
                self.tracking_proc.terminate()
        finally:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
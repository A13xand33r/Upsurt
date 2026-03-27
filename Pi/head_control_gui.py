import os
import subprocess
import sys
import time
import tkinter as tk

from light_menu_view import ClimateMenuView, KeyboardMenuView, LightMenuView

try:
    import requests
except Exception:
    requests = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "head_tracking_opencv.py")
CMD_PATH = os.path.join(BASE_DIR, "head_tracking_cmd.txt")
BOUNDS_PATH = os.path.join(BASE_DIR, "head_tracking_bounds.txt")
STATUS_PATH = os.path.join(BASE_DIR, "head_tracking_status.txt")
CLIMATE_TEMP_PATH = os.path.join(BASE_DIR, "climate_temp.txt")
CLIMATE_MODE_PATH = os.path.join(BASE_DIR, "climate_mode.txt")
ESP32_URL = "http://esp32.local/control"
STATUS_POLL_INTERVAL_MS = 40

GUI_ONLY = ("--gui-only" in sys.argv) or ("--no-start" in sys.argv)


def send_cmd(cmd: str) -> None:
    tmp_path = CMD_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(cmd)
    os.replace(tmp_path, CMD_PATH)


def send_esp_command(device: str, action: str) -> None:
    if requests is None:
        raise RuntimeError("requests is not available")
    payload = {"device": device, "action": action}
    requests.post(ESP32_URL, json=payload, timeout=1)


def ui_font(size, weight="normal"):
    return ("DejaVu Sans", size, weight)


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
            font=ui_font(18, "bold"),
            fg="white",
            bg=bg_color
        )
        self.title_label.pack(anchor="w", padx=16, pady=(18, 2))

        self.subtitle_label = tk.Label(
            self.inner,
            text=subtitle,
            font=ui_font(10),
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
        self.resizable(True, True)
        self.configure(bg="#0b1020")
        self.bind("<Escape>", self._exit_fullscreen)
        self.bind("<Configure>", self._on_configure)

        self.tracking_proc = None
        self.launch_tracking = not GUI_ONLY

        self.status_var = tk.StringVar(value="System idle")
        self.mode_var = tk.StringVar(value="Waiting")
        self._last_status_event = ""
        self._status_read_pos = 0
        self._suppress_clicks_until = 0.0
        self.click_targets = []
        self.main_click_targets = []
        self.light_click_targets = []
        self.fan_click_targets = []
        self.light_view = None
        self.climate_view = None
        self.keyboard_view = None
        self.fan_view = None
        self.current_menu = "main"
        self.light_is_on = False
        self.climate_is_on = False
        self.climate_is_warm = True
        self.climate_temp_value = 24
        self.climate_led2_brightness = 255
        self.keyboard_is_open = False
        self.tv_led1_brightness = 255
        self.tv_blink_interval_sec = 0.8
        self.fan_is_on = False
        self.btn_exit = None
        self.header_frame = None
        self.status_card = None
        self.bottom_frame = None
        self.status_label = None
        self.mode_label = None

        self.build_ui()
        self.after(200, self.publish_bounds)
        self.after(STATUS_POLL_INTERVAL_MS, self.poll_tracking_status)
        self.after(500, self.poll_climate_state)

        if self.launch_tracking:
            self.after(100, self.start_tracking)
        else:
            self.set_status("GUI only mode")
            self.set_mode("head_tracking.py is started by main.py")

    def _exit_fullscreen(self, _event=None):
        self.attributes("-fullscreen", False)
        self.publish_bounds()

    def _on_configure(self, _event=None):
        # Keep head-mouse clamp aligned with whichever window is currently visible.
        self.after(30, self.publish_bounds)
        if self.current_menu == "light" and self.light_view is not None and self.light_view.winfo_ismapped():
            self.after(30, self.publish_bounds, self.light_view)
        elif self.current_menu == "climate" and self.climate_view is not None and self.climate_view.winfo_ismapped():
            self.after(30, self.publish_bounds, self.climate_view)
        elif self.current_menu == "keyboard" and self.keyboard_view is not None and self.keyboard_view.winfo_ismapped():
            self.after(30, self.publish_bounds, self.keyboard_view)
        elif self.current_menu == "fan" and self.fan_view is not None and self.fan_view.winfo_ismapped():
            self.after(30, self.publish_bounds, self.fan_view)

    def publish_bounds(self, target_widget=None):
        try:
            self.update_idletasks()
            widget = target_widget or self
            x = int(widget.winfo_rootx())
            y = int(widget.winfo_rooty())
            w = int(widget.winfo_width())
            h = int(widget.winfo_height())
            if w <= 1 or h <= 1:
                return
            tmp_path = BOUNDS_PATH + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(f"{x},{y},{w},{h}\n")
            os.replace(tmp_path, BOUNDS_PATH)
        except Exception:
            pass

    def poll_tracking_status(self):
        try:
            if os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, "r", encoding="utf-8") as f:
                    f.seek(self._status_read_pos)
                    new_lines = f.readlines()
                    self._status_read_pos = f.tell()

                for raw in new_lines:
                    raw = raw.strip()
                    if not raw:
                        continue
                    self._last_status_event = raw
                    parts = raw.split("|", 1)
                    event = parts[1] if len(parts) == 2 else raw
                    if event == "blink_click":
                        if time.time() >= float(self._suppress_clicks_until or 0.0):
                            self.click_widget_under_cursor()
                        self.set_mode("Blink click detected", "#34d399")
                    elif event == "blink_1":
                        self.set_mode("Blink 1/2 detected", "#fbbf24")
                    elif event == "blink_2":
                        # Показваме индикация, но кликът се прави само на blink_click.
                        self.set_mode("Blink 2/2 detected", "#34d399")
        except Exception:
            pass
        self.after(STATUS_POLL_INTERVAL_MS, self.poll_tracking_status)

    def suppress_blink_clicks(self, seconds: float = 2.0) -> None:
        try:
            self._suppress_clicks_until = time.time() + float(seconds)
        except Exception:
            self._suppress_clicks_until = time.time() + 2.0

    def poll_climate_state(self):
        """
        Optionally reads temperature + mode from files written by an external climate controller:
          - `climate_temp.txt` : e.g. "24" or "24.5"
          - `climate_mode.txt` : "warm" / "cold" (or 1/0)
        If files do not exist, keeps values updated from button presses.
        """
        new_temp = self.climate_temp_value
        new_is_warm = self.climate_is_warm

        try:
            if os.path.exists(CLIMATE_TEMP_PATH):
                with open(CLIMATE_TEMP_PATH, "r", encoding="utf-8") as f:
                    raw = f.read().strip().replace(",", ".")
                if raw:
                    new_temp = float(raw)
        except Exception:
            pass

        try:
            if os.path.exists(CLIMATE_MODE_PATH):
                with open(CLIMATE_MODE_PATH, "r", encoding="utf-8") as f:
                    raw = f.read().strip().lower()
                if raw:
                    if raw in ("warm", "hot", "toplo", "1", "true", "t"):
                        new_is_warm = True
                    elif raw in ("cold", "cool", "studen", "0", "false", "f"):
                        new_is_warm = False
        except Exception:
            pass

        # Keep internal state in sync, and update UI if climate page is visible.
        temp_changed = new_temp != self.climate_temp_value
        mode_changed = new_is_warm != self.climate_is_warm
        self.climate_temp_value = new_temp
        self.climate_is_warm = new_is_warm

        if (
            self.current_menu == "climate"
            and self.climate_view is not None
            and self.climate_view.winfo_ismapped()
            and temp_changed
        ):
            try:
                self.climate_view.set_temperature(self.climate_temp_value)
            except Exception:
                pass

        self.after(500, self.poll_climate_state)

    def click_widget_under_cursor(self):
        try:
            px, py = self.winfo_pointerxy()
            for target in self.click_targets:
                if not target.winfo_ismapped():
                    continue
                x1 = target.winfo_rootx()
                y1 = target.winfo_rooty()
                x2 = x1 + target.winfo_width()
                y2 = y1 + target.winfo_height()
                if x1 <= px < x2 and y1 <= py < y2:
                    if isinstance(target, tk.Button):
                        target.invoke()
                    elif hasattr(target, "command") and callable(target.command):
                        target.command()
                    return

            widget = self.winfo_containing(px, py)
            if widget is None:
                return
            rx = max(0, px - widget.winfo_rootx())
            ry = max(0, py - widget.winfo_rooty())
            widget.event_generate("<Button-1>", x=rx, y=ry)
            widget.event_generate("<ButtonRelease-1>", x=rx, y=ry)
        except Exception:
            pass

    def show_main_menu(self):
        # Switch "page" back to main menu (same window, no new Toplevel).
        if self.light_view is not None:
            try:
                self.light_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.light_view.pack_forget()
            except Exception:
                pass
        if self.climate_view is not None:
            try:
                self.climate_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.climate_view.pack_forget()
            except Exception:
                pass
        if self.keyboard_view is not None:
            try:
                self.keyboard_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.keyboard_view.pack_forget()
            except Exception:
                pass
        if self.fan_view is not None:
            try:
                self.fan_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.fan_view.pack_forget()
            except Exception:
                pass
        if self.header_frame is not None and not self.header_frame.winfo_ismapped():
            try:
                self.header_frame.pack(fill="x", pady=(0, 18))
            except Exception:
                pass
        if self.main_menu_frame is not None and not self.main_menu_frame.winfo_ismapped():
            self.main_menu_frame.pack(fill="both", expand=True)
        # status_card and bottom_frame removed to maximize button space.

        self.click_targets = list(self.main_click_targets)
        self.current_menu = "main"
        if self.btn_exit is not None:
            self.btn_exit.configure(text="Exit")
        # Restore the same baseline labels as initial screen.
        if self.launch_tracking:
            self.set_status("Tracking active", "#7ee787")
            self.set_mode("Connected", "#8ab4ff")
        else:
            self.set_status("GUI only mode")
            self.set_mode("head_tracking.py is started by main.py")

    def show_light_menu(self):
        self.open_light_window()

    def show_climate_menu(self):
        self.open_climate_window()

    def show_keyboard_menu(self):
        self.open_keyboard_window()

    def show_fan_menu(self):
        self.open_fan_window()

    def open_light_window(self):
        # Switch "page" to light menu inside the same window.
        if self.light_view is None:
            return

        if self.header_frame is not None:
            try:
                self.header_frame.pack_forget()
            except Exception:
                pass
        if self.main_menu_frame is not None:
            try:
                self.main_menu_frame.pack_forget()
            except Exception:
                pass
        if self.status_card is not None:
            try:
                self.status_card.pack_forget()
            except Exception:
                pass
        if self.bottom_frame is not None:
            try:
                self.bottom_frame.pack_forget()
            except Exception:
                pass

        # Hide other pages.
        if self.climate_view is not None:
            try:
                self.climate_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.climate_view.pack_forget()
            except Exception:
                pass
        if self.keyboard_view is not None:
            try:
                self.keyboard_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.keyboard_view.pack_forget()
            except Exception:
                pass
        if self.fan_view is not None:
            try:
                self.fan_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.fan_view.pack_forget()
            except Exception:
                pass
        if self.fan_view is not None:
            try:
                self.fan_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.fan_view.pack_forget()
            except Exception:
                pass

        self.light_view.set_light_state(self.light_is_on)
        self.light_view.pack(fill="both", expand=True)
        try:
            self.light_view.lift()
        except Exception:
            pass

        # Start in-menu cursor rendering (more reliable than overlay z-order).
        try:
            self.light_view.start_cursor_updates()
        except Exception:
            pass

        self.light_click_targets = list(self.light_view.click_targets)
        self.click_targets = list(self.light_click_targets)
        self.current_menu = "light"
        self.set_mode("Light menu", "#fbbf24")

        # Update head-clamp bounds to match the light page.
        self.publish_bounds(self.light_view)

    def close_light_window(self):
        # Return from light menu back to main menu (no new Toplevel).
        self.suppress_blink_clicks(2.0)
        try:
            if self.light_view is not None:
                self.light_view.stop_cursor_updates()
        except Exception:
            pass
        self.light_click_targets = []
        self.show_main_menu()
        self.publish_bounds()

    def open_climate_window(self):
        if self.climate_view is None:
            return

        if self.header_frame is not None:
            try:
                self.header_frame.pack_forget()
            except Exception:
                pass
        if self.main_menu_frame is not None:
            try:
                self.main_menu_frame.pack_forget()
            except Exception:
                pass
        if self.status_card is not None:
            try:
                self.status_card.pack_forget()
            except Exception:
                pass
        if self.bottom_frame is not None:
            try:
                self.bottom_frame.pack_forget()
            except Exception:
                pass

        # Hide other pages.
        if self.light_view is not None:
            try:
                self.light_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.light_view.pack_forget()
            except Exception:
                pass
        if self.keyboard_view is not None:
            try:
                self.keyboard_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.keyboard_view.pack_forget()
            except Exception:
                pass

        # Climate menu is stateless in UI; actions send commands directly.
        self.climate_view.pack(fill="both", expand=True)
        try:
            self.climate_view.lift()
        except Exception:
            pass

        # Sync UI labels with current remembered state.
        try:
            self.climate_view.set_power_state(self.climate_is_on)
        except Exception:
            pass
        try:
            self.climate_view.set_temperature(self.climate_temp_value)
        except Exception:
            pass

        try:
            self.climate_view.start_cursor_updates()
        except Exception:
            pass

        self.climate_click_targets = list(self.climate_view.click_targets)
        self.click_targets = list(self.climate_click_targets)
        self.current_menu = "climate"
        self.set_mode("Climate menu", "#60a5fa")
        self.set_status("Климатик: готов")
        self.publish_bounds(self.climate_view)

    def close_climate_window(self):
        self.suppress_blink_clicks(2.0)
        try:
            if self.climate_view is not None:
                self.climate_view.stop_cursor_updates()
        except Exception:
            pass
        self.show_main_menu()
        self.publish_bounds()

    def open_keyboard_window(self):
        if self.keyboard_view is None:
            return

        if self.header_frame is not None:
            try:
                self.header_frame.pack_forget()
            except Exception:
                pass
        if self.main_menu_frame is not None:
            try:
                self.main_menu_frame.pack_forget()
            except Exception:
                pass
        if self.status_card is not None:
            try:
                self.status_card.pack_forget()
            except Exception:
                pass
        if self.bottom_frame is not None:
            try:
                self.bottom_frame.pack_forget()
            except Exception:
                pass

        # Hide other pages.
        if self.light_view is not None:
            try:
                self.light_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.light_view.pack_forget()
            except Exception:
                pass
        if self.climate_view is not None:
            try:
                self.climate_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.climate_view.pack_forget()
            except Exception:
                pass
        if self.fan_view is not None:
            try:
                self.fan_view.stop_cursor_updates()
            except Exception:
                pass
            try:
                self.fan_view.pack_forget()
            except Exception:
                pass

        self.keyboard_view.set_power_state(self.keyboard_is_open)
        self.keyboard_view.pack(fill="both", expand=True)
        try:
            self.keyboard_view.lift()
        except Exception:
            pass

        try:
            self.keyboard_view.start_cursor_updates()
        except Exception:
            pass

        self.keyboard_click_targets = list(self.keyboard_view.click_targets)
        self.click_targets = list(self.keyboard_click_targets)
        self.current_menu = "keyboard"
        self.set_mode("TV menu", "#34d399")
        self.set_status("Телевизор: готов")
        self.publish_bounds(self.keyboard_view)

    def close_keyboard_window(self):
        self.suppress_blink_clicks(2.0)
        try:
            if self.keyboard_view is not None:
                self.keyboard_view.stop_cursor_updates()
        except Exception:
            pass
        self.show_main_menu()
        self.publish_bounds()

    def open_fan_window(self):
        if self.fan_view is None:
            return

        if self.header_frame is not None:
            try:
                self.header_frame.pack_forget()
            except Exception:
                pass
        if self.main_menu_frame is not None:
            try:
                self.main_menu_frame.pack_forget()
            except Exception:
                pass
        if self.status_card is not None:
            try:
                self.status_card.pack_forget()
            except Exception:
                pass
        if self.bottom_frame is not None:
            try:
                self.bottom_frame.pack_forget()
            except Exception:
                pass

        for view in (self.light_view, self.climate_view, self.keyboard_view):
            if view is None:
                continue
            try:
                view.stop_cursor_updates()
            except Exception:
                pass
            try:
                view.pack_forget()
            except Exception:
                pass

        self.fan_view.set_power_state(self.fan_is_on)
        self.fan_view.pack(fill="both", expand=True)
        try:
            self.fan_view.lift()
        except Exception:
            pass

        try:
            self.fan_view.start_cursor_updates()
        except Exception:
            pass

        self.fan_click_targets = list(self.fan_view.click_targets)
        self.click_targets = list(self.fan_click_targets)
        self.current_menu = "fan"
        self.set_mode("Fan menu", "#22c55e")
        self.set_status("Вентилатор: готов")
        self.publish_bounds(self.fan_view)

    def close_fan_window(self):
        self.suppress_blink_clicks(2.0)
        try:
            if self.fan_view is not None:
                self.fan_view.stop_cursor_updates()
        except Exception:
            pass
        self.show_main_menu()
        self.publish_bounds()

    def update_light_status_label(self):
        state_text = "включена" if self.light_is_on else "изключена"
        state_color = "#fbbf24" if self.light_is_on else "#9ca3af"
        self.set_status(f"Лампа (LED3): {state_text}", state_color)
        if self.light_view is not None:
            self.light_view.set_light_state(self.light_is_on)

    def build_ui(self):
        main = tk.Frame(self, bg="#0b1020")
        main.pack(fill="both", expand=True, padx=24, pady=24)

        self.header_frame = tk.Frame(main, bg="#0b1020")
        self.header_frame.pack(fill="x", pady=(0, 18))

        tk.Label(
            self.header_frame,
            text="Assistive Room Control",
            font=ui_font(24, "bold"),
            fg="white",
            bg="#0b1020"
        ).pack(anchor="w")

        tk.Label(
            self.header_frame,
            text="Управление на устройства и помощни функции чрез достъпен интерфейс",
            font=ui_font(11),
            fg="#98a2b3",
            bg="#0b1020"
        ).pack(anchor="w", pady=(4, 0))

        self.main_menu_frame = tk.Frame(main, bg="#0b1020")
        self.main_menu_frame.pack(fill="both", expand=True)

        self.main_menu_frame.grid_columnconfigure(0, weight=1)
        self.main_menu_frame.grid_columnconfigure(1, weight=1)
        self.main_menu_frame.grid_rowconfigure(0, weight=1)
        self.main_menu_frame.grid_rowconfigure(1, weight=1)
        self.main_menu_frame.grid_rowconfigure(2, weight=1)

        tile1 = ActionTile(
            self.main_menu_frame,
            title="Климатик",
            subtitle="Включване и управление на климатика",
            bg_color="#2563eb",
            command=self.on_climate
        )
        tile1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_click_targets.append(tile1)

        tile2 = ActionTile(
            self.main_menu_frame,
            title="Лампа",
            subtitle="Включване и изключване на осветлението",
            bg_color="#f59e0b",
            command=self.on_light
        )
        tile2.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_click_targets.append(tile2)

        tile3 = ActionTile(
            self.main_menu_frame,
            title="Телевизор",
            subtitle="Включване и изключване на телевизора (LED1)",
            bg_color="#10b981",
            command=self.on_keyboard
        )
        tile3.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.main_click_targets.append(tile3)

        tile4 = ActionTile(
            self.main_menu_frame,
            title="Вентилатор",
            subtitle="Включване и изключване на вентилатор (Motor)",
            bg_color="#14b8a6",
            command=self.on_fan,
        )
        tile4.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.main_click_targets.append(tile4)

        tile5 = ActionTile(
            self.main_menu_frame,
            title="Помощ",
            subtitle="Извикване на помощ или авариен сигнал",
            bg_color="#ef4444",
            command=self.on_help
        )
        tile5.grid(row=2, column=1, sticky="nsew", padx=10, pady=10)
        self.main_click_targets.append(tile5)

        exit_tile = ActionTile(
            self.main_menu_frame,
            title="Изход",
            subtitle="Затвори приложението",
            bg_color="#7f1d1d",
            command=self.on_exit,
        )
        exit_tile.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.main_click_targets.append(exit_tile)

        # Light menu page (inside same window, no Toplevel).
        self.light_view = LightMenuView(
            main,
            on_light_on=self.on_light_on,
            on_light_off=self.on_light_off,
            on_back=self.close_light_window,
        )
        self.light_view.pack(fill="both", expand=True)
        self.light_view.pack_forget()
        self.light_click_targets = list(self.light_view.click_targets)

        # Climate menu page (inside same window, no Toplevel).
        self.climate_view = ClimateMenuView(
            main,
            on_power_on=self.on_climate_power_on,
            on_power_off=self.on_climate_power_off,
            on_temp_up=self.on_climate_temp_up,
            on_temp_down=self.on_climate_temp_down,
            on_back=self.close_climate_window,
        )
        self.climate_view.pack(fill="both", expand=True)
        self.climate_view.pack_forget()

        # Keyboard menu page (inside same window, no Toplevel).
        self.keyboard_view = KeyboardMenuView(
            main,
            on_power_on=self.on_tv_on,
            on_power_off=self.on_tv_off,
            on_back=self.close_keyboard_window,
            on_channel_up=self.on_tv_channel_up,
            on_channel_down=self.on_tv_channel_down,
            on_volume_up=self.on_tv_volume_up,
            on_volume_down=self.on_tv_volume_down,
            menu_title="Телевизор",
            state_off_text="изключен",
        )
        self.keyboard_view.pack(fill="both", expand=True)
        self.keyboard_view.pack_forget()

        self.fan_view = KeyboardMenuView(
            main,
            on_power_on=self.on_fan_on,
            on_power_off=self.on_fan_off,
            on_back=self.close_fan_window,
            menu_title="Вентилатор",
            state_off_text="изключен",
        )
        self.fan_view.pack(fill="both", expand=True)
        self.fan_view.pack_forget()

        self.show_main_menu()

    def make_small_button(self, parent, text, bg, command):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=ui_font(11, "bold"),
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
        if self.status_label is not None:
            try:
                self.status_label.configure(fg=color)
            except Exception:
                pass

    def set_mode(self, text, color="#8ab4ff"):
        self.mode_var.set(text)
        if self.mode_label is not None:
            try:
                self.mode_label.configure(fg=color)
            except Exception:
                pass

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

    def on_climate_temp_up(self):
        try:
            send_cmd("climate_temp_up")
            self.climate_temp_value = min(30, self.climate_temp_value + 1)
            self.climate_led2_brightness = max(0, self.climate_led2_brightness + 50)
            if self.climate_view is not None:
                self.climate_view.set_temperature(self.climate_temp_value)
            self.set_status(
                f"Климатик: {self.climate_temp_value} C | LED2 brightness {self.climate_led2_brightness}",
                "#60a5fa",
            )
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_climate_temp_down(self):
        try:
            send_cmd("climate_temp_down")
            self.climate_temp_value = max(10, self.climate_temp_value - 1)
            self.climate_led2_brightness = max(0, self.climate_led2_brightness - 50)
            if self.climate_view is not None:
                self.climate_view.set_temperature(self.climate_temp_value)
            self.set_status(
                f"Климатик: {self.climate_temp_value} C | LED2 brightness {self.climate_led2_brightness}",
                "#60a5fa",
            )
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_climate_warm_air(self):
        try:
            send_cmd("climate_warm_air")
            self.climate_is_warm = True
            self.set_status("Климатик: Духа топло", "#60a5fa")
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_climate_cold_air(self):
        try:
            send_cmd("climate_cold_air")
            self.climate_is_warm = False
            self.set_status("Климатик: Духа студено", "#60a5fa")
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_climate_power_on(self):
        try:
            send_esp_command("led2", "on")
            self.climate_is_on = True
            try:
                if self.climate_view is not None:
                    self.climate_view.set_power_state(self.climate_is_on)
            except Exception:
                pass
            self.set_status("Климатик: LED2 включен", "#60a5fa")
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_climate_power_off(self):
        try:
            send_esp_command("led2", "off")
            self.climate_is_on = False
            try:
                if self.climate_view is not None:
                    self.climate_view.set_power_state(self.climate_is_on)
            except Exception:
                pass
            self.set_status("Климатик: LED2 изключен", "#60a5fa")
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def _on_climate_cmd(self):
        try:
            send_cmd("climate")
            self.set_status("Климатик: команда изпратена", "#60a5fa")
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_light(self):
        self.show_light_menu()

    def on_climate(self):
        self.show_climate_menu()

    def on_keyboard(self):
        self.show_keyboard_menu()

    def on_fan(self):
        self.show_fan_menu()

    def on_light_on(self):
        try:
            send_esp_command("led3", "on")
            self.light_is_on = True
            self.update_light_status_label()
        except Exception as e:
            self.set_status(f"Light failed: {e}", "#ff6b6b")

    def on_light_off(self):
        try:
            send_esp_command("led3", "off")
            self.light_is_on = False
            self.update_light_status_label()
        except Exception as e:
            self.set_status(f"Light failed: {e}", "#ff6b6b")

    def on_climate_on(self):
        try:
            send_cmd("climate")
            self.climate_is_on = True
            self.open_climate_window()
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_climate_off(self):
        try:
            # Нямаме отделна команда climate_off в tracking.
            send_cmd("climate")
            self.climate_is_on = False
            self.open_climate_window()
        except Exception as e:
            self.set_status(f"Climate failed: {e}", "#ff6b6b")

    def on_tv_on(self):
        try:
            send_cmd("tv_power_on")
            send_esp_command("led1", "on")
            self.keyboard_is_open = True
            self.open_keyboard_window()
        except Exception as e:
            self.set_status(f"TV failed: {e}", "#ff6b6b")

    def on_tv_off(self):
        try:
            # Ensure blinking loop (in tracking process) stops and cannot turn it back on.
            send_cmd("tv_power_off")
            send_esp_command("led1", "off")
            self.keyboard_is_open = False
            self.open_keyboard_window()
        except Exception as e:
            self.set_status(f"TV failed: {e}", "#ff6b6b")

    def on_tv_volume_up(self):
        try:
            send_cmd("tv_volume_up")
            self.tv_led1_brightness = max(0, self.tv_led1_brightness + 50)
            self.set_status(
                f"Телевизор: сила + | brightness {self.tv_led1_brightness}",
                "#60a5fa",
            )
        except Exception as e:
            self.set_status(f"TV failed: {e}", "#ff6b6b")

    def on_tv_volume_down(self):
        try:
            send_cmd("tv_volume_down")
            self.tv_led1_brightness = max(0, self.tv_led1_brightness - 50)
            self.set_status(
                f"Телевизор: сила - | brightness {self.tv_led1_brightness}",
                "#60a5fa",
            )
        except Exception as e:
            self.set_status(f"TV failed: {e}", "#ff6b6b")

    def on_tv_channel_up(self):
        try:
            send_cmd("tv_channel_up")
            self.tv_blink_interval_sec = max(0.1, self.tv_blink_interval_sec - 0.1)
            self.set_status(
                f"Телевизор: канал + | мигане {self.tv_blink_interval_sec:.1f}s",
                "#60a5fa",
            )
        except Exception as e:
            self.set_status(f"TV failed: {e}", "#ff6b6b")

    def on_tv_channel_down(self):
        try:
            send_cmd("tv_channel_down")
            self.tv_blink_interval_sec = min(1.5, self.tv_blink_interval_sec + 0.1)
            self.set_status(
                f"Телевизор: канал - | мигане {self.tv_blink_interval_sec:.1f}s",
                "#60a5fa",
            )
        except Exception as e:
            self.set_status(f"TV failed: {e}", "#ff6b6b")

    def on_fan_on(self):
        try:
            send_esp_command("motor", "on")
            self.fan_is_on = True
            self.open_fan_window()
        except Exception as e:
            self.set_status(f"Fan failed: {e}", "#ff6b6b")

    def on_fan_off(self):
        try:
            send_esp_command("motor", "off")
            self.fan_is_on = False
            self.open_fan_window()
        except Exception as e:
            self.set_status(f"Fan failed: {e}", "#ff6b6b")

    def on_help(self):
        try:
            send_cmd("help")
            self.set_status("ПОМОЩ: команда изпратена", "#f87171")
            self.set_mode("Emergency / Help", "#f87171")
        except Exception as e:
            self.set_status(f"Help failed: {e}", "#ff6b6b")

    def on_exit(self):
        if self.current_menu == "light":
            self.close_light_window()
            return
        if self.current_menu == "climate":
            self.close_climate_window()
            return
        if self.current_menu == "keyboard":
            self.close_keyboard_window()
            return
        if self.current_menu == "fan":
            self.close_fan_window()
            return
        try:
            send_cmd("exit")
            self.set_status("Stopping system...", "#fbbf24")
            self.set_mode("Disconnecting", "#fbbf24")
        except Exception:
            pass

        self.after(500, self.try_close_proc)

    def try_close_proc(self):
        try:
            if os.path.exists(BOUNDS_PATH):
                os.remove(BOUNDS_PATH)
            if os.path.exists(STATUS_PATH):
                os.remove(STATUS_PATH)
        except Exception:
            pass
        if self.tracking_proc is not None:
            try:
                if self.tracking_proc.poll() is None:
                    self.tracking_proc.terminate()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
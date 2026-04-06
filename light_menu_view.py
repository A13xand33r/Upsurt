import tkinter as tk
import time


def ui_font(size, weight="normal"):
    return ("DejaVu Sans", size, weight)


class ActionTile(tk.Frame):
    def __init__(self, parent, title, subtitle, bg_color, command, *, width=None, height=None):
        super().__init__(
            parent,
            bg=bg_color,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.command = command
        self.bg_color = bg_color
        self.hover_color = self._darken(bg_color, 0.90)

        if width is not None or height is not None:
            cfg = {}
            if width is not None:
                cfg["width"] = int(width)
            if height is not None:
                cfg["height"] = int(height)
            self.configure(**cfg)
        self.pack_propagate(False)

        self.inner = tk.Frame(self, bg=bg_color, bd=0, highlightthickness=0)
        self.inner.pack(fill="both", expand=True)

        self.title_label = tk.Label(
            self.inner,
            text=title,
            font=ui_font(18, "bold"),
            fg="white",
            bg=bg_color,
        )
        self.title_label.pack(anchor="w", padx=16, pady=(18, 2))

        self.subtitle_label = tk.Label(
            self.inner,
            text=subtitle,
            font=ui_font(10),
            fg="#e8edf7",
            bg=bg_color,
            justify="left",
            wraplength=145,
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

    def set_base_bg(self, bg_color: str) -> None:
        """
        Променя "базовия" фон на плочката (за да може да показва on/off състояние).
        """
        self.bg_color = bg_color
        self.hover_color = self._darken(bg_color, 0.90)
        self._set_bg(bg_color)

    def _on_enter(self, _event):
        self._set_bg(self.hover_color)

    def _on_leave(self, _event):
        self._set_bg(self.bg_color)

    def _on_click(self, _event):
        if self.command:
            self.command()


class DwellClickController:
    def __init__(
        self,
        *,
        view: tk.Widget,
        click_targets: list[tk.Widget],
        dwell_ms: int = 2000,
        cooldown_ms: int = 600,
    ) -> None:
        self._view = view
        self._click_targets = click_targets
        self._dwell_ms = int(dwell_ms)
        self._cooldown_ms = int(cooldown_ms)

        self._current_target: tk.Widget | None = None
        self._entered_at: float | None = None
        self._fired_for_target: tk.Widget | None = None
        self._last_fire_at: float = 0.0

    def reset(self) -> None:
        self._current_target = None
        self._entered_at = None
        self._fired_for_target = None

    def update(self) -> None:
        if not self._view.winfo_exists():
            return

        try:
            px = self._view.winfo_pointerx()
            py = self._view.winfo_pointery()
        except Exception:
            self.reset()
            return

        # If pointer is outside this view, do nothing.
        try:
            root_x = self._view.winfo_rootx()
            root_y = self._view.winfo_rooty()
            w = self._view.winfo_width()
            h = self._view.winfo_height()
        except Exception:
            self.reset()
            return

        local_x = px - root_x
        local_y = py - root_y
        if not (0 <= local_x <= w and 0 <= local_y <= h):
            self.reset()
            return

        # IMPORTANT: we draw a cursor canvas under the pointer which can sit on top
        # of tiles; winfo_containing() would then "see" the cursor overlay instead
        # of the tile. Use geometry hit-testing against known click targets.
        target = self._find_target_by_geometry(px, py)
        now = time.monotonic()

        if target is None:
            self.reset()
            return

        if target is not self._current_target:
            self._current_target = target
            self._entered_at = now
            if self._fired_for_target is not None and self._fired_for_target is not target:
                self._fired_for_target = None

        if self._entered_at is None:
            self._entered_at = now

        # Prevent repeated firing while pointer stays on same tile.
        if self._fired_for_target is target:
            return

        # Global cooldown after any fire.
        if (now - self._last_fire_at) * 1000.0 < self._cooldown_ms:
            return

        if (now - self._entered_at) * 1000.0 >= self._dwell_ms:
            self._invoke_target(target)
            self._fired_for_target = target
            self._last_fire_at = now

    def _find_target_by_geometry(self, px: int, py: int) -> tk.Widget | None:
        # Check from last to first so later-added targets win if they overlap.
        for target in reversed(self._click_targets):
            try:
                if not target.winfo_exists() or not target.winfo_ismapped():
                    continue
                tx = target.winfo_rootx()
                ty = target.winfo_rooty()
                tw = target.winfo_width()
                th = target.winfo_height()
            except Exception:
                continue

            if tw <= 1 or th <= 1:
                continue

            if tx <= px <= (tx + tw) and ty <= py <= (ty + th):
                return target
        return None

    def _invoke_target(self, target: tk.Widget) -> None:
        cmd = getattr(target, "command", None)
        if callable(cmd):
            try:
                cmd()
            except Exception:
                pass
            return
        try:
            target.event_generate("<Button-1>")
        except Exception:
            pass


class LightMenuView(tk.Frame):
    def __init__(self, parent, on_light_on, on_light_off, on_back):
        super().__init__(parent, bg="#0b1020")
        self.click_targets = []
        self._cursor_job = None
        self._dwell = DwellClickController(view=self, click_targets=self.click_targets, dwell_ms=2000)

        # Визуален курсор, който се чертае ВНЪТРЕ в LightMenuView,
        # за да не зависи от custom cursor overlay (z-order / overrideredirect).
        self._cursor_size = 22
        self._cursor_pad = self._cursor_size // 2
        self._cursor_canvas = tk.Canvas(
            self,
            width=self._cursor_size,
            height=self._cursor_size,
            highlightthickness=0,
            bd=0,
            bg="#0b1020",
        )
        outer = self._cursor_canvas.create_oval(
            2,
            2,
            self._cursor_size - 2,
            self._cursor_size - 2,
            outline="#00E5FF",
            width=3,
        )
        inner = self._cursor_canvas.create_oval(
            self._cursor_size // 2 - 2,
            self._cursor_size // 2 - 2,
            self._cursor_size // 2 + 2,
            self._cursor_size // 2 + 2,
            fill="#00E5FF",
            outline="#00E5FF",
        )
        # Keep ref to avoid unused warnings in some linters.
        self._cursor_outer_id = outer
        self._cursor_inner_id = inner

        light_header = tk.Frame(self, bg="#0b1020")
        light_header.pack(fill="x", pady=(0, 12))
        tk.Label(
            light_header,
            text="Меню: Лампа",
            font=ui_font(22, "bold"),
            fg="white",
            bg="#0b1020",
        ).pack(anchor="w")
        self.state_label = tk.Label(
            light_header,
            text="Състояние: изключена",
            font=ui_font(11),
            fg="#9ca3af",
            bg="#0b1020",
        )
        self.state_label.pack(anchor="w", pady=(4, 0))

        light_actions = tk.Frame(self, bg="#0b1020")
        light_actions.pack(fill="both", expand=True)
        light_actions.grid_columnconfigure(0, weight=1)
        light_actions.grid_columnconfigure(1, weight=1)
        light_actions.grid_rowconfigure(0, weight=1)
        light_actions.grid_rowconfigure(1, weight=1)

        self.light_on_tile = ActionTile(
            light_actions,
            title="Включи",
            subtitle="Включи лампата",
            bg_color="#f59e0b",
            command=on_light_on,
        )
        self.light_on_tile.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.click_targets.append(self.light_on_tile)

        self.light_off_tile = ActionTile(
            light_actions,
            title="Изключи",
            subtitle="Изключи лампата",
            bg_color="#b45309",
            command=on_light_off,
        )
        self.light_off_tile.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.click_targets.append(self.light_off_tile)

        back_tile = ActionTile(
            light_actions,
            title="Назад",
            subtitle="Върни се към основното меню",
            bg_color="#374151",
            command=on_back,
        )
        back_tile.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.click_targets.append(back_tile)

    def set_light_state(self, is_on: bool):
        # Цветове за "активен" бутон (без да блокираме кликовете и на двата).
        ON_ACTIVE_BG = "#fbbf24"
        OFF_ACTIVE_BG = "#fb923c"
        INACTIVE_BG = "#92400e"

        if is_on:
            self.state_label.configure(text="Състояние: включена", fg="#fbbf24")
            self.light_on_tile.set_base_bg(ON_ACTIVE_BG)
            self.light_off_tile.set_base_bg(INACTIVE_BG)
        else:
            self.state_label.configure(text="Състояние: изключена", fg="#9ca3af")
            self.light_off_tile.set_base_bg(OFF_ACTIVE_BG)
            self.light_on_tile.set_base_bg(INACTIVE_BG)

    def start_cursor_updates(self, interval_ms: int = 30) -> None:
        # Start only once.
        if self._cursor_job is not None:
            return
        self._cursor_job = self.after(interval_ms, self._update_cursor)

    def stop_cursor_updates(self) -> None:
        if self._cursor_job is not None:
            try:
                self.after_cancel(self._cursor_job)
            except Exception:
                pass
            self._cursor_job = None
        try:
            self._dwell.reset()
        except Exception:
            pass
        try:
            self._cursor_canvas.place_forget()
        except Exception:
            pass

    def _update_cursor(self) -> None:
        # Keep loop running while menu exists.
        if not self.winfo_exists():
            self._cursor_job = None
            return

        # Current OS cursor position (screen coords).
        try:
            px = self.winfo_pointerx()
            py = self.winfo_pointery()
        except Exception:
            px, py = None, None

        if px is not None and py is not None:
            root_x = self.winfo_rootx()
            root_y = self.winfo_rooty()
            w = self.winfo_width()
            h = self.winfo_height()
            local_x = px - root_x
            local_y = py - root_y

            # Hide cursor when it leaves the window area.
            if 0 <= local_x <= w and 0 <= local_y <= h:
                # Center the cursor over the pointer.
                self._cursor_canvas.place(
                    x=int(local_x) - self._cursor_pad,
                    y=int(local_y) - self._cursor_pad,
                )
                try:
                    self._cursor_canvas.lift()
                except Exception:
                    pass
            else:
                try:
                    self._cursor_canvas.place_forget()
                except Exception:
                    pass

        try:
            self._dwell.update()
        except Exception:
            pass
        self._cursor_job = self.after(30, self._update_cursor)


class ClimateMenuView(tk.Frame):
    def __init__(
        self,
        parent,
        on_power_on,
        on_power_off,
        on_back,
        on_temp_up=None,
        on_temp_down=None,
    ):
        super().__init__(parent, bg="#0b1020")
        self.click_targets = []
        self._dwell = DwellClickController(view=self, click_targets=self.click_targets, dwell_ms=2000)

        self._cursor_job = None
        self._cursor_size = 22
        self._cursor_pad = self._cursor_size // 2
        self._cursor_canvas = tk.Canvas(
            self,
            width=self._cursor_size,
            height=self._cursor_size,
            highlightthickness=0,
            bd=0,
            bg="#0b1020",
        )
        self._cursor_canvas.place_forget()
        self._cursor_canvas.create_oval(
            2,
            2,
            self._cursor_size - 2,
            self._cursor_size - 2,
            outline="#00E5FF",
            width=3,
        )
        self._cursor_canvas.create_oval(
            self._cursor_size // 2 - 2,
            self._cursor_size // 2 - 2,
            self._cursor_size // 2 + 2,
            self._cursor_size // 2 + 2,
            fill="#00E5FF",
            outline="#00E5FF",
        )

        climate_header = tk.Frame(self, bg="#0b1020")
        climate_header.pack(fill="x", pady=(0, 12))
        tk.Label(
            climate_header,
            text="Меню: Климатик",
            font=ui_font(22, "bold"),
            fg="white",
            bg="#0b1020",
        ).pack(anchor="w")

        self.power_label = tk.Label(
            climate_header,
            text="Състояние: Изключен",
            font=ui_font(11, "bold"),
            fg="#f87171",
            bg="#0b1020",
        )
        self.power_label.pack(anchor="w", pady=(6, 0))
        self.temperature_label = tk.Label(
            climate_header,
            text="Температура: 24 C",
            font=ui_font(11),
            fg="#93c5fd",
            bg="#0b1020",
        )
        self.temperature_label.pack(anchor="w", pady=(4, 0))

        climate_actions = tk.Frame(self, bg="#0b1020")
        climate_actions.pack(fill="both", expand=True)

        # Compact layout: on/off + back.
        self._climate_center = tk.Frame(climate_actions, bg="#0b1020")
        # Place it centered; we will update its width on resize.
        self._climate_center.place(relx=0.5, rely=0.5, relheight=1.0, anchor="center")

        def _resize_center(_event=None):
            try:
                w = climate_actions.winfo_width()
            except Exception:
                return
            if w <= 1:
                return
            # Use most of the width but keep some margins.
            desired = int(w * 0.92)
            try:
                self._climate_center.place_configure(width=desired)
            except Exception:
                pass

        climate_actions.bind("<Configure>", _resize_center)
        _resize_center()

        self._climate_center.grid_columnconfigure(0, weight=1, uniform="climate_col")
        self._climate_center.grid_columnconfigure(1, weight=1, uniform="climate_col")
        self._climate_center.grid_rowconfigure(0, weight=1, uniform="climate_row")
        self._climate_center.grid_rowconfigure(1, weight=1, uniform="climate_row")
        self._climate_center.grid_rowconfigure(2, weight=1, uniform="climate_row")

        self.power_on_tile = ActionTile(
            self._climate_center,
            title="Включи",
            subtitle="Включи LED2",
            bg_color="#22c55e",
            command=on_power_on,
        )
        self.power_on_tile.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        self.click_targets.append(self.power_on_tile)

        self.power_off_tile = ActionTile(
            self._climate_center,
            title="Изключи",
            subtitle="Изключи LED2",
            bg_color="#ef4444",
            command=on_power_off,
        )
        self.power_off_tile.grid(row=0, column=1, sticky="nsew", padx=10, pady=8)
        self.click_targets.append(self.power_off_tile)

        self.temp_up_tile = ActionTile(
            self._climate_center,
            title="Температура +",
            subtitle="Увеличи LED2 brightness",
            bg_color="#2563eb",
            command=on_temp_up,
        )
        self.temp_up_tile.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        self.click_targets.append(self.temp_up_tile)

        self.temp_down_tile = ActionTile(
            self._climate_center,
            title="Температура -",
            subtitle="Намали LED2 brightness",
            bg_color="#1d4ed8",
            command=on_temp_down,
        )
        self.temp_down_tile.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        self.click_targets.append(self.temp_down_tile)

        back_tile = ActionTile(
            self._climate_center,
            title="Назад",
            subtitle="Върни се към основното меню",
            bg_color="#374151",
            command=on_back,
        )
        back_tile.grid(row=2, column=1, sticky="nsew", padx=10, pady=8)
        self.click_targets.append(back_tile)

    def set_temperature(self, temp_c: float | int) -> None:
        try:
            temp_val = float(temp_c)
        except Exception:
            return
        temp_int = int(round(temp_val))
        self.temperature_label.configure(text=f"Температура: {temp_int} C")

    def set_power_state(self, is_on: bool) -> None:
        if is_on:
            self.power_label.configure(text="Състояние: Включен", fg="#34d399")
        else:
            self.power_label.configure(text="Състояние: Изключен", fg="#f87171")

    def start_cursor_updates(self, interval_ms: int = 30) -> None:
        if self._cursor_job is not None:
            return
        try:
            self._cursor_canvas.lift()
        except Exception:
            pass
        self._cursor_job = self.after(interval_ms, self._update_cursor)

    def stop_cursor_updates(self) -> None:
        if self._cursor_job is not None:
            try:
                self.after_cancel(self._cursor_job)
            except Exception:
                pass
            self._cursor_job = None
        try:
            self._dwell.reset()
        except Exception:
            pass
        try:
            self._cursor_canvas.place_forget()
        except Exception:
            pass

    def _update_cursor(self) -> None:
        if not self.winfo_exists():
            self._cursor_job = None
            return

        try:
            px = self.winfo_pointerx()
            py = self.winfo_pointery()
        except Exception:
            px, py = None, None

        if px is not None and py is not None:
            root_x = self.winfo_rootx()
            root_y = self.winfo_rooty()
            w = self.winfo_width()
            h = self.winfo_height()
            local_x = px - root_x
            local_y = py - root_y

            if 0 <= local_x <= w and 0 <= local_y <= h:
                self._cursor_canvas.place(
                    x=int(local_x) - self._cursor_pad,
                    y=int(local_y) - self._cursor_pad,
                )
                try:
                    self._cursor_canvas.lift()
                except Exception:
                    pass
            else:
                try:
                    self._cursor_canvas.place_forget()
                except Exception:
                    pass

        try:
            self._dwell.update()
        except Exception:
            pass
        self._cursor_job = self.after(30, self._update_cursor)


class KeyboardMenuView(tk.Frame):
    def __init__(
        self,
        parent,
        on_power_on,
        on_power_off,
        on_back,
        on_channel_up=None,
        on_channel_down=None,
        on_volume_up=None,
        on_volume_down=None,
        menu_title="Телевизор",
        state_off_text="изключен",
    ):
        super().__init__(parent, bg="#0b1020")
        self.click_targets = []
        self._state_off_text = state_off_text
        self._dwell = DwellClickController(view=self, click_targets=self.click_targets, dwell_ms=2000)

        self._cursor_job = None
        self._cursor_size = 22
        self._cursor_pad = self._cursor_size // 2
        self._cursor_canvas = tk.Canvas(
            self,
            width=self._cursor_size,
            height=self._cursor_size,
            highlightthickness=0,
            bd=0,
            bg="#0b1020",
        )
        self._cursor_canvas.place_forget()
        self._cursor_canvas.create_oval(
            2,
            2,
            self._cursor_size - 2,
            self._cursor_size - 2,
            outline="#00E5FF",
            width=3,
        )
        self._cursor_canvas.create_oval(
            self._cursor_size // 2 - 2,
            self._cursor_size // 2 - 2,
            self._cursor_size // 2 + 2,
            self._cursor_size // 2 + 2,
            fill="#00E5FF",
            outline="#00E5FF",
        )

        keyboard_header = tk.Frame(self, bg="#0b1020")
        keyboard_header.pack(fill="x", pady=(0, 12))
        tk.Label(
            keyboard_header,
            text=f"Меню: {menu_title}",
            font=ui_font(22, "bold"),
            fg="white",
            bg="#0b1020",
        ).pack(anchor="w")
        self.state_label = tk.Label(
            keyboard_header,
            text=f"Състояние: {state_off_text}",
            font=ui_font(11),
            fg="#9ca3af",
            bg="#0b1020",
        )
        self.state_label.pack(anchor="w", pady=(4, 0))

        keyboard_actions = tk.Frame(self, bg="#0b1020")
        keyboard_actions.pack(fill="both", expand=True)
        keyboard_actions.grid_columnconfigure(0, weight=1)
        keyboard_actions.grid_columnconfigure(1, weight=1)
        keyboard_actions.grid_rowconfigure(0, weight=1)
        keyboard_actions.grid_rowconfigure(1, weight=1)
        keyboard_actions.grid_rowconfigure(2, weight=1)
        keyboard_actions.grid_rowconfigure(3, weight=1)

        self.power_on_tile = ActionTile(
            keyboard_actions,
            title="Включи",
            subtitle=f"Включи {menu_title.lower()}",
            bg_color="#10b981",
            command=on_power_on,
        )
        self.power_on_tile.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.click_targets.append(self.power_on_tile)

        self.power_off_tile = ActionTile(
            keyboard_actions,
            title="Изключи",
            subtitle=f"Изключи {menu_title.lower()}",
            bg_color="#ef4444",
            command=on_power_off,
        )
        self.power_off_tile.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.click_targets.append(self.power_off_tile)

        has_tv_controls = any(
            cb is not None for cb in (on_channel_up, on_channel_down, on_volume_up, on_volume_down)
        )
        if has_tv_controls:
            channel_up_tile = ActionTile(
                keyboard_actions,
                title="Канал +",
                subtitle="Следващ канал",
                bg_color="#2563eb",
                command=on_channel_up,
            )
            channel_up_tile.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
            self.click_targets.append(channel_up_tile)

            channel_down_tile = ActionTile(
                keyboard_actions,
                title="Канал -",
                subtitle="Предишен канал",
                bg_color="#1d4ed8",
                command=on_channel_down,
            )
            channel_down_tile.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
            self.click_targets.append(channel_down_tile)

            volume_up_tile = ActionTile(
                keyboard_actions,
                title="Сила на звука +",
                subtitle="Увеличи звука",
                bg_color="#0ea5e9",
                command=on_volume_up,
            )
            volume_up_tile.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
            self.click_targets.append(volume_up_tile)

            volume_down_tile = ActionTile(
                keyboard_actions,
                title="Сила на звука -",
                subtitle="Намали звука",
                bg_color="#0284c7",
                command=on_volume_down,
            )
            volume_down_tile.grid(row=2, column=1, sticky="nsew", padx=10, pady=10)
            self.click_targets.append(volume_down_tile)

        back_tile = ActionTile(
            keyboard_actions,
            title="Назад",
            subtitle="Върни се към основното меню",
            bg_color="#374151",
            command=on_back,
        )
        if has_tv_controls:
            back_tile.grid(row=3, column=1, sticky="nsew", padx=10, pady=10)
        else:
            back_tile.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.click_targets.append(back_tile)

    def set_power_state(self, is_on: bool):
        if is_on:
            self.state_label.configure(text="Състояние: включен", fg="#34d399")
        else:
            self.state_label.configure(text=f"Състояние: {self._state_off_text}", fg="#9ca3af")

    def set_keyboard_state(self, is_open: bool):
        # Backward-compatible alias for old call sites.
        self.set_power_state(is_open)

    def start_cursor_updates(self, interval_ms: int = 30) -> None:
        if self._cursor_job is not None:
            return
        self._cursor_job = self.after(interval_ms, self._update_cursor)

    def stop_cursor_updates(self) -> None:
        if self._cursor_job is not None:
            try:
                self.after_cancel(self._cursor_job)
            except Exception:
                pass
            self._cursor_job = None
        try:
            self._dwell.reset()
        except Exception:
            pass
        try:
            self._cursor_canvas.place_forget()
        except Exception:
            pass

    def _update_cursor(self) -> None:
        if not self.winfo_exists():
            self._cursor_job = None
            return

        try:
            px = self.winfo_pointerx()
            py = self.winfo_pointery()
        except Exception:
            px, py = None, None

        if px is not None and py is not None:
            root_x = self.winfo_rootx()
            root_y = self.winfo_rooty()
            w = self.winfo_width()
            h = self.winfo_height()
            local_x = px - root_x
            local_y = py - root_y

            if 0 <= local_x <= w and 0 <= local_y <= h:
                self._cursor_canvas.place(
                    x=int(local_x) - self._cursor_pad,
                    y=int(local_y) - self._cursor_pad,
                )
                try:
                    self._cursor_canvas.lift()
                except Exception:
                    pass
            else:
                try:
                    self._cursor_canvas.place_forget()
                except Exception:
                    pass

        try:
            self._dwell.update()
        except Exception:
            pass
        self._cursor_job = self.after(30, self._update_cursor)

import argparse
import difflib
import os
import queue
import tempfile
import threading
import time
import sys

import numpy as np
import sounddevice as sd
import tkinter as tk
from scipy.io.wavfile import write
from openai import OpenAI


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 2.5
DEFAULT_RMS_THRESHOLD = 100.0
DEFAULT_DEVICE_INDEX = None
DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEBOUNCE_SECONDS = 1.2


def parse_args():
    parser = argparse.ArgumentParser(description="BG voice commands GUI")
    parser.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX)
    parser.add_argument("--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD)
    parser.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    return parser.parse_args()


def transcribe_wav(client: OpenAI, wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=DEFAULT_MODEL,
            file=f,
            response_format="text",
            prompt="The spoken language is Bulgarian.",
        )
    return result.strip()


def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in t)
    return " ".join(t.split())


def tokenize(text: str) -> list[str]:
    t = normalize_text(text)
    return [w for w in t.split(" ") if w]


def fuzzy_contains_any(tokens: list[str], targets: set[str], cutoff: float = 0.84) -> bool:
    # Идея: Whisper понякога бърка 1-2 букви ("светината").
    # Правим "close match" на ниво дума, без външни зависимости.
    for tok in tokens:
        if tok in targets:
            return True
        m = difflib.get_close_matches(tok, targets, n=1, cutoff=cutoff)
        if m:
            return True
    return False


class VoiceThread(threading.Thread):
    def __init__(self, out_queue: "queue.Queue[tuple[str, str]]", args):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.args = args
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        if not os.environ.get("OPENAI_API_KEY"):
            self.out_queue.put(("error", "OPENAI_API_KEY не е зададен"))
            return

        try:
            client = OpenAI()
        except Exception as e:
            self.out_queue.put(("error", f"OpenAI init failed: {e}"))
            return

        self.out_queue.put(("status", "Listening..."))

        while not self.stop_event.is_set():
            try:
                audio = sd.rec(
                    int(self.args.chunk_seconds * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    device=self.args.device_index,
                )
                sd.wait()

                pcm = audio.reshape(-1)
                rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
                self.out_queue.put(("rms", f"{rms:.1f}"))

                if rms < self.args.rms_threshold:
                    time.sleep(0.05)
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name

                try:
                    self.out_queue.put(("status", "Transcribing..."))
                    write(wav_path, SAMPLE_RATE, pcm)
                    text = transcribe_wav(client, wav_path)
                    self.out_queue.put(("text", text if text else ""))
                finally:
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass

            except Exception as e:
                self.out_queue.put(("error", str(e)))
                time.sleep(0.2)

        self.out_queue.put(("status", "Stopped"))


class App(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.title("Assistive Voice Control")
        self.geometry("860x560")
        self.resizable(False, False)

        self.configure(bg="#0b1020")

        self.status_var = tk.StringVar(value="Starting...")
        self.rms_var = tk.StringVar(value="RMS: —")
        self.last_text_var = tk.StringVar(value="Last recognized: —")

        self.last_cmd = None
        self.last_cmd_at = 0.0

        self.voice_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.voice_thread = VoiceThread(self.voice_queue, args)

        self._build_ui()
        self.voice_thread.start()
        self.after(120, self._poll_voice)
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _build_ui(self):
        main = tk.Frame(self, bg="#0b1020")
        main.pack(fill="both", expand=True, padx=24, pady=24)

        header = tk.Frame(main, bg="#0b1020")
        header.pack(fill="x", pady=(0, 16))

        tk.Label(
            header,
            text="Assistive Voice Control",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#0b1020",
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Кажи: клавиатура, лампа/светлина/осветление, климатик, помощ",
            font=("Segoe UI", 11),
            fg="#98a2b3",
            bg="#0b1020",
        ).pack(anchor="w", pady=(6, 0))

        info = tk.Frame(main, bg="#121a2e", bd=0, highlightthickness=0)
        info.pack(fill="x", pady=(0, 18))

        tk.Label(info, text="STATUS", font=("Segoe UI", 9, "bold"), fg="#8b95a7", bg="#121a2e").pack(
            anchor="w", padx=18, pady=(14, 4)
        )
        self._status_label = tk.Label(
            info, textvariable=self.status_var, font=("Segoe UI", 14, "bold"), fg="#8ab4ff", bg="#121a2e"
        )
        self._status_label.pack(anchor="w", padx=18)

        row2 = tk.Frame(info, bg="#121a2e")
        row2.pack(fill="x", padx=18, pady=(10, 14))
        tk.Label(row2, textvariable=self.last_text_var, font=("Segoe UI", 11, "bold"), fg="white", bg="#121a2e").pack(
            side="left", anchor="w"
        )
        tk.Label(row2, textvariable=self.rms_var, font=("Segoe UI", 11, "bold"), fg="#8ab4ff", bg="#121a2e").pack(
            side="right", anchor="e"
        )

        grid = tk.Frame(main, bg="#0b1020")
        grid.pack(fill="both", expand=True)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)

        self.btn_keyboard = self._make_tile(grid, "Клавиатура", "Екранна клавиатура", "#10b981", self.on_keyboard)
        self.btn_light = self._make_tile(grid, "Лампа", "Осветление / светлина", "#f59e0b", self.on_light)
        self.btn_climate = self._make_tile(grid, "Климатик", "Климатизация", "#2563eb", self.on_climate)
        self.btn_help = self._make_tile(grid, "Помощ", "Спешно / SOS", "#ef4444", self.on_help)

        self.btn_keyboard.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.btn_light.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.btn_climate.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.btn_help.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        bottom = tk.Frame(main, bg="#0b1020")
        bottom.pack(fill="x", pady=(18, 0))
        tk.Button(
            bottom,
            text="Exit",
            command=self.on_exit,
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg="#7f1d1d",
            activebackground="#7f1d1d",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
        ).pack(side="right")

    def _make_tile(self, parent, title: str, subtitle: str, color: str, command):
        tile = tk.Frame(parent, bg=color, bd=0, highlightthickness=0, cursor="hand2")
        tile.pack_propagate(False)
        inner = tk.Frame(tile, bg=color, bd=0, highlightthickness=0)
        inner.pack(fill="both", expand=True)

        lbl1 = tk.Label(inner, text=title, font=("Segoe UI", 20, "bold"), fg="white", bg=color)
        lbl1.pack(anchor="w", padx=18, pady=(22, 6))
        lbl2 = tk.Label(inner, text=subtitle, font=("Segoe UI", 11), fg="#eef3ff", bg=color, wraplength=320, justify="left")
        lbl2.pack(anchor="w", padx=18)

        def darken(hex_color: str, factor: float) -> str:
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            r = max(0, min(255, int(r * factor)))
            g = max(0, min(255, int(g * factor)))
            b = max(0, min(255, int(b * factor)))
            return f"#{r:02x}{g:02x}{b:02x}"

        hover = darken(color, 0.90)

        def set_bg(c: str):
            tile.configure(bg=c)
            inner.configure(bg=c)
            lbl1.configure(bg=c)
            lbl2.configure(bg=c)

        def on_enter(_e):
            set_bg(hover)

        def on_leave(_e):
            set_bg(color)

        def on_click(_e):
            command()

        for w in (tile, inner, lbl1, lbl2):
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", on_click)

        # expose invoke() like a Button
        tile.invoke = command  # type: ignore[attr-defined]
        return tile

    def _poll_voice(self):
        if not self.winfo_exists():
            return

        while True:
            try:
                kind, payload = self.voice_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self.status_var.set(payload)
            elif kind == "rms":
                self.rms_var.set(f"RMS: {payload}")
            elif kind == "error":
                self.status_var.set(f"Error: {payload}")
            elif kind == "text":
                shown = payload if payload else "[empty]"
                self.last_text_var.set(f"Last recognized: {shown}")
                if payload:
                    self._handle_text(payload)

        self.after(120, self._poll_voice)

    def _debounced(self, cmd: str) -> bool:
        now = time.time()
        if cmd == self.last_cmd and (now - self.last_cmd_at) < DEBOUNCE_SECONDS:
            return True
        self.last_cmd = cmd
        self.last_cmd_at = now
        return False

    def _handle_text(self, text: str):
        tokens = tokenize(text)

        help_words = {"помощ", "помогни", "спешно", "авария", "sos", "сос"}
        climate_words = {"климатик", "климатика", "климатикът", "климатизация", "климатика"}
        light_words = {
            "лампа",
            "лампата",
            "светлина",
            "светлината",
            "осветление",
            "осветлението",
            "светни",
            "светло",
            "светлинка",
            "светината",  # често грешно разпознато
        }
        keyboard_words = {"клавиатура", "клавиатурата", "keyboard", "екранна"}

        # Приоритет: помощ > климатик > лампа/светлина > клавиатура
        if fuzzy_contains_any(tokens, help_words, cutoff=0.84):
            if not self._debounced("help"):
                self.btn_help.invoke()
            return

        if fuzzy_contains_any(tokens, climate_words, cutoff=0.86):
            if not self._debounced("climate"):
                self.btn_climate.invoke()
            return

        if fuzzy_contains_any(tokens, light_words, cutoff=0.84):
            if not self._debounced("light"):
                self.btn_light.invoke()
            return

        # клавиатура: искаме да не се активира само от "екранна", затова търсим по-строго
        if "клавиатура" in tokens or "клавиатурата" in tokens or fuzzy_contains_any(tokens, {"keyboard"}, cutoff=0.9):
            if not self._debounced("keyboard"):
                self.btn_keyboard.invoke()
            return

        self.status_var.set("Чут текст (без команда)")

    def _mark_pressed(self, name: str):
        msg = f"Натиснат бутон: {name}"
        self.status_var.set(msg)
        print(msg, flush=True)

    def on_keyboard(self):
        self._mark_pressed("Клавиатура")

    def on_light(self):
        self._mark_pressed("Лампа")

    def on_climate(self):
        self._mark_pressed("Климатик")

    def on_help(self):
        self._mark_pressed("Помощ")

    def on_exit(self):
        try:
            self.voice_thread.stop()
        except Exception:
            pass
        self.destroy()


def main():
    args = parse_args()
    app = App(args)
    app.mainloop()


if __name__ == "__main__":
    main()
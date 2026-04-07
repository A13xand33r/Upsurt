import queue
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import tkinter as tk
from faster_whisper import WhisperModel


# ===== Настройки (можеш да ги пипнеш) =====
MODEL_SIZE = "tiny"      # tiny/base/small/...
DEVICE_INDEX = None      # None = default input device; иначе число (пример: 1)
SAMPLE_RATE = 16000

CHUNK_SECONDS = 0.5
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_SECONDS)

RMS_THRESHOLD = 800.0    # вдигни ако хваща шум; снижи ако не хваща реч
SILENCE_CHUNKS = 3       # колко “тихи” chunks = край на фразата
MIN_SPEECH_CHUNKS = 2    # минимум chunks реч, за да транскрибира

DEBOUNCE_SECONDS = 1.2   # да не “натиска” 10 пъти от една фраза


def _write_wav_pcm16(path: str, pcm_int16: np.ndarray) -> None:
    pcm_int16 = np.asarray(pcm_int16, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


class SimpleVoiceGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Voice Commands (BG)")
        self.geometry("520x340")
        self.resizable(False, False)

        self.status_var = tk.StringVar(value="Готово. Кажи: 'клавиатура', 'лампа', 'климатик', 'помощ'.")
        self.last_voice_var = tk.StringVar(value="Последно разпознато: —")

        top = tk.Frame(self)
        top.pack(fill="x", padx=16, pady=(16, 8))

        tk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(top, textvariable=self.last_voice_var, font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 0))

        grid = tk.Frame(self)
        grid.pack(fill="both", expand=True, padx=16, pady=16)

        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)

        self.btn_keyboard = tk.Button(grid, text="Клавиатура", font=("Segoe UI", 14, "bold"), command=self.on_keyboard)
        self.btn_light = tk.Button(grid, text="Лампа", font=("Segoe UI", 14, "bold"), command=self.on_light)
        self.btn_climate = tk.Button(grid, text="Климатик", font=("Segoe UI", 14, "bold"), command=self.on_climate)
        self.btn_help = tk.Button(grid, text="Помощ", font=("Segoe UI", 14, "bold"), fg="white", bg="#b91c1c", command=self.on_help)

        self.btn_keyboard.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.btn_light.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.btn_climate.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.btn_help.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        bottom = tk.Frame(self)
        bottom.pack(fill="x", padx=16, pady=(0, 16))
        tk.Button(bottom, text="Изход", command=self.destroy).pack(side="right")

        # Voice thread infra
        self.voice_queue: queue.Queue[str] = queue.Queue()
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        self.after(150, self._poll_voice_queue)

        self._last_cmd = None
        self._last_cmd_at = 0.0

    # ===== “Действия” на бутоните =====
    def _mark_pressed(self, name: str) -> None:
        msg = f"Натиснат бутон: {name}"
        print(msg, flush=True)
        self.status_var.set(msg)

    def on_keyboard(self):
        self._mark_pressed("Клавиатура")

    def on_light(self):
        self._mark_pressed("Лампа")

    def on_climate(self):
        self._mark_pressed("Климатик")

    def on_help(self):
        self._mark_pressed("Помощ")

    # ===== Voice обработка =====
    def _poll_voice_queue(self):
        if not self.winfo_exists():
            return

        while True:
            try:
                text = self.voice_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_voice_text(text)

        self.after(150, self._poll_voice_queue)

    def _handle_voice_text(self, text: str) -> None:
        self.last_voice_var.set(f"Последно разпознато: {text}")
        t = _normalize_text(text)

        cmd = None
        if "клавиатура" in t:
            cmd = "keyboard"
        elif any(k in t for k in ["лампа", "светлина", "осветление"]):
            cmd = "light"
        elif "климатик" in t:
            cmd = "climate"
        elif any(k in t for k in ["помощ", "спешно", "авария"]):
            cmd = "help"

        if cmd is None:
            return

        now = time.time()
        if cmd == self._last_cmd and (now - self._last_cmd_at) < DEBOUNCE_SECONDS:
            return
        self._last_cmd = cmd
        self._last_cmd_at = now

        # “Натискаме” бутона
        if cmd == "keyboard":
            self.btn_keyboard.invoke()
        elif cmd == "light":
            self.btn_light.invoke()
        elif cmd == "climate":
            self.btn_climate.invoke()
        elif cmd == "help":
            self.btn_help.invoke()

    def _voice_worker(self):
        try:
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        except Exception as e:
            self.voice_queue.put(f"[Грешка при зареждане на Whisper] {e}")
            return

        q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()

        def callback(indata, frames, time_info, status):
            x = np.asarray(indata).reshape(-1).astype(np.int16)
            rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
            q.put((x, rms))

        speaking = False
        silence = 0
        speech = 0
        buf = []

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=callback,
                blocksize=BLOCKSIZE,
                device=DEVICE_INDEX,
            ):
                self.voice_queue.put("[Voice] Слушам...")

                while True:
                    chunk, rms = q.get()

                    if not speaking:
                        if rms >= RMS_THRESHOLD:
                            speaking = True
                            silence = 0
                            speech = 1
                            buf = [chunk]
                        continue

                    buf.append(chunk)
                    if rms >= RMS_THRESHOLD:
                        silence = 0
                        speech += 1
                    else:
                        silence += 1

                    if silence >= SILENCE_CHUNKS:
                        speaking = False

                        if speech >= MIN_SPEECH_CHUNKS and buf:
                            pcm = np.concatenate(buf, axis=0)
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                path = tmp.name
                            try:
                                _write_wav_pcm16(path, pcm)
                                segments, _info = model.transcribe(
                                    path,
                                    language="bg",
                                    task="transcribe",
                                    vad_filter=False,
                                    beam_size=3,
                                    initial_prompt="Говоря на български език.",
                                )
                                text = " ".join(seg.text.strip() for seg in segments).strip()
                                if text:
                                    self.voice_queue.put(text)
                            finally:
                                try:
                                    import os
                                    os.remove(path)
                                except OSError:
                                    pass

                        silence = 0
                        speech = 0
                        buf = []

        except Exception as e:
            self.voice_queue.put(f"[Voice error] {e}")


if __name__ == "__main__":
    app = SimpleVoiceGUI()
    app.mainloop()


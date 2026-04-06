
import os
import time
import tempfile

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import RPi.GPIO as GPIO

SAMPLE_RATE = 16000
RECORD_SECONDS = 3
MODEL_NAME = "tiny"
LANGUAGE = "bg"

LEFT_IN1 = 7
LEFT_IN2 = 8
RIGHT_IN1 = 23
RIGHT_IN2 = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_IN1, GPIO.OUT)
GPIO.setup(LEFT_IN2, GPIO.OUT)
GPIO.setup(RIGHT_IN1, GPIO.OUT)
GPIO.setup(RIGHT_IN2, GPIO.OUT)

def motors_stop():
    GPIO.output(LEFT_IN1, 0)
    GPIO.output(LEFT_IN2, 0)
    GPIO.output(RIGHT_IN1, 0)
    GPIO.output(RIGHT_IN2, 0)
    print("STOP")

def motors_forward():
    GPIO.output(LEFT_IN1, 0)
    GPIO.output(LEFT_IN2, 1)
    GPIO.output(RIGHT_IN1, 0)
    GPIO.output(RIGHT_IN2, 1)
    print("FORWARD")

def motors_backward():
    GPIO.output(LEFT_IN1, 1)
    GPIO.output(LEFT_IN2, 0)
    GPIO.output(RIGHT_IN1, 1)
    GPIO.output(RIGHT_IN2, 0)
    print("BACKWARD")

def motors_left():
    GPIO.output(LEFT_IN1, 1)
    GPIO.output(LEFT_IN2, 0)
    GPIO.output(RIGHT_IN1, 0)
    GPIO.output(RIGHT_IN2, 1)
    print("LEFT")

def motors_right():
    GPIO.output(LEFT_IN1, 0)
    GPIO.output(LEFT_IN2, 1)
    GPIO.output(RIGHT_IN1, 1)
    GPIO.output(RIGHT_IN2, 0)
    print("RIGHT")

def record_audio(filename, seconds=3, fs=16000):
    print("Listening...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    write(filename, fs, audio)

def normalize_text(text):
    text = text.lower().strip()
    replacements = {
        "наляво": "ляво",
        "надясно": "дясно",
        "спри": "стоп",
        "стой": "стоп",
        "тръгни": "напред",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def execute_command(text):
    text = normalize_text(text)
    print("Recognized:", text)

    if "напред" in text:
        motors_forward()
    elif "назад" in text:
        motors_backward()
    elif "ляво" in text:
        motors_left()
    elif "дясно" in text:
        motors_right()
    elif "стоп" in text:
        motors_stop()
    else:
        print("Unknown command")

print("Loading model...")
model = whisper.load_model(MODEL_NAME)
print("Model loaded.")

try:
    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_file = tmp.name

        record_audio(wav_file, seconds=RECORD_SECONDS, fs=SAMPLE_RATE)

        result = model.transcribe(wav_file, language=LANGUAGE, fp16=False)
        spoken_text = result["text"].strip()

        if spoken_text:
            execute_command(spoken_text)

        os.remove(wav_file)
        time.sleep(0.2)

except KeyboardInterrupt:
    pass

finally:
    motors_stop()
    GPIO.cleanup()
EOF

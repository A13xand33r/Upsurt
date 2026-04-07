import argparse
import os
import queue
import tempfile
import threading
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

CHUNK_SECONDS = 0.3
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_SECONDS)

DEFAULT_MODEL_SIZE = "small"
DEFAULT_RMS_THRESHOLD = 80.0
DEFAULT_SILENCE_CHUNKS = 3
DEFAULT_MIN_SPEECH_CHUNKS = 1
DEFAULT_DEVICE_INDEX = None


def parse_args():
    parser = argparse.ArgumentParser(description="Bulgarian voice to text test")
    parser.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX, help="Microphone device index")
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, help="Whisper model size")
    parser.add_argument("--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD, help="Speech RMS threshold")
    parser.add_argument("--silence-chunks", type=int, default=DEFAULT_SILENCE_CHUNKS, help="Silent chunks before phrase ends")
    parser.add_argument("--min-speech-chunks", type=int, default=DEFAULT_MIN_SPEECH_CHUNKS, help="Minimum speech chunks")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return parser.parse_args()


def write_wav_pcm16(path: str, pcm_int16: np.ndarray) -> None:
    pcm_int16 = np.asarray(pcm_int16, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())


class VoiceToTextWorker:
    def __init__(self, device_index, model_size, rms_threshold, silence_chunks, min_speech_chunks):
        self.device_index = device_index
        self.model_size = model_size
        self.rms_threshold = rms_threshold
        self.silence_chunks = silence_chunks
        self.min_speech_chunks = min_speech_chunks
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        print(f"Loading model: {self.model_size}")
        model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        print("Model ready.")

        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if self.stop_event.is_set():
                return

            if status:
                print("Audio status:", status)

            x = np.asarray(indata).reshape(-1).astype(np.int16)
            rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
            print(f"\rRMS: {rms:8.1f}", end="", flush=True)
            audio_queue.put((x, rms))

        speaking = False
        silence_count = 0
        speech_chunks = 0
        audio_buffer = []

        stream_kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback,
            blocksize=BLOCKSIZE,
        )

        if self.device_index is not None:
            stream_kwargs["device"] = self.device_index

        print("\nListening... Press Ctrl+C to stop.")
        print("Say something in Bulgarian.")

        with sd.InputStream(**stream_kwargs):
            while not self.stop_event.is_set():
                try:
                    chunk_pcm, rms = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if not speaking:
                    if rms >= self.rms_threshold:
                        speaking = True
                        silence_count = 0
                        speech_chunks = 1
                        audio_buffer = [chunk_pcm]
                    continue

                audio_buffer.append(chunk_pcm)

                if rms >= self.rms_threshold:
                    silence_count = 0
                    speech_chunks += 1
                else:
                    silence_count += 1

                if silence_count >= self.silence_chunks:
                    speaking = False

                    if speech_chunks >= self.min_speech_chunks and audio_buffer:
                        pcm = np.concatenate(audio_buffer, axis=0)

                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp_path = tmp.name

                        try:
                            write_wav_pcm16(tmp_path, pcm)

                            print("\nTranscribing...")

                            segments, info = model.transcribe(
                                tmp_path,
                                language="bg",
                                task="transcribe",
                                beam_size=4,
                                vad_filter=False,
                                initial_prompt="Говоря на български език."
                            )

                            parts = []
                            for seg in segments:
                                part = seg.text.strip()
                                if part:
                                    parts.append(part)

                            text = " ".join(parts).strip()
                            text = text.replace(" ,", ",").replace(" .", ".")

                            if text:
                                print(f"TEXT: {text}")
                            else:
                                print("TEXT: [no speech recognized]")

                        except Exception as e:
                            print(f"\nTranscribe failed: {e}")

                        finally:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                    silence_count = 0
                    speech_chunks = 0
                    audio_buffer = []


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print("Default devices:", sd.default.device)
    if args.device_index is not None:
        print("Using input device index:", args.device_index)
    else:
        print("Using default input device.")

    worker = VoiceToTextWorker(
        device_index=args.device_index,
        model_size=args.model_size,
        rms_threshold=args.rms_threshold,
        silence_chunks=args.silence_chunks,
        min_speech_chunks=args.min_speech_chunks,
    )

    try:
        worker.run()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        worker.stop()
    except Exception as e:
        print(f"\nFatal error: {e}")


if __name__ == "__main__":
    main()
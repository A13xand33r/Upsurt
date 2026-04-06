import json
import os
import threading
import time
from typing import Any


try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover
    mqtt = None


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or str(val).strip() == "" else str(val).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else int(default)
    except Exception:
        return int(default)


MQTT_HOST = _env_str("MQTT_HOST", "localhost")
MQTT_PORT = _env_int("MQTT_PORT", 1883)
MQTT_USERNAME = _env_str("MQTT_USERNAME", "")
MQTT_PASSWORD = _env_str("MQTT_PASSWORD", "")
MQTT_CLIENT_ID = _env_str("MQTT_CLIENT_ID", "upsurt-pi")

# Default control topic (payload is JSON).
MQTT_TOPIC_CONTROL = _env_str("MQTT_TOPIC_CONTROL", "upsurt/esp32/control")


class MqttPublisher:
    def __init__(self) -> None:
        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed (pip install paho-mqtt)")

        self._client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
        if MQTT_USERNAME:
            self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)

        self._connected = False
        self._connect_lock = threading.Lock()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Start network loop; connection happens lazily on first publish.
        self._client.loop_start()

    def _on_connect(self, _client, _userdata, _flags, rc, _properties=None):  # type: ignore[no-untyped-def]
        self._connected = (rc == 0)

    def _on_disconnect(self, _client, _userdata, _rc, _properties=None):  # type: ignore[no-untyped-def]
        self._connected = False

    def ensure_connected(self, timeout_sec: float = 2.0) -> bool:
        if self._connected:
            return True
        with self._connect_lock:
            if self._connected:
                return True
            try:
                # Connect synchronously; publish calls can happen frequently.
                self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
            except Exception:
                self._connected = False
                return False

        # Wait briefly for on_connect.
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            if self._connected:
                return True
            time.sleep(0.02)
        return bool(self._connected)

    def publish_json(self, topic: str, payload: dict[str, Any], qos: int = 0, retain: bool = False) -> bool:
        ok = self.ensure_connected()
        if not ok:
            return False
        try:
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            info = self._client.publish(topic, payload=data, qos=int(qos), retain=bool(retain))
            # paho-mqtt returns immediately; success means queued for sending.
            return info.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception:
            return False


_publisher: MqttPublisher | None = None


def get_publisher() -> MqttPublisher:
    global _publisher
    if _publisher is None:
        _publisher = MqttPublisher()
    return _publisher


def publish_control(payload: dict[str, Any]) -> bool:
    return get_publisher().publish_json(MQTT_TOPIC_CONTROL, payload)


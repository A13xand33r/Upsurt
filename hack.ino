#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <ESPmDNS.h> // mDNS

// Wi-Fi credentials
const char* ssid = "Parolataenz123456";
const char* password = "nz123456";

// HTTP server
WebServer server(80);

// Pins
const int led1 = 12;
const int led2 = 25;
const int led3 = 27;
const int motorPin = 17;  // DC motor
const int servoPin = 18; //tings
const int ledFreq = 5000;
const int ledRes = 8;      // 0-255

// Handle HTTP POST commands
void handleControl() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "No body");
    return;
  }

  String body = server.arg("plain");
  Serial.println("Incoming:");
  Serial.println(body);

  StaticJsonDocument<256> doc;
  DeserializationError error = deserializeJson(doc, body);
  if (error) {
    server.send(400, "text/plain", "Bad JSON");
    return;
  }

  String device = doc["device"];
  String action = doc["action"];

  // LED1
  if (device == "led1") {
    if (action == "on") {
      int brightness = doc["speed"] | 255;
      ledcWrite(led1, brightness);
    } else {
      ledcWrite(led1, 0);
    }
  }

  // LED2
  if (device == "led2") {
    if (action == "on") {
      int brightness = doc["speed"] | 255;
      ledcWrite(led2, brightness);
    } else {
      ledcWrite(led2, 0);
    }
  }

  if (device == "led3") {
    if (action == "on") {
      int brightness = doc["speed"] | 255;
      ledcWrite(led3, brightness);
    } else {
      ledcWrite(led3, 0);
    }
  }

  // DC Motor ON/OFF (digitalWrite)
  if (device == "motor") {
    if (action == "off") {
      digitalWrite(motorPin, HIGH);
    } else {
      digitalWrite(motorPin, LOW);
    }
  }

  server.send(200, "application/json", "{\"status\":\"ok\"}");
}

void setup() {
  Serial.begin(115200);

  // PWM setup for LEDs
  ledcAttach(led1, ledFreq, ledRes);
  ledcAttach(led2, ledFreq, ledRes);
  ledcAttach(led3, ledFreq, ledRes);

  // DC Motor
  pinMode(motorPin, OUTPUT);
  digitalWrite(motorPin, HIGH); // start OFF

  // Connect Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");

  // mDNS
  if (!MDNS.begin("esp32")) {
    Serial.println("Error starting mDNS");
  } else {
    Serial.println("mDNS started: esp32.local");
  }

  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // HTTP route
  server.on("/control", HTTP_POST, handleControl);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
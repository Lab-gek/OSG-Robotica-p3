/*
 * robot_controller.ino
 * ESP32 firmware – Line-following robot HTTP command receiver
 *
 * -----------------------------------------------------------------------
 * NOTE: This is a REFERENCE sketch that implements the HTTP API expected
 * by the Python controller (esp32_client.py).  If you already have an
 * existing .ino, merge the WiFi + WebServer sections below into it.
 * The only endpoint the Python side calls is:
 *
 *   GET /command?action=<ACTION>&speed=<SPEED>
 *
 *   ACTION : FORWARD | LEFT | RIGHT | STOP
 *   SPEED  : 0–100  (mapped to PWM duty cycle, i.e. analogWrite 0–255)
 * -----------------------------------------------------------------------
 *
 * Hardware assumed
 * ----------------
 *   Motor driver : L298N (or equivalent dual H-bridge)
 *   Motor A (left wheel)  : IN1, IN2, ENA (PWM)
 *   Motor B (right wheel) : IN3, IN4, ENB (PWM)
 *
 * Wiring (adjust pin numbers to match your board)
 * -------------------------------------------------
 *   ENA  → GPIO 14   (PWM channel 0)
 *   IN1  → GPIO 27
 *   IN2  → GPIO 26
 *   ENB  → GPIO 25   (PWM channel 1)
 *   IN3  → GPIO 33
 *   IN4  → GPIO 32
 *
 * WiFi modes
 * ----------
 *   Access-Point (AP) mode is the default: the ESP32 creates its own WiFi
 *   network.  Connect your PC to that network and set ESP32_BASE_URL to
 *   "http://192.168.4.1" in config.py (already the default).
 *
 *   To use Station (STA) mode instead, set WIFI_MODE to WIFI_STA and fill
 *   in WIFI_SSID / WIFI_PASSWORD below.  Then update ESP32_BASE_URL in
 *   config.py to the DHCP address printed on the Serial monitor.
 *
 * Dependencies
 * ------------
 *   - Arduino core for ESP32  (https://github.com/espressif/arduino-esp32)
 *   - No additional libraries needed; WebServer.h is bundled with the core.
 *
 * Build & upload
 * --------------
 *   Board     : "ESP32 Dev Module" (or your specific board)
 *   Upload speed : 115200 baud
 */

#include <WiFi.h>
#include <WebServer.h>

// -------------------------------------------------------------------------
// WiFi configuration
// -------------------------------------------------------------------------
#define WIFI_MODE    WIFI_AP    // Change to WIFI_STA for station mode

// AP mode credentials (ESP32 creates this network)
const char* AP_SSID     = "RobotAP";
const char* AP_PASSWORD = "robot1234";   // min 8 chars; use "" for open network

// STA mode credentials (only used when WIFI_MODE == WIFI_STA)
const char* STA_SSID     = "YourNetworkName";
const char* STA_PASSWORD = "YourNetworkPassword";

// -------------------------------------------------------------------------
// Motor driver pin assignments  – adjust to match your wiring
// -------------------------------------------------------------------------
// Left motor  (Motor A)
const int PIN_ENA = 14;
const int PIN_IN1 = 27;
const int PIN_IN2 = 26;

// Right motor (Motor B)
const int PIN_ENB = 25;
const int PIN_IN3 = 33;
const int PIN_IN4 = 32;

// PWM settings (ESP32 LEDC)
const int PWM_FREQ       = 1000;  // Hz
const int PWM_RESOLUTION = 8;     // bits → 0-255
const int PWM_CHANNEL_A  = 0;
const int PWM_CHANNEL_B  = 1;

// -------------------------------------------------------------------------
// HTTP server
// -------------------------------------------------------------------------
WebServer server(80);

// -------------------------------------------------------------------------
// Motor helpers
// -------------------------------------------------------------------------

// Convert speed percentage (0-100) to 8-bit PWM value (0-255)
int speedToPwm(int speedPct) {
  return map(constrain(speedPct, 0, 100), 0, 100, 0, 255);
}

void motorStop() {
  ledcWrite(PWM_CHANNEL_A, 0);
  ledcWrite(PWM_CHANNEL_B, 0);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
}

void motorForward(int speedPct) {
  int pwm = speedToPwm(speedPct);
  // Left motor forward
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  ledcWrite(PWM_CHANNEL_A, pwm);
  // Right motor forward
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  ledcWrite(PWM_CHANNEL_B, pwm);
}

void motorLeft(int speedPct) {
  // Arc left: left motor reverses at half speed, right motor drives forward
  int pwm = speedToPwm(speedPct);
  // Left motor: reverse at half speed (tighter arc)
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  ledcWrite(PWM_CHANNEL_A, pwm / 2);
  // Right motor forward at full turn speed
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  ledcWrite(PWM_CHANNEL_B, pwm);
}

void motorRight(int speedPct) {
  // Arc right: right motor reverses at half speed, left motor drives forward
  int pwm = speedToPwm(speedPct);
  // Left motor forward at full turn speed
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  ledcWrite(PWM_CHANNEL_A, pwm);
  // Right motor: reverse at half speed (tighter arc)
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  ledcWrite(PWM_CHANNEL_B, pwm / 2);
}

// -------------------------------------------------------------------------
// HTTP handler  –  GET /command?action=FORWARD&speed=60
// -------------------------------------------------------------------------
void handleCommand() {
  if (!server.hasArg("action")) {
    server.send(400, "text/plain", "Missing 'action' parameter");
    return;
  }

  String action = server.arg("action");
  int    speed  = server.hasArg("speed") ? server.arg("speed").toInt() : 50;

  Serial.printf("[CMD] action=%s speed=%d\n", action.c_str(), speed);

  if      (action == "FORWARD") motorForward(speed);
  else if (action == "LEFT")    motorLeft(speed);
  else if (action == "RIGHT")   motorRight(speed);
  else if (action == "STOP")    motorStop();
  else {
    server.send(400, "text/plain", "Unknown action: " + action);
    return;
  }

  server.send(200, "text/plain", "OK");
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// -------------------------------------------------------------------------
// Setup
// -------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(100);

  // Motor pins
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);

  // PWM channels (ESP32 LEDC API)
  ledcSetup(PWM_CHANNEL_A, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CHANNEL_B, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(PIN_ENA, PWM_CHANNEL_A);
  ledcAttachPin(PIN_ENB, PWM_CHANNEL_B);

  motorStop();

  // WiFi
  if (WIFI_MODE == WIFI_AP) {
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    Serial.printf("[WiFi] AP mode – SSID: %s  IP: %s\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str());
  } else {
    WiFi.begin(STA_SSID, STA_PASSWORD);
    Serial.print("[WiFi] Connecting to ");
    Serial.print(STA_SSID);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.printf("\n[WiFi] Connected – IP: %s\n",
                  WiFi.localIP().toString().c_str());
  }

  // HTTP routes
  server.on("/command", HTTP_GET, handleCommand);
  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println("[HTTP] Server started");
  Serial.println("[HTTP] Endpoint: GET /command?action=<ACTION>&speed=<SPEED>");
}

// -------------------------------------------------------------------------
// Loop
// -------------------------------------------------------------------------
void loop() {
  server.handleClient();
}

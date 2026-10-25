/*
  Bionic Hand — ESP32 Firmware
  ════════════════════════════
  Hardware : ESP32 DevKit V1 (38-pin)
             PCA9685 16-ch PWM driver  (I2C: SDA=GPIO21, SCL=GPIO22, addr=0x40)
             6 × SG90 / MG90S servos

  Channels : 0=Index  1=Middle  2=Ring  3=Pinky  4=Thumb-curl  5=Thumb-CMC

  Serial   : 115200 baud
  Protocol : "a0,a1,a2,a3,a4,a5\n"  — 6 comma-separated ints, 0-180 each

  Tight elastic handling (on Python side):
    RAMP_SPEED and DEAD_BAND in five_finger_control.py already limit how
    fast angles change, so the ESP32 just needs to apply them faithfully.
    No extra delay needed here.

  Library  : Adafruit PWM Servo Driver Library
             Install via Arduino Library Manager → "Adafruit PWM Servo Driver"
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ── PCA9685 ──────────────────────────────────────────────────
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

// SG90 / MG90S pulse range (microseconds → 12-bit PCA9685 ticks at 50 Hz)
// 50 Hz → 1 tick = 1,000,000 / (50 × 4096) ≈ 4.88 µs
// 500 µs = ~102 ticks   2400 µs = ~491 ticks
#define SERVO_MIN   102    // 0°
#define SERVO_MAX   491    // 180°
#define NUM_SERVOS  6

// ── Current servo positions ───────────────────────────────────
int currentAngle[NUM_SERVOS] = {10, 10, 10, 10, 10, 80};

// ── Angle → PCA9685 pulse ─────────────────────────────────────
uint16_t angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

// ── Move one servo ────────────────────────────────────────────
void setServo(uint8_t ch, int angle) {
  pca.setPWM(ch, 0, angleToPulse(angle));
  currentAngle[ch] = angle;
}

// ── Startup: open all fingers gently ─────────────────────────
void openAllFingers() {
  // Move in small steps to not shock tight elastic on boot
  for (int a = 90; a >= 10; a -= 5) {
    for (int ch = 0; ch < NUM_SERVOS; ch++) {
      setServo(ch, a);
    }
    delay(20);
  }
  // Thumb CMC to neutral
  setServo(5, 80);
}

// ── Parse and apply incoming serial line ─────────────────────
void applyAngles(String line) {
  line.trim();
  if (line.length() == 0) return;

  int angles[NUM_SERVOS];
  int idx   = 0;
  int start = 0;

  for (int i = 0; i <= (int)line.length() && idx < NUM_SERVOS; i++) {
    if (i == (int)line.length() || line[i] == ',') {
      String token = line.substring(start, i);
      token.trim();
      if (token.length() > 0) {
        angles[idx++] = constrain(token.toInt(), 0, 180);
      }
      start = i + 1;
    }
  }

  // Only apply if we got all 6 values
  if (idx == NUM_SERVOS) {
    for (int ch = 0; ch < NUM_SERVOS; ch++) {
      setServo(ch, angles[ch]);
    }
  }
}

// ── setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("Bionic Hand — ESP32 ready");

  Wire.begin(21, 22);    // SDA=GPIO21, SCL=GPIO22
  pca.begin();
  pca.setOscillatorFrequency(27000000);  // calibrate if servos twitch
  pca.setPWMFreq(50);                    // 50 Hz for SG90/MG90S

  delay(10);
  openAllFingers();
  Serial.println("Fingers opened. Waiting for serial commands...");
}

// ── loop ──────────────────────────────────────────────────────
void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    applyAngles(line);
  }
}

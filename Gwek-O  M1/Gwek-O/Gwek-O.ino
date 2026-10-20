
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define SERVO_MIN_US  500
#define SERVO_MAX_US  2400
#define SERVO_FREQ    50

int angleToPulse(int angle) {
  int us = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  return (int)(us * SERVO_FREQ * 4096.0 / 1000000.0);
}

void writeServo(int channel, int angle) {
  angle = constrain(angle, 0, 180);
  pca.setPWM(channel, 0, angleToPulse(angle));
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);

  Wire.begin(21, 22);
  pca.begin();
  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(SERVO_FREQ);
  delay(300);

  for (int ch = 0; ch < 5; ch++) {
    writeServo(ch, 0);
  }

  Serial.println("READY - FIVE FINGERS");
  Serial.println("Format: thumb,index,middle,ring,little");
}

void loop() {
  if (!Serial.available()) return;

  String data = Serial.readStringUntil('\n');
  data.trim();

  int angles[5];
  int start = 0;

  for (int i = 0; i < 5; i++) {
    int comma = data.indexOf(',', start);

    if (i < 4 && comma == -1) return;
    if (i == 4 && comma != -1) return;

    String part;
    if (i < 4) {
      part = data.substring(start, comma);
      start = comma + 1;
    } else {
      part = data.substring(start);
    }

    part.trim();
    if (part.length() == 0) return;

    // Reject anything other than digits.
    for (unsigned int j = 0; j < part.length(); j++) {
      if (!isDigit(part[j])) return;
    }

    angles[i] = constrain(part.toInt(), 0, 180);
  }

  for (int ch = 0; ch < 5; ch++) {
    writeServo(ch, angles[ch]);
  }
}

/*
  GWEK-O  -  ESP32 firmware
  Receives one line per frame over USB serial:  a0,a1,a2,a3,a4,a5\n   (0..180 degrees)
  and drives six servos on PCA9685 channels 0..5.

  Channels: 0 Index | 1 Middle | 2 Ring | 3 Pinky | 4 Thumb curl | 5 Thumb CMC
  Wiring  : GPIO21 -> SDA, GPIO22 -> SCL, 3V3 -> VCC, GND -> GND,
            external 5 V / 3 A supply -> PCA9685 V+ / GND (grounds joined with ESP32)
  Board   : ESP32 Dev Module     Library: Adafruit PWM Servo Driver
*/
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

const int NUM_CH    = 6;
const int SERVO_MIN = 102;   // 0.5 ms pulse at 50 Hz (0 degrees)
const int SERVO_MAX = 491;   // 2.4 ms pulse at 50 Hz (180 degrees)

char    buf[64];
uint8_t len = 0;

// Split "a0,a1,...,a5" into six integers. Returns true only if exactly six values.
bool parseLine(char *s, int *out) {
  int   count = 0;
  char *tok   = strtok(s, ",");
  while (tok != NULL && count < NUM_CH) {
    out[count++] = constrain(atoi(tok), 0, 180);
    tok = strtok(NULL, ",");
  }
  return (count == NUM_CH && tok == NULL);   // reject short or over-long packets
}

void applyAngles(const int *a) {
  for (int ch = 0; ch < NUM_CH; ch++) {
    int pulse = map(a[ch], 0, 180, SERVO_MIN, SERVO_MAX);
    pca.setPWM(ch, 0, pulse);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);          // SDA, SCL
  pca.begin();
  pca.setPWMFreq(50);          // 50 Hz for hobby servos
  Serial.println("GWEK-O ready: send a0,a1,a2,a3,a4,a5");
}

void loop() {
  // Non-blocking line reader: never waits, so old frames cannot pile up.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf[len] = '\0';
      int a[NUM_CH];
      if (parseLine(buf, a)) applyAngles(a);   // bad packets are ignored
      len = 0;
    } else if (c != '\r') {
      if (len < sizeof(buf) - 1) buf[len++] = c;
      else len = 0;                            // overflow: discard and resync
    }
  }
}

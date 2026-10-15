/*
  Plot_IMU.ino
  Usage: upload, open Serial Plotter, hold the board in one posture's
  orientation until the trace settles (~2-3s), screenshot it, repeat for
  the next posture.
*/

#include <Arduino_LSM9DS1.h>

const unsigned long SAMPLE_INTERVAL_MS = 20;  // ~50 Hz, smooth enough to watch live
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);

  if (!IMU.begin()) {
    // Serial Plotter can't show text, so just fast-blink the LED to signal failure.
    pinMode(LED_BUILTIN, OUTPUT);
    while (1) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(100);
    }
  }
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  float ax, ay, az;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);

    Serial.print("AX:");
    Serial.print(ax, 3);
    Serial.print(",AY:");
    Serial.print(ay, 3);
    Serial.print(",AZ:");
    Serial.println(az, 3);
  }
}

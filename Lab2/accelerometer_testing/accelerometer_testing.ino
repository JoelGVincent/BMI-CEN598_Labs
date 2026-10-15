// Board: Arduino Nano 33 BLE Sense (Rev1) -- onboard IMU is the LSM9DS1.
#include <Arduino_LSM9DS1.h>
// #include "Arduino_BMI270_BMM150.h" this library is for the rev 2 board. 
// the board i'm using requires LSM9DS1, which I learned after debugging with Claude

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);

  // Wait up to 3s for a Serial Monitor to connect
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) {}

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    // Fast-blink forever so failure is visible even with no Serial Monitor
    // connected.
    while (1) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(100);
    }
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  Serial.println("Accelerometer in g's");
  Serial.println("X\tY\tZ");
}

void loop() {
  float x, y, z;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    Serial.print(x);
    Serial.print("\t");
    Serial.print(y);
    Serial.print("\t");
    Serial.println(z);
  }

  delay(100);
}

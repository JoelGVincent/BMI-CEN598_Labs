/*
  Serial_Test.ino
  Diagnostic sketch — no `while (!Serial)` gate, no IMU calls.

  The LED blinks on its own regardless of whether a Serial Monitor is
  connected, so you can tell "is the board even running my code?"
  independently of "is Serial actually getting through?".

  Use this to bisect a no-output problem:
    - LED doesn't blink at all -> upload/board problem (wrong board/port
      selected, upload didn't actually take, bad cable).
    - LED blinks but no text ever appears in Serial Monitor -> Serial
      Monitor problem (wrong port, wrong baud rate — must be 115200,
      monitor opened before/instead of after upload finished).
    - Both work here, but accelerometer_testing.ino still hangs with
      zero output -> the hang is inside IMU.begin() (I2C/library issue),
      see Serial_Test_With_IMU below.
*/

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
}

unsigned long lastToggle = 0;
bool ledOn = false;
unsigned long counter = 0;

void loop() {
  unsigned long now = millis();
  if (now - lastToggle >= 500) {
    lastToggle = now;
    ledOn = !ledOn;
    digitalWrite(LED_BUILTIN, ledOn ? HIGH : LOW);

    // Serial.print works fine even with no Monitor connected on native-USB
    // boards -- it just gets discarded until something opens the port.
    Serial.print("alive #");
    Serial.println(counter++);
  }
}

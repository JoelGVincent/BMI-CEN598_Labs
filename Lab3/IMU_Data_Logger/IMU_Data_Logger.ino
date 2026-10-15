/*
  IMU_Data_Logger.ino
  Phase 1/2 of the lying-posture project.

  Idles until you type 's' + Enter into the Serial Monitor's input box,
  then streams accelerometer + gyroscope readings as CSV for a fixed
  recording window, prints an END marker, and goes back to idle. That
  gives each trial a clean, bounded block of output you can Clear +
  Save (Serial Monitor's export icon) into its own file, e.g.
  supine-trial1.csv, without having to hand-time a continuous stream.
*/

#include <Arduino_LSM9DS1.h>

// How often we take a sample and print a CSV row.
// Keeping this low also keeps each trial's row count small enough to
// select/copy by hand.
const unsigned long SAMPLE_INTERVAL_MS = 200;     // 5 Hz logging rate
const unsigned long RECORD_DURATION_MS = 20000;   // ~20s per trial = ~100 rows.
                                                   // Do 3 trials per posture (~60s total) to meet
                                                   // the  >=1 min per posture requirement.

enum State { IDLE,
             RECORDING };
State state = IDLE;
unsigned long recordStart = 0;
unsigned long lastSampleTime = 0;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);

  // Wait up to 3s for a Serial Monitor, but don't hang forever if none
  // connects (an indefinite `while (!Serial);` is a common cause of a
  // sketch that appears to produce zero output).
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) {}

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    // Fast-blink forever so failure is visible with no Monitor connected.
    while (1) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(100);
    }
  }

  Serial.print("# Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("# Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  printIdlePrompt();
}

void loop() {
  handleSerialCommand();

  if (state != RECORDING) {
    return;
  }

  unsigned long now = millis();

  if (now - recordStart >= RECORD_DURATION_MS) {
    Serial.println("# === END trial. Everything above the header down to here is your CSV -- delete this and the START line before saving. ===");
    digitalWrite(LED_BUILTIN, LOW);
    state = IDLE;
    printIdlePrompt();
    return;
  }

  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleTime = now;

  float ax, ay, az;
  float gx, gy, gz;

  // Only print once both a fresh accel and gyro sample are available,
  // so every row has matched, current readings.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    Serial.print(now);
    Serial.print(",");
    Serial.print(ax, 4);
    Serial.print(",");
    Serial.print(ay, 4);
    Serial.print(",");
    Serial.print(az, 4);
    Serial.print(",");
    Serial.print(gx, 4);
    Serial.print(",");
    Serial.print(gy, 4);
    Serial.print(",");
    Serial.println(gz, 4);
  }
}

void handleSerialCommand() {
  while (Serial.available()) {
    char c = Serial.read();
    if ((c == 's' || c == 'S') && state == IDLE) {
      state = RECORDING;
      recordStart = millis();
      lastSampleTime = 0;
      digitalWrite(LED_BUILTIN, HIGH);  // stays on for the whole recording window
      Serial.println("# === START trial (recording for ~20s) === -- delete this line before saving.");
      Serial.println("timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
    }
  }
}

void printIdlePrompt() {
  Serial.println();
  Serial.println("# Ready. Hold the board in the target posture, then type 's' + Enter above to start a ~20s recording.");
}

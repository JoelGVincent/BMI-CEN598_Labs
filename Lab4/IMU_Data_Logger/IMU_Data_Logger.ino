/*
  IMU_Data_Logger.ino
  Phase 1/2 of the Lab 4 sensor-agnostic posture project.

  Extends the Lab 3 logger with the magnetometer instrument -- Lab 4 needs all
  three sensors recorded together, since the trained model must work on a 3-channel (x,y,z) window from ANY
  ONE of them at deployment time. 
*/

#include <Arduino_LSM9DS1.h>

// How often we take a sample and print a CSV row. Posture is a held,
// near-static orientation, not fast motion, so a low logging rate is fine
// -- kept at 5 Hz to match Lab 2/3 and keep each trial's row count small.
const unsigned long SAMPLE_INTERVAL_MS = 200;     // 5 Hz logging rate
const unsigned long RECORD_DURATION_MS = 20000;   // ~20s per trial = ~100 rows.

enum State { IDLE,
             RECORDING };
State state = IDLE;
unsigned long recordStart = 0;
unsigned long lastSampleTime = 0;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);

  // Wait up to 3s for a Serial Monitor, but don't hang forever if none
  // connects.
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) {}

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
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
  Serial.print("# Magnetic field sample rate = ");
  Serial.print(IMU.magneticFieldSampleRate());
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
    Serial.println("# === END trial. Delete this and the START line before saving. ===");
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
  float mx, my, mz;

  // Only print once a fresh sample is available from all three sensors, so
  // every row has matched, current readings.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    IMU.readMagneticField(mx, my, mz);

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
    Serial.print(gz, 4);
    Serial.print(",");
    Serial.print(mx, 4);
    Serial.print(",");
    Serial.print(my, 4);
    Serial.print(",");
    Serial.println(mz, 4);
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
      Serial.println("timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mx_uT,my_uT,mz_uT");
    }
  }
}

void printIdlePrompt() {
  Serial.println();
  Serial.println("# Ready. Hold the board in the target posture, then type 's' + Enter above to start a ~20s recording.");
}

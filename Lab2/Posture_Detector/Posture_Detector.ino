/*
  Posture_Detector.ino
  part 4 of the lying-posture project: real-time posture classification
    - Blink once  -> supine
    - Blink twice -> prone
    - Blink 3x    -> side (left or right)
    - LED off     -> none of the above / not confident
*/

#include <Arduino_LSM9DS1.h>

// Threshold (g) the reading must be greater than to be a valid position
// according to collected data, we will set threshold to 0.7
const float GRAVITY_THRESHOLD = 0.7;

// this gyroscope threshold ensures that the board is still when selecting a state
// so that angular movement wont interfere with the readings
const float STILLNESS_GYRO_THRESHOLD = 40.0;

// ---------------- Timing ----------------
const unsigned long SAMPLE_INTERVAL_MS = 20;   // ~50 Hz raw sampling, feeds the smoothing filter
const unsigned long UPDATE_INTERVAL_MS = 500;  // how often we re-classify posture, using
                                                // whatever the (continuously live) buffer holds
const int SMOOTHING_WINDOW = 10;               // moving-average window (10 * 20ms = 200ms of data)

const unsigned long LED_ON_MS = 150;
const unsigned long LED_OFF_MS = 150;
const unsigned long BLINK_GROUP_PAUSE_MS = 1000;  // 1 second delay between blinking groups to identify how many blinks are being displayed

const int LED_PIN = LED_BUILTIN; 

enum Posture { SUPINE,
               PRONE,
               SIDE,
               UNKNOWN };

float axBuf[SMOOTHING_WINDOW] = { 0 };
float ayBuf[SMOOTHING_WINDOW] = { 0 };
float azBuf[SMOOTHING_WINDOW] = { 0 };
int bufIndex = 0;
int bufCount = 0;

unsigned long lastSampleTime = 0;
unsigned long lastUpdateTime = 0;

volatile Posture latestPosture = UNKNOWN;

// ---------------- LED state machine (non-blocking) ----------------
enum LedState { LED_STARTING,
                LED_ON_PHASE,
                LED_OFF_PHASE,
                LED_GROUP_PAUSE };
LedState ledState = LED_STARTING;
unsigned long ledPhaseStart = 0;
int ledBlinksTotal = 0;
int ledBlinksDone = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }
}

void loop() {
  unsigned long now = millis();

  sampleIMU(now);       // always runs, every SAMPLE_INTERVAL_MS -- never blocked
  classifyIfDue(now);    // always runs, every UPDATE_INTERVAL_MS -- never blocked
  serviceLED(now);       // advances the blink/pause state machine -- no delay() calls
}

// 1. Keep a rolling average of recent accelerometer samples (cheap noise filter).
// Runs on every loop() pass regardless of what the LED is doing.
void sampleIMU(unsigned long now) {
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  float ax, ay, az;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    axBuf[bufIndex] = ax;
    ayBuf[bufIndex] = ay;
    azBuf[bufIndex] = az;
    bufIndex = (bufIndex + 1) % SMOOTHING_WINDOW;
    if (bufCount < SMOOTHING_WINDOW) bufCount++;
  }
}

// 2. Periodically classify posture from the (always up-to-date) buffer.
// Also runs on every loop() pass regardless of what the LED is doing.
void classifyIfDue(unsigned long now) {
  if (now - lastUpdateTime < UPDATE_INTERVAL_MS || bufCount != SMOOTHING_WINDOW) return;
  lastUpdateTime = now;

  float axAvg = average(axBuf, SMOOTHING_WINDOW);
  float ayAvg = average(ayBuf, SMOOTHING_WINDOW);
  float azAvg = average(azBuf, SMOOTHING_WINDOW);

  float gx, gy, gz;
  float gyroMag = 0;
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
    gyroMag = sqrt(gx * gx + gy * gy + gz * gz);
  }

  Posture p = UNKNOWN;
  if (gyroMag < STILLNESS_GYRO_THRESHOLD) {
    p = classifyPosture(axAvg, ayAvg, azAvg);
  }

  Serial.print("ax=");
  Serial.print(axAvg, 2);
  Serial.print(" ay=");
  Serial.print(ayAvg, 2);
  Serial.print(" az=");
  Serial.print(azAvg, 2);
  Serial.print(" gyroMag=");
  Serial.print(gyroMag, 1);
  Serial.print(" -> ");
  Serial.println(postureName(p));

  latestPosture = p;
}

// ---------------- Posture classification ----------------
// Axis mapping below is empirical, from IMU_Data_Collection trial data with
// this specific board/mounting convention (component side facing outward
// from the chest, USB-C toward the head):
//   supine     -> az ~= +1.0
//   prone      -> az ~= -1.0
//   side-left  -> ay ~= +1.0
//   side-right -> ay ~= -1.0
// If your mounting convention differs, re-derive this from your own plots.
Posture classifyPosture(float ax, float ay, float az) {
  if (az > GRAVITY_THRESHOLD) {
    return SUPINE;  // chest facing up -> Z axis points away from earth
  } else if (az < -GRAVITY_THRESHOLD) {
    return PRONE;  // chest facing down -> Z axis points toward earth
  } else if (fabs(ay) > GRAVITY_THRESHOLD) {
    return SIDE;  // chest facing sideways -- sign of ay tells you left vs. right
  }
  return UNKNOWN;
}

float average(float *buf, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) sum += buf[i];
  return sum / n;
}

const char *postureName(Posture p) {
  switch (p) {
    case SUPINE: return "SUPINE";
    case PRONE: return "PRONE";
    case SIDE: return "SIDE";
    default: return "UNKNOWN";
  }
}

// ---------------- LED feedback (non-blocking state machine) ----------------
// Plays blink groups back-to-back forever: [blink pattern] -> [1s dark pause] -> repeat.
// Each time it's ready to start a new group, it picks up whatever `latestPosture`
// currently is -- which is always updated, since classifyIfDue() above runs on every
// loop() pass and is never blocked by this state machine.
void serviceLED(unsigned long now) {
  switch (ledState) {
    case LED_STARTING:
      startBlinkGroup(now);
      break;

    case LED_ON_PHASE:
      if (now - ledPhaseStart >= LED_ON_MS) {
        digitalWrite(LED_PIN, LOW);
        ledPhaseStart = now;
        ledBlinksDone++;
        ledState = (ledBlinksDone >= ledBlinksTotal) ? LED_GROUP_PAUSE : LED_OFF_PHASE;
      }
      break;

    case LED_OFF_PHASE:
      if (now - ledPhaseStart >= LED_OFF_MS) {
        digitalWrite(LED_PIN, HIGH);
        ledPhaseStart = now;
        ledState = LED_ON_PHASE;
      }
      break;

    case LED_GROUP_PAUSE:
      if (now - ledPhaseStart >= BLINK_GROUP_PAUSE_MS) {
        ledState = LED_STARTING;
      }
      break;
  }
}

void startBlinkGroup(unsigned long now) {
  ledBlinksTotal = blinkCountFor(latestPosture);
  ledBlinksDone = 0;

  if (ledBlinksTotal == 0) {
    digitalWrite(LED_PIN, LOW);
    ledPhaseStart = now;
    ledState = LED_GROUP_PAUSE;  // still pause, so the update cadence stays consistent
  } else {
    digitalWrite(LED_PIN, HIGH);
    ledPhaseStart = now;
    ledState = LED_ON_PHASE;
  }
}

int blinkCountFor(Posture p) {
  switch (p) {
    case SUPINE: return 1;
    case PRONE: return 2;
    case SIDE: return 3;
    default: return 0;
  }
}

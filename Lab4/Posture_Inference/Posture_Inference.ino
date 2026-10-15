/*
  Posture_Inference.ino
  Phases 7-8 of the Lab 4 sensor-agnostic posture project: on-device
  real-time inference, driven by requests from a base-station app.

  The microcontroller does not predict continuously. It continuously
  buffers the last ~3s of raw IMU samples in the background (so a fresh
  window is always ready), but only runs the model and reports a result
  when it receives a request over Serial. The base station sends a single
  digit specifying which sensor to use for that prediction
    '1' = accelerometer, '2' = gyroscope, '3' = magnetometer
  and the board replies with one line:
    PREDICTION sensor=<ACCEL|GYRO|MAG> class=<name> confidence=<0-1>
*/

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_data.h"

#include <Arduino_LSM9DS1.h>

// These must match what the Colab notebook used to build the training
// data, or the model will see input that doesn't match what it learned.
const int WINDOW_LEN = 15;      // samples per channel (~3.0s at 5 Hz)
const int N_RAW_CHANNELS = 9;   // ax,ay,az,gx,gy,gz,mx,my,mz
const int N_CLASSES = 5;
const unsigned long SAMPLE_INTERVAL_MS = 200;  // 5 Hz -- matches training's logging rate

const char *CLASS_NAMES[N_CLASSES] = { "SUPINE", "PRONE", "SIDE", "SITTING", "UNKNOWN" };

// Which 3 of the 9 raw channels each sensor selection reads from (matches
// SENSOR_CHANNEL_IDX in the notebook).
const int ACCEL_CHANNEL_START = 0;  // ax, ay, az
const int GYRO_CHANNEL_START = 3;   // gx, gy, gz
const int MAG_CHANNEL_START = 6;    // mx, my, mz

// Per-channel normalization, computed from TRAINING data only (printed by
// the notebook's "Export as a C header" cell) -- order: ax,ay,az,gx,gy,gz,mx,my,mz.
const float CHANNEL_MEAN[9] = { -0.110877f, 0.010027f, 0.020532f, 4.610532f, 4.010490f, 0.354839f, 1.637123f, 28.107430f, -125.581444f };
const float CHANNEL_STD[9] = { 0.566604f, 0.485196f, 0.662110f, 62.144825f, 36.580048f, 39.981728f, 17.847170f, 18.614256f, 21.648241f };

// Sampling never stops
float rawBuffer[N_RAW_CHANNELS][WINDOW_LEN] = { 0 };
int bufIndex = 0;   // next slot to write; also the OLDEST sample once bufCount == WINDOW_LEN
int bufCount = 0;
unsigned long lastSampleTime = 0;

// ---------------- TensorFlow Lite Micro globals ----------------
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

// Static memory buffer TFLM uses for the model's tensors.
constexpr int tensorArenaSize = 20 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

void setup() {
  Serial.begin(115200);
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) {}

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1) {}
  }

  // ---- Load and validate the model ----
  tflModel = tflite::GetModel(g_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1) {}
  }

  static tflite::MicroInterpreter staticInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter = &staticInterpreter;

  TfLiteStatus allocateStatus = tflInterpreter->AllocateTensors();
  if (allocateStatus != kTfLiteOk) {
    Serial.println("AllocateTensors() failed! Try increasing tensorArenaSize.");
    while (1) {}
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  printIdlePrompt();
}

void loop() {
  unsigned long now = millis();
  sampleIMU(now);              // always runs -- keeps the window buffer fresh
  handlePredictionRequest();   // only acts when the base station sends a request
}

// Continuously buffer the last WINDOW_LEN samples of all 9 raw channels.
void sampleIMU(unsigned long now) {
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  float ax, ay, az, gx, gy, gz, mx, my, mz;
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    IMU.readMagneticField(mx, my, mz);

    float sample[N_RAW_CHANNELS] = { ax, ay, az, gx, gy, gz, mx, my, mz };
    for (int c = 0; c < N_RAW_CHANNELS; c++) {
      rawBuffer[c][bufIndex] = sample[c];
    }
    bufIndex = (bufIndex + 1) % WINDOW_LEN;
    if (bufCount < WINDOW_LEN) bufCount++;
  }
}

// Waits for a base-station request ('1'/'2'/'3')
void handlePredictionRequest() {
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c != '1' && c != '2' && c != '3') return;  // ignore newlines/anything else

  if (bufCount < WINDOW_LEN) {
    Serial.println("ERROR: window buffer not full yet, wait a few seconds and retry.");
    return;
  }

  int channelStart;
  const char *sensorName;
  if (c == '1') {
    channelStart = ACCEL_CHANNEL_START;
    sensorName = "ACCEL";
  } else if (c == '2') {
    channelStart = GYRO_CHANNEL_START;
    sensorName = "GYRO";
  } else {
    channelStart = MAG_CHANNEL_START;
    sensorName = "MAG";
  }

  fillNormalizedWindow(channelStart, tflInputTensor->data.f);

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("ERROR: model Invoke() failed.");
    return;
  }

  int bestClass = 0;
  float bestProb = tflOutputTensor->data.f[0];
  for (int i = 1; i < N_CLASSES; i++) {
    if (tflOutputTensor->data.f[i] > bestProb) {
      bestProb = tflOutputTensor->data.f[i];
      bestClass = i;
    }
  }

  Serial.print("PREDICTION sensor=");
  Serial.print(sensorName);
  Serial.print(" class=");
  Serial.print(CLASS_NAMES[bestClass]);
  Serial.print(" confidence=");
  Serial.println(bestProb, 3);
}

// Reads the requested sensor's 3 channels out of the circular buffer in
// chronological order, normalizes each value with its
// channel's training-derived (mean, std), and writes them into `out` in
// the SAME [t0_x,t0_y,t0_z, t1_x,t1_y,t1_z, ...] order
void fillNormalizedWindow(int channelStart, float *out) {
  for (int t = 0; t < WINDOW_LEN; t++) {
    int srcIdx = (bufIndex + t) % WINDOW_LEN;  // bufIndex is the oldest sample's slot
    for (int c = 0; c < 3; c++) {
      int channel = channelStart + c;
      float raw = rawBuffer[channel][srcIdx];
      out[t * 3 + c] = (raw - CHANNEL_MEAN[channel]) / CHANNEL_STD[channel];
    }
  }
}

void printIdlePrompt() {
  Serial.println();
  Serial.println("Ready. Send '1' (accel), '2' (gyro), or '3' (mag) to request a prediction.");
}

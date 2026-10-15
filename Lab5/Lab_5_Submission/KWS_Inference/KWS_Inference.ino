/*
  KWS_Inference.ino
  Real-time absolutist-keyword-spotting inference for Lab 5, driven by
  requests from a base-station app over serial.

  The board continuously buffers the last ~1.25s of raw audio from its
  onboard PDM microphone in the background. 

  The FFT here is a small, self-contained radix-2 implementation rather
  than the kissfft library this TensorFlow Lite Micro package bundles.
*/

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_data.h"

#include <PDM.h>
#include <math.h>

// Audio / feature constants 
// These must match what the training notebook used, or the model will see
// input that doesn't match what it learned.
const int SAMPLE_RATE = 16000;
const int CLIP_LEN = 16000;     // 1 second window
const int FRAME_LENGTH = 256;   // also the FFT size, already a power of 2
const int FRAME_STEP = 400;
const int NUM_FRAMES = 40;
const int NUM_BINS = 65;        // cropped to 0-4 kHz of the full 0-8 kHz range
const int N_CLASSES = 7;

const char *CLASS_NAMES[N_CLASSES] = {
  "_silence_", "_unknown_", "all", "must", "never", "none", "only"
};

// Training-derived normalization constants, printed by the notebook's
// "Export as a C header" cell, applied identically here.
const float SPEC_MEAN = -2.331846f;
const float SPEC_STD = 2.627883f;

// Audio capture (continuous, background) 
// A bit more than the 1s window we actually use, so there is margin
// between the write pointer and the oldest sample we still need while
// feature computation is running. Kept modest since RAM is tight.
const int CAPTURE_BUFFER_LEN = 17000;
volatile int16_t captureBuffer[CAPTURE_BUFFER_LEN];
volatile int captureBufferIndex = 0;
volatile long totalSamplesCaptured = 0;
volatile bool bufferFilled = false;

short pdmSampleBuffer[512];

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(pdmSampleBuffer, bytesAvailable);
  int numSamples = bytesAvailable / 2;
  for (int i = 0; i < numSamples; i++) {
    captureBuffer[captureBufferIndex] = pdmSampleBuffer[i];
    captureBufferIndex = (captureBufferIndex + 1) % CAPTURE_BUFFER_LEN;
  }
  totalSamplesCaptured += numSamples;
  if (totalSamplesCaptured >= CLIP_LEN) bufferFilled = true;
}

// FFT (radix-2 decimation-in-time, N=256)
const int FFT_N = FRAME_LENGTH;
const int FFT_LOG2N = 8;

float twiddleCos[FFT_N / 2];
float twiddleSin[FFT_N / 2];
float hannWindow[FRAME_LENGTH];
float fftRe[FFT_N];
float fftIm[FFT_N];

void initDSPTables() {
  for (int k = 0; k < FFT_N / 2; k++) {
    float angle = -2.0f * PI * k / FFT_N;
    twiddleCos[k] = cosf(angle);
    twiddleSin[k] = sinf(angle);
  }
  // Periodic Hann window (divides by N, not N-1), matches tf.signal.stft's
  // default window_fn.
  for (int n = 0; n < FRAME_LENGTH; n++) {
    hannWindow[n] = 0.5f - 0.5f * cosf(2.0f * PI * n / FRAME_LENGTH);
  }
}

void fft256(float *re, float *im) {
  int j = 0;
  for (int i = 0; i < FFT_N - 1; i++) {
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
    int m = FFT_N >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (int stage = 1; stage <= FFT_LOG2N; stage++) {
    int m = 1 << stage;
    int mHalf = m >> 1;
    int twiddleStep = FFT_N / m;
    for (int k = 0; k < FFT_N; k += m) {
      for (int n = 0; n < mHalf; n++) {
        int tIdx = n * twiddleStep;
        float wr = twiddleCos[tIdx];
        float wi = twiddleSin[tIdx];
        int a = k + n;
        int b = k + n + mHalf;
        float br = re[b] * wr - im[b] * wi;
        float bi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - br;
        im[b] = im[a] - bi;
        re[a] = re[a] + br;
        im[a] = im[a] + bi;
      }
    }
  }
}

// ---------------- Feature extraction ----------------
// Computes the (NUM_FRAMES x NUM_BINS) normalized log-magnitude spectrogram
// of the most recent 1 second of buffered audio and writes it into
// modelInput in [frame][bin] row-major order, matching the notebook's
// load_and_pad -> get_spectrogram -> normalize pipeline exactly.
void computeFeatures(float *modelInput) {
  noInterrupts();
  int writeIndex = captureBufferIndex;
  interrupts();
  int windowStart = (writeIndex - CLIP_LEN + CAPTURE_BUFFER_LEN) % CAPTURE_BUFFER_LEN;

  // Peak-normalize over the whole 1s window first, same as load_and_pad
  // did in training.
  int16_t peak = 1;
  for (int i = 0; i < CLIP_LEN; i++) {
    int idx = (windowStart + i) % CAPTURE_BUFFER_LEN;
    int16_t s = captureBuffer[idx];
    int16_t a = s < 0 ? (int16_t)(-s) : s;
    if (a > peak) peak = a;
  }

  for (int f = 0; f < NUM_FRAMES; f++) {
    int frameStart = f * FRAME_STEP;
    for (int n = 0; n < FFT_N; n++) {
      int idx = (windowStart + frameStart + n) % CAPTURE_BUFFER_LEN;
      float sample = (float)captureBuffer[idx] / (float)peak;
      fftRe[n] = sample * hannWindow[n];
      fftIm[n] = 0.0f;
    }
    fft256(fftRe, fftIm);
    for (int b = 0; b < NUM_BINS; b++) {
      float mag = sqrtf(fftRe[b] * fftRe[b] + fftIm[b] * fftIm[b]);
      float logmag = logf(mag + 1e-6f);
      modelInput[f * NUM_BINS + b] = (logmag - SPEC_MEAN) / SPEC_STD;
    }
  }
}

// TensorFlow Lite Micro globals
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

// The first compile with a generous 150 KB arena succeeded, confirming
// that's more than enough, but it also left only ~11 KB of RAM for the
// stack
constexpr int tensorArenaSize = 100 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

void setup() {
  Serial.begin(115200);
  unsigned long waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) {}

  initDSPTables();

  // onReceive() must be registered before begin(), which starts capturing immediately.
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Failed to initialize PDM microphone!");
    while (1) {}
  }
  PDM.setGain(13);

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
  handlePredictionRequest();
}

// Waits for a base-station request ('p'). There is only one microphone
// here, unlike Lab 4's sensor-selection protocol, so a single trigger
// character is enough.
void handlePredictionRequest() {
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c != 'p') return;

  if (!bufferFilled) {
    Serial.println("ERROR: audio buffer not full yet, wait a moment and retry.");
    return;
  }

  computeFeatures(tflInputTensor->data.f);

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

  Serial.print("PREDICTION class=");
  Serial.print(CLASS_NAMES[bestClass]);
  Serial.print(" confidence=");
  Serial.println(bestProb, 3);
}

void printIdlePrompt() {
  Serial.println();
  Serial.println("Ready. Send 'p' to request a prediction.");
}

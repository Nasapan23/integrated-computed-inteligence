#include <Arduino.h>
#include <ESP_I2S.h>
#include <math.h>

#include "../../yes_no_model.h"
#include "../../yes_no_model.cc"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define SAMPLE_RATE     16000
#define NUM_SAMPLES     16000
#define FRAME_LENGTH    255
#define FRAME_STEP      128
#define NUM_FRAMES      124
#define NUM_FREQ_BINS   128
#define FFT_LENGTH      256
#define ARENA_SIZE      (200 * 1024)

static constexpr int MIC_CLK_PIN = 42;
static constexpr int MIC_DATA_PIN = 41;
static constexpr int NUM_CLASSES = 2;
static constexpr float CONFIDENCE_THRESHOLD = 70.0f;
static constexpr float kPi = 3.14159265358979323846f;

uint8_t* tensor_arena = nullptr;
int16_t* audio_buffer = nullptr;
float* spectrogram = nullptr;
float* hann_window = nullptr;
float* twiddle_cos = nullptr;
float* twiddle_sin = nullptr;

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;

static tflite::MicroMutableOpResolver<11> resolver;
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static I2SClass i2s;

const char* LABELS[] = {"no", "yes"};

void halt(const char* message) {
  Serial.println(message);
  while (true) {
    delay(1000);
  }
}

void printMemoryStatus() {
  Serial.printf("Free heap: %u bytes\n", ESP.getFreeHeap());
  Serial.printf("Total PSRAM: %u bytes\n", ESP.getPsramSize());
  Serial.printf("Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

void allocateBuffers() {
  tensor_arena = static_cast<uint8_t*>(ps_malloc(ARENA_SIZE));
  audio_buffer = static_cast<int16_t*>(ps_malloc(NUM_SAMPLES * sizeof(int16_t)));
  spectrogram = static_cast<float*>(ps_malloc(NUM_FRAMES * NUM_FREQ_BINS * sizeof(float)));
  hann_window = static_cast<float*>(ps_malloc(FRAME_LENGTH * sizeof(float)));
  twiddle_cos = static_cast<float*>(ps_malloc(NUM_FREQ_BINS * FRAME_LENGTH * sizeof(float)));
  twiddle_sin = static_cast<float*>(ps_malloc(NUM_FREQ_BINS * FRAME_LENGTH * sizeof(float)));

  if (!tensor_arena || !audio_buffer || !spectrogram || !hann_window || !twiddle_cos || !twiddle_sin) {
    Serial.println("ERROR: PSRAM allocation failed!");
    while (1) {
      delay(1000);
    }
  }
}

void prepareDftTables() {
  for (int n = 0; n < FRAME_LENGTH; ++n) {
    hann_window[n] = 0.5f - 0.5f * cosf((2.0f * kPi * static_cast<float>(n)) / static_cast<float>(FRAME_LENGTH - 1));
  }

  for (int k = 0; k < NUM_FREQ_BINS; ++k) {
    for (int n = 0; n < FRAME_LENGTH; ++n) {
      const float angle = (2.0f * kPi * static_cast<float>(k) * static_cast<float>(n)) / static_cast<float>(FFT_LENGTH);
      const int index = k * FRAME_LENGTH + n;
      twiddle_cos[index] = cosf(angle);
      twiddle_sin[index] = sinf(angle);
    }
  }
}

void initMicrophone() {
  i2s.setPinsPdmRx(MIC_CLK_PIN, MIC_DATA_PIN);
  if (!i2s.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    halt("ERROR: I2S microphone initialization failed!");
  }

  delay(100);
  Serial.printf("I2S RX sample rate: %u Hz\n", i2s.rxSampleRate());
}

void initModel() {
  tfl_model = tflite::GetModel(yes_no_model_quant_tflite);
  if (tfl_model == nullptr) {
    halt("ERROR: Failed to map TFLite model.");
  }

  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    halt("ERROR: TFLite schema version mismatch.");
  }

  if (resolver.AddQuantize() != kTfLiteOk ||
      resolver.AddResizeBilinear() != kTfLiteOk ||
      resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddShape() != kTfLiteOk ||
      resolver.AddStridedSlice() != kTfLiteOk ||
      resolver.AddPack() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk ||
      resolver.AddDequantize() != kTfLiteOk) {
    halt("ERROR: Failed to register TensorFlow Lite Micro ops.");
  }

  static tflite::MicroInterpreter static_interpreter(
      tfl_model,
      resolver,
      tensor_arena,
      ARENA_SIZE,
      error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    halt("ERROR: Tensor allocation failed.");
  }

  input_tensor = interpreter->input(0);
  if (input_tensor == nullptr) {
    halt("ERROR: Model input tensor is null.");
  }

  Serial.printf("Input tensor type: %d\n", input_tensor->type);
  Serial.printf("Input bytes: %d\n", input_tensor->bytes);
}

void countdown(int seconds) {
  for (int i = seconds; i > 0; i--) {
    Serial.printf("Recording starts in %d...\n", i);
    delay(1000);
  }
  Serial.println("Recording now!");
}

void captureAudio() {
  size_t total = 0;
  char* ptr = reinterpret_cast<char*>(audio_buffer);
  const size_t target = NUM_SAMPLES * sizeof(int16_t);

  while (total < target) {
    const size_t bytes_read = i2s.readBytes(ptr + total, target - total);
    if (bytes_read == 0) {
      delay(1);
      continue;
    }
    total += bytes_read;
  }
}

void computeSpectrogram() {
  for (int frame = 0; frame < NUM_FRAMES; ++frame) {
    const int start = frame * FRAME_STEP;

    for (int k = 0; k < NUM_FREQ_BINS; ++k) {
      float real = 0.0f;
      float imag = 0.0f;

      for (int n = 0; n < FRAME_LENGTH; ++n) {
        const float sample = (static_cast<float>(audio_buffer[start + n]) / 32768.0f) * hann_window[n];
        const int index = k * FRAME_LENGTH + n;
        real += sample * twiddle_cos[index];
        imag -= sample * twiddle_sin[index];
      }

      spectrogram[frame * NUM_FREQ_BINS + k] = sqrtf(real * real + imag * imag);
    }
  }
}

void fillInputTensor() {
  const int total_bins = NUM_FRAMES * NUM_FREQ_BINS;

  if (input_tensor->type == kTfLiteFloat32) {
    for (int i = 0; i < total_bins; ++i) {
      input_tensor->data.f[i] = spectrogram[i];
    }
    return;
  }

  if (input_tensor->type == kTfLiteInt8) {
    const float scale = input_tensor->params.scale;
    const int zero_point = input_tensor->params.zero_point;

    for (int i = 0; i < total_bins; ++i) {
      const int32_t quantized = static_cast<int32_t>(roundf(spectrogram[i] / scale)) + zero_point;
      input_tensor->data.int8[i] = static_cast<int8_t>(constrain(quantized, -128, 127));
    }
    return;
  }

  halt("ERROR: Unsupported input tensor type.");
}

void softmax(float* input, float* output, int len) {
  float max_val = input[0];
  for (int i = 1; i < len; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < len; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  for (int i = 0; i < len; i++) {
    output[i] /= sum;
  }
}

void normalizeProbabilities(const float* raw, float* prob, int len) {
  bool already_probabilities = true;
  float sum = 0.0f;

  for (int i = 0; i < len; ++i) {
    if (raw[i] < -0.01f || raw[i] > 1.01f) {
      already_probabilities = false;
    }
    sum += raw[i];
  }

  if (already_probabilities && fabsf(sum - 1.0f) < 0.05f) {
    for (int i = 0; i < len; ++i) {
      prob[i] = raw[i] / sum;
    }
    return;
  }

  float mutable_raw[NUM_CLASSES];
  for (int i = 0; i < len; ++i) {
    mutable_raw[i] = raw[i];
  }
  softmax(mutable_raw, prob, len);
}

void runInference() {
  fillInputTensor();

  if (interpreter->Invoke() != kTfLiteOk) {
    halt("ERROR: Interpreter invoke failed.");
  }

  TfLiteTensor* output_tensor = interpreter->output(0);
  if (output_tensor == nullptr) {
    halt("ERROR: Output tensor is null.");
  }

  float raw[NUM_CLASSES];
  float prob[NUM_CLASSES];

  if (output_tensor->type == kTfLiteInt8) {
    const int8_t q_no = output_tensor->data.int8[0];
    const int8_t q_yes = output_tensor->data.int8[1];
    const float scale = output_tensor->params.scale;
    const int zero = output_tensor->params.zero_point;

    raw[0] = (q_no - zero) * scale;
    raw[1] = (q_yes - zero) * scale;
  } else if (output_tensor->type == kTfLiteFloat32) {
    raw[0] = output_tensor->data.f[0];
    raw[1] = output_tensor->data.f[1];
  } else {
    halt("ERROR: Unsupported output tensor type.");
  }

  normalizeProbabilities(raw, prob, NUM_CLASSES);

  const float score_no = prob[0];
  const float score_yes = prob[1];
  const int predicted = (score_yes > score_no) ? 1 : 0;
  const float confidence = max(score_no, score_yes) * 100.0f;

  Serial.printf("no: %.1f%% yes: %.1f%% -> %s\n", score_no * 100.0f, score_yes * 100.0f, LABELS[predicted]);

  if (confidence > CONFIDENCE_THRESHOLD) {
    Serial.printf("Detected: %s (%.1f%%)\n", LABELS[predicted], confidence);
  } else {
    Serial.println("(uncertain)");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("XIAO ESP32S3 Sense yes/no recognizer");

  printMemoryStatus();
  if (!psramFound()) {
    halt("ERROR: PSRAM not detected. Enable OPI PSRAM in board settings.");
  }

  allocateBuffers();
  prepareDftTables();
  initMicrophone();
  initModel();
  printMemoryStatus();
}

void loop() {
  countdown(3);
  captureAudio();
  computeSpectrogram();
  runInference();
  Serial.println();
  delay(1000);
}

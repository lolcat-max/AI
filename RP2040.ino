#include <Arduino.h>
// Advanced Curvature Analysis Oscilloscope for Adafruit Feather RP2040
// Specialized for signal pattern analysis with periodic static discharge
// Implements locally convex topological vector space analysis + charge management
#include <math.h>
int bet = 0;
int count = 0;
// --- RP2040 SPECIFIC CHANGES ---
#define NUM_CHANNELS 4       // RP2040 has 4 analog inputs (A0-A3)
#define MATRIX_SIZE 4        // Matrix size must match the number of channels
// --- END RP2040 CHANGES ---

#define SAMPLE_RATE_MS 50   // 50Hz sampling
#define HISTORY_SIZE 20    // Circular buffer for signal history
#define SMOOTHING_WINDOW 7
#define STATE_SPACE 999999
#define FRECHET_THRESHOLD 0.0003 // Convergence threshold for Cauchy sequences

// Charge discharge settings
#define DISCHARGE_INTERVAL_MS 500000 // Discharge every 5 seconds
#define DISCHARGE_PIN 13           // GPIO pin for discharge (Feather RP2040 has a red LED on pin 13)
#define BASELINE_SAMPLES 10        // Samples for baseline calibration
#define CHARGE_THRESHOLD 0.1       // Voltage drift threshold for auto-discharge

// --- RP2040 SPECIFIC CHANGES ---
// Analog pins for Feather RP2040 are A0, A1, A2, A3
const int analogPins[NUM_CHANNELS] = {A0, A1, A2, A3};
// --- END RP2040 CHANGES ---

// Signal data structures
float channelHistory[NUM_CHANNELS][HISTORY_SIZE];
float smoothedData[NUM_CHANNELS][HISTORY_SIZE];
float curvatureData[HISTORY_SIZE];
float baselineVoltages[NUM_CHANNELS];   // Baseline reference voltages
float chargeAccumulation[NUM_CHANNELS]; // Track charge buildup per channel

// Fréchet space matrices
float currentMatrix[MATRIX_SIZE][MATRIX_SIZE];
float curvatureMatrix[MATRIX_SIZE][MATRIX_SIZE];
float metricMatrix[MATRIX_SIZE][MATRIX_SIZE];
float seminormVector[MATRIX_SIZE];
float cauchySequence[MATRIX_SIZE];

// Analysis variables
int historyIndex = 0;
unsigned long lastSample = 0;
unsigned long lastDischarge = 0;
int selectedChannel = 0;
unsigned long analysisCount = 0;
int flipCount = 0;
float frechetMetric = 0.0;
bool isLocallyConvex = false;
bool chargeDetected = false;

void setup();
void loop();
void establishBaseline();
void detectChargeAccumulation();
void initializeMatrices();
void handleSerialCommands();
void displayChargeStatus();
void sampleAllChannels();
void applySmoothingFilter();
void updateCurrentMatrix();
float calculateCorrelation(int ch1, int ch2);
void performCurvatureMatrixAnalysis();
void calculateFrechetMetrics();
void updateCauchySequence();
void detectSignFlips();
void performAdvancedMatrixAnalysis();
void displayMatrixOscilloscope();
void displayEnhancedMatrixStats();
void displayCurrentMatrix();
void displayCurvatureMatrix();
void displayFrechetAnalysis();
float getCurrentValue(int channel);
float getPeakValue(int channel);
float getRMSValue(int channel);
float getAverageCurvature();
void resetAnalysis();

void setup() {
  Serial.begin(115200);

  // Configure discharge pin
  pinMode(DISCHARGE_PIN, OUTPUT);
  digitalWrite(DISCHARGE_PIN, LOW);


  // Initialize all data structures
  initializeMatrices();

  // Establish baseline voltages
  establishBaseline();

  Serial.println("FRÉCHET SPACE CURVATURE OSCILLOSCOPE (RP2040 FEATHER)");
  Serial.println("======================================================");
  // --- RP2040 SPECIFIC CHANGE ---
  Serial.println("Commands: 0-3 (channel), r (reset), c (curvature), f (Fréchet)");
  // --- END RP2040 CHANGE ---
  Serial.println("         m (matrix), d (manual discharge), b (recalibrate)");
  Serial.println("Auto-discharge every 5s | Charge monitoring enabled");
  Serial.println("======================================================");
}

void loop() {
  unsigned long currentTime = millis();

  handleSerialCommands();

 

  if (currentTime - lastSample >= SAMPLE_RATE_MS) {
    // Core sampling and analysis pipeline
    sampleAllChannels();
    detectChargeAccumulation();
    applySmoothingFilter();
    updateCurrentMatrix();
    performCurvatureMatrixAnalysis();
    calculateFrechetMetrics();
    detectSignFlips();
    displayMatrixOscilloscope();
    performAdvancedMatrixAnalysis();
    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    analysisCount++;
    lastSample = currentTime;
  }
}


void establishBaseline() {
  Serial.println("■ ESTABLISHING BASELINE VOLTAGES...");
  delay(100);
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum = 0;
    for (int i = 0; i < BASELINE_SAMPLES; i++) {
      int rawValue = analogRead(analogPins[ch]);
      sum += (rawValue / 4095.0) * 3.3;
      delay(10);
    }
    baselineVoltages[ch] = sum / BASELINE_SAMPLES;
    chargeAccumulation[ch] = 0.0;
  }
  Serial.println("■ BASELINE ESTABLISHED");
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    Serial.print("CH");
    Serial.print(ch);
    Serial.print(": ");
    Serial.print(baselineVoltages[ch], 4);
    Serial.println("V");
  }
}


void detectChargeAccumulation() {
  chargeDetected = false;
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float currentVoltage = channelHistory[ch][historyIndex];
    float drift = abs(currentVoltage - baselineVoltages[ch]);
    chargeAccumulation[ch] = 0.9 * chargeAccumulation[ch] + 0.1 * drift;
    if (chargeAccumulation[ch] > CHARGE_THRESHOLD) {
      chargeDetected = true;
    }
  }
}

void initializeMatrices() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    for (int i = 0; i < HISTORY_SIZE; i++) {
      channelHistory[ch][i] = 0.0;
      smoothedData[ch][i] = 0.0;
    }
    chargeAccumulation[ch] = 0.0;
  }
  for (int i = 0; i < MATRIX_SIZE; i++) {
    seminormVector[i] = 1.0;
    cauchySequence[i] = 1.0;
    for (int j = 0; j < MATRIX_SIZE; j++) {
      currentMatrix[i][j] = 1.0;
      curvatureMatrix[i][j] = 1.0;
      metricMatrix[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (int i = 0; i < HISTORY_SIZE; i++) {
    curvatureData[i] = 0.0;
  }
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd >= '0' && cmd < ('0' + NUM_CHANNELS)) {
      selectedChannel = cmd - '0';
      Serial.print("■ SELECTED CHANNEL ");
      Serial.println(selectedChannel);
    } else if (cmd == 'r') {
      resetAnalysis();
      Serial.println("■ ANALYSIS RESET");
    } else if (cmd == 'c') {
      displayCurvatureMatrix();
    } else if (cmd == 'f') {
      displayFrechetAnalysis();
    } else if (cmd == 'm') {
      displayCurrentMatrix();
    } else if (cmd == 'b') {
      establishBaseline();
      Serial.println("■ BASELINE RECALIBRATED");
    } else if (cmd == 's') {
      displayChargeStatus();
    }
  }
}

void displayChargeStatus() {
  Serial.println("████ CHARGE ACCUMULATION STATUS ████");
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    Serial.print("CH");
    Serial.print(ch);
    Serial.print(": Baseline=");
    Serial.print(baselineVoltages[ch], 4);
    Serial.print("V Current=");
    Serial.print(channelHistory[ch][historyIndex], 4);
    Serial.print("V Charge=");
    Serial.print(chargeAccumulation[ch], 4);
    Serial.println();
  }
  Serial.print("Charge Detected: ");
  Serial.println(chargeDetected ? "YES" : "NO");
  Serial.println("██████████████████████████████████");
}

void sampleAllChannels() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    int rawValue = analogRead(analogPins[ch]);
    channelHistory[ch][historyIndex] = (rawValue / 4095.0) * 3.3;
    if (historyIndex > 0) {
      float diff = abs(channelHistory[ch][historyIndex] - channelHistory[ch][(historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE]);
      float threshold = 0.5 + chargeAccumulation[ch] * 2.0;
      if (diff > threshold) {
        channelHistory[ch][historyIndex] = (channelHistory[ch][historyIndex] + channelHistory[ch][(historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE]) / 2.0;
      }
    }
  }
}

void applySmoothingFilter() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum = 0;
    int count = 0;
    for (int i = 0; i < SMOOTHING_WINDOW && i < HISTORY_SIZE; i++) {
      int idx = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
      sum += channelHistory[ch][idx];
      count++;
    }
    if (count > 0) {
      smoothedData[ch][historyIndex] = sum / count;
    }
  }
}

void updateCurrentMatrix() {
  for (int i = 0; i < NUM_CHANNELS && i < MATRIX_SIZE; i++) {
    for (int j = 0; j < NUM_CHANNELS && j < MATRIX_SIZE; j++) {
      if (i == j) {
        currentMatrix[i][j] = smoothedData[i][historyIndex] - baselineVoltages[i];
      } else {
        float correlation = calculateCorrelation(i, j);
        currentMatrix[i][j] = correlation;
      }
    }
  }
}

float calculateCorrelation(int ch1, int ch2) {
  if (analysisCount < 3) return 0.0;
  float sum1 = 0, sum2 = 0, sum12 = 0;
  int samples = (int)min((unsigned long)HISTORY_SIZE, analysisCount);
  for (int i = 0; i < samples; i++) {
    int idx = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
    float val1 = smoothedData[ch1][idx] - baselineVoltages[ch1];
    float val2 = smoothedData[ch2][idx] - baselineVoltages[ch2];
    sum1 += val1;
    sum2 += val2;
    sum12 += val1 * val2;
  }
  float mean1 = sum1 / samples;
  float mean2 = sum2 / samples;
  return (sum12 / samples) - (mean1 * mean2);
}

void performCurvatureMatrixAnalysis() {
  if (analysisCount < 3) return;
  int curr = historyIndex;
  int prev = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
  int prev2 = (historyIndex - 2 + HISTORY_SIZE) % HISTORY_SIZE;
  float currentVal = smoothedData[selectedChannel][curr];
  float prevVal = smoothedData[selectedChannel][prev];
  float prev2Val = smoothedData[selectedChannel][prev2];
  float secondDeriv = currentVal - 2 * prevVal + prev2Val;
  curvatureData[historyIndex] = abs(secondDeriv) * 100;
  for (int i = 0; i < NUM_CHANNELS && i < MATRIX_SIZE; i++) {
    for (int j = 0; j < NUM_CHANNELS && j < MATRIX_SIZE; j++) {
      if (i == selectedChannel && j == selectedChannel) {
        curvatureMatrix[i][j] = curvatureData[historyIndex];
      } else if (i == selectedChannel || j == selectedChannel) {
        curvatureMatrix[i][j] = curvatureData[historyIndex] * currentMatrix[i][j];
      } else {
        curvatureMatrix[i][j] *= 0.95; // Decay factor
      }
    }
  }
}

void calculateFrechetMetrics() {
  for (int i = 0; i < MATRIX_SIZE; i++) {
    float norm = 0;
    for (int j = 0; j < MATRIX_SIZE; j++) {
      norm += currentMatrix[i][j] * currentMatrix[i][j];
    }
    seminormVector[i] = sqrt(norm);
  }
  for (int i = 0; i < MATRIX_SIZE; i++) {
    for (int j = 0; j < MATRIX_SIZE; j++) {
      float dist = abs(seminormVector[i] - seminormVector[j]);
      metricMatrix[i][j] = dist;
    }
  }
  frechetMetric = 0;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    frechetMetric += seminormVector[i] / (1 + seminormVector[i]);
  }
  updateCauchySequence();
}

void updateCauchySequence() {
  for (int i = MATRIX_SIZE - 1; i > 0; i--) {
    cauchySequence[i] = cauchySequence[i - 1];
  }
  cauchySequence[0] = frechetMetric;
  bool converging = true;
  for (int i = 1; i < MATRIX_SIZE; i++) {
    if (abs(cauchySequence[0] - cauchySequence[i]) > FRECHET_THRESHOLD) {
      converging = false;
      break;
    }
  }
  isLocallyConvex = (frechetMetric < 1.0) && converging;
}

void detectSignFlips() {
  flipCount = 0;
  const int window = 15;
  if (analysisCount < window) return;
  float diffs[window] = {0};
  float signs[window - 1] = {0};
  for (int i = 0; i < window; i++) {
    int idx1 = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
    int idx2 = (historyIndex - i - 1 + HISTORY_SIZE) % HISTORY_SIZE;
    diffs[i] = smoothedData[selectedChannel][idx1] - smoothedData[selectedChannel][idx2];
  }
  for (int i = 0; i < window - 1; i++) {
    if (diffs[i] > 0.01) signs[i] = 1;
    else if (diffs[i] < -0.01) signs[i] = -1;
    else signs[i] = 0;
  }
  for (int i = 0; i < window - 2; i++) {
    if (signs[i] != signs[i + 1] && signs[i] != 0 && signs[i + 1] != 0) {
      flipCount++;
    }
  }

}

void performAdvancedMatrixAnalysis() {
  float matrixTrace = 0;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    matrixTrace += currentMatrix[i][i];
  }
  // Determinant calculation for 4x4 is complex, showing for 2x2 part as an example
  float matrixDeterminant = currentMatrix[0][0] * currentMatrix[1][1] - currentMatrix[0][1] * currentMatrix[1][0];
  Serial.print("■ MATRIX PATTERN: Flips=");
  Serial.print(flipCount);
  Serial.print(" Trace=");
  Serial.print(matrixTrace, 4);
  Serial.print(" Det(2x2)=");
  Serial.print(matrixDeterminant, 4);
  Serial.print(" Fréchet=");
  Serial.println(frechetMetric, 4);
}

void displayMatrixOscilloscope() {
  displayEnhancedMatrixStats();
}

void displayEnhancedMatrixStats() {
  float currentVal = getCurrentValue(selectedChannel);
  float peakVal = getPeakValue(selectedChannel);
  float rmsVal = getRMSValue(selectedChannel);
  float curvatureAvg = getAverageCurvature();

  Serial.print("CH");
  Serial.print(selectedChannel);
  Serial.print(": ");
  Serial.print(currentVal, 3);
  Serial.print("V │ PEAK: ");
  Serial.print(peakVal, 3);
  Serial.print("V │ RMS: ");
  Serial.print(rmsVal, 3);
  Serial.print("V │ CURVE: ");
  Serial.print(curvatureAvg, 4);
  Serial.print(" │ FLIPS: ");
  Serial.print(flipCount);
  Serial.print(" │ F-METRIC: ");
  Serial.print(frechetMetric, 4);
  Serial.print(" │ CHARGE: ");
  Serial.print(chargeAccumulation[selectedChannel], 3);
  Serial.print(" │ CONVEX: ");
  Serial.println(isLocallyConvex ? "YES" : "NO");

}

void displayCurrentMatrix() {
  Serial.println("████ CURRENT MATRIX STATE ████");
  for (int i = 0; i < MATRIX_SIZE; i++) {
    Serial.print("│");
    for (int j = 0; j < MATRIX_SIZE; j++) {
      Serial.print(currentMatrix[i][j], 3);
      Serial.print(" ");
    }
    Serial.println("│");
  }
  Serial.println("██████████████████████████████");
}

void displayCurvatureMatrix() {
  Serial.println("████ CURVATURE MATRIX ████");
  for (int i = 0; i < MATRIX_SIZE; i++) {
    Serial.print("│");
    for (int j = 0; j < MATRIX_SIZE; j++) {
      Serial.print(curvatureMatrix[i][j], 3);
      Serial.print(" ");
    }
    Serial.println("│");
  }
  Serial.println("███████████████████████████");
}

void displayFrechetAnalysis() {
  Serial.println("█████ FRÉCHET SPACE ANALYSIS █████");
  Serial.print("Metric: ");
  Serial.println(frechetMetric, 6);
  Serial.print("Locally Convex: ");
  Serial.println(isLocallyConvex ? "TRUE" : "FALSE");
  Serial.print("Seminorms: [");
  for (int i = 0; i < MATRIX_SIZE; i++) {
    Serial.print(seminormVector[i], 3);
    if (i < MATRIX_SIZE - 1) Serial.print(", ");
  }
  Serial.println("]");
  Serial.print("Cauchy Convergence: [");
  int cauchyCount = min(4, MATRIX_SIZE);
  for (int i = 0; i < cauchyCount; i++) {
    Serial.print(cauchySequence[i], 3);
    if (i < cauchyCount - 1) Serial.print(", ");
  }
  Serial.println("]");
  Serial.println("██████████████████████████████████");
}

float getCurrentValue(int channel) {
  int currentIndex = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
  return channelHistory[channel][currentIndex];
}

float getPeakValue(int channel) {
  float peak = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    if (abs(channelHistory[channel][i]) > peak)
      peak = abs(channelHistory[channel][i]);
  }
  return peak;
}

float getRMSValue(int channel) {
  float sum = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    sum += channelHistory[channel][i] * channelHistory[channel][i];
  }
  return sqrt(sum / HISTORY_SIZE);
}

float getAverageCurvature() {
  float sum = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    sum += curvatureData[i];
  }
  return sum / HISTORY_SIZE;
}

void resetAnalysis() {
  initializeMatrices();
  establishBaseline();
  analysisCount = 0;
  flipCount = 0;
  frechetMetric = 0.0;
  isLocallyConvex = false;
  chargeDetected = false;
}
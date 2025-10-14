/*
 * Arduino High-Speed Electrode Reader for Real-Time Quantum Modulation
 * Optimized for real-time streaming to Python
 * 
 * Connections:
 * - Electrode signal -> A0
 * - Electrode ground -> GND
 * - LED indicator -> Pin 13
 */

const int ELECTRODE_PIN = A0;
const int LED_PIN = 13;

// High-speed sampling
float smoothedValue = 0.0;
float alpha = 0.15;  // Smoothing factor (lower = more smoothing)

// Send data every 20ms (50Hz update rate)
unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 20;

// Statistics
float runningMin = 1023.0;
float runningMax = 0.0;
int sampleCount = 0;

void setup() {
  // Higher baud rate for faster transmission
  Serial.begin(9600);
  
  pinMode(ELECTRODE_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Ready blink
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("ARDUINO:READY");
  
  // Calibration
  Serial.println("STATUS:CALIBRATING");
  for (int i = 0; i < 100; i++) {
    int val = analogRead(ELECTRODE_PIN);
    smoothedValue += val;
    delay(5);
  }
  smoothedValue /= 100.0;
  Serial.println("STATUS:READY");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Read and smooth every loop
  int rawValue = analogRead(ELECTRODE_PIN);
  smoothedValue = alpha * rawValue + (1.0 - alpha) * smoothedValue;
  
  // Update stats
  if (smoothedValue < runningMin) runningMin = smoothedValue;
  if (smoothedValue > runningMax) runningMax = smoothedValue;
  sampleCount++;
  
  // Send at fixed interval
  if (currentTime - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = currentTime;
    
    // Send electrode value
    Serial.print("ELECTRODE:");
    Serial.println((int)smoothedValue);
    
    // LED feedback
    int ledBrightness = map((int)smoothedValue, 0, 1023, 0, 255);
    analogWrite(LED_PIN, ledBrightness);
    
    // Send stats every 5 seconds
    if (sampleCount >= 250) {
      Serial.print("STATS:");
      Serial.print((int)runningMin);
      Serial.print(",");
      Serial.print((int)runningMax);
      Serial.print(",");
      Serial.println((int)smoothedValue);
      
      // Reset stats
      sampleCount = 0;
      runningMin = 1023.0;
      runningMax = 0.0;
    }
  }
}

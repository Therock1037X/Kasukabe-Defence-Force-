/*
 * KDF Navigation — RoadWatch Nagpur
 * ESP32 Vehicle Telemetry Node Firmware
 *
 * Sensors:
 * - NEO-6M GPS (Serial2: RX=16, TX=17) -> Latitude, Longitude, Ground Speed (km/h)
 * - MPU6050 6-Axis IMU (I2C: SDA=21, SCL=22) -> Acceleration & Harsh Braking (>0.4g)
 * - Rain Drop Sensor (Analog Pin 34) -> Moisture / Precipitation level (0-4095)
 * - LDR Photoresistor (Analog Pin 35) -> Ambient Light level (0-4095)
 *
 * Dispatches JSON telemetry via HTTP POST to /ingest every 2 seconds.
 * Includes synthetic fallback coordinates for indoor testing without GPS lock.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <TinyGPSPlus.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>

// Include local configuration (copy config.example.h to config.h)
#if __has_include("config.h")
  #include "config.h"
#else
  #include "config.example.h"
#endif

// Hardware objects
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);
Adafruit_MPU6050 mpu;

// Telemetry state variables
unsigned long lastTelemetryTime = 0;
float prevAx = 0.0;
float prevAy = 0.0;
float prevAz = 9.81;
int harshBrakeDetected = 0;
int impactDetected = 0;

// Nagpur Dharampeth default coordinates for indoor prototyping fallback
const double DEFAULT_LAT = 21.1442;
const double DEFAULT_LON = 79.0664;
const float DEFAULT_SPEED = 32.5;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n==========================================");
  Serial.println("  KDF Navigation — ESP32 Sensor Telemetry ");
  Serial.println("  Nagpur RoadWatch Safety Dashboard       ");
  Serial.println("==========================================");

  pinMode(PIN_STATUS_LED, OUTPUT);
  pinMode(PIN_RAIN_ANALOG, INPUT);
  pinMode(PIN_LDR_ANALOG, INPUT);
  digitalWrite(PIN_STATUS_LED, LOW);

  // Initialize GPS UART
  gpsSerial.begin(9600, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX);
  Serial.println("[GPS] Listening on Serial2 (RX=16, TX=17)...");

  // Initialize I2C & MPU6050
  Wire.begin(21, 22);
  if (!mpu.begin()) {
    Serial.println("[MPU6050] Failed to find MPU6050 chip! Check I2C wiring.");
  } else {
    Serial.println("[MPU6050] MPU6050 Initialized.");
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // Connect to Wi-Fi
  connectWiFi();
}

void connectWiFi() {
  Serial.printf("[WIFI] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(PIN_STATUS_LED, HIGH);
    Serial.println("\n[WIFI] Connected successfully!");
    Serial.printf("[WIFI] IP Address: %s\n", WiFi.localIP().toString().c_str());
  } else {
    digitalWrite(PIN_STATUS_LED, LOW);
    Serial.println("\n[WIFI] Connection timeout. Retrying in background...");
  }
}

void checkIMU() {
  sensors_event_t a, g, temp;
  if (!mpu.getEvent(&a, &g, &temp)) return;

  // Convert m/s^2 to g (1g = 9.80665 m/s^2)
  float ax_g = a.acceleration.x / 9.80665f;
  float ay_g = a.acceleration.y / 9.80665f;
  float az_g = a.acceleration.z / 9.80665f;

  // Calculate deceleration magnitude relative to Earth's gravity
  float total_acc_g = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
  float delta_horizontal = sqrt(pow(ax_g - prevAx, 2) + pow(ay_g - prevAy, 2));

  // Harsh deceleration detection (> 0.4g horizontal impulse)
  if (delta_horizontal >= HARSH_BRAKE_G_THRESHOLD) {
    harshBrakeDetected = 1;
    Serial.printf("⚠️ [ALERT] Harsh Braking Detected: %.2fg\n", delta_horizontal);
  }

  // High-g impact collision detection (> 3.0g)
  if (total_acc_g >= CRASH_IMPACT_G_THRESHOLD) {
    impactDetected = 1;
    Serial.printf("🚨 [CRITICAL] Impact Collision Detected: %.2fg!\n", total_acc_g);
  }

  prevAx = ax_g;
  prevAy = ay_g;
  prevAz = az_g;
}

void loop() {
  // Feed GPS parser
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // Continuously monitor accelerometer events
  checkIMU();

  // Send telemetry at configured interval
  unsigned long now = millis();
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;

    // Check Wi-Fi link
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WIFI] Disconnected, reconnecting...");
      WiFi.reconnect();
      return;
    }

    // Read analog sensors (12-bit ADC: 0 - 4095)
    // Rain sensor: Low voltage / low value = Wet; High value = Dry
    // Inverted to 0-1023 scale for backend compatibility
    int rawRain = analogRead(PIN_RAIN_ANALOG);
    float rainScore = map(4095 - rawRain, 0, 4095, 0, 1023);

    // LDR: High value = bright daylight; Low value = dark night
    int rawLdr = analogRead(PIN_LDR_ANALOG);
    float ldrScore = map(rawLdr, 0, 4095, 0, 1023);

    // Extract GPS or use fallback for indoor demo
    double currentLat = DEFAULT_LAT;
    double currentLon = DEFAULT_LON;
    float currentSpeed = DEFAULT_SPEED;

    if (gps.location.isValid()) {
      currentLat = gps.location.lat();
      currentLon = gps.location.lng();
      if (gps.speed.isValid()) {
        currentSpeed = gps.speed.kmph();
      }
    } else {
      // Simulate slight drift for indoor demo if GPS has no fix
      static float drift = 0.0;
      drift += 0.0001;
      currentLat += sin(drift) * 0.0005;
      currentLon += cos(drift) * 0.0005;
    }

    // Assemble JSON document
    StaticJsonDocument<256> doc;
    doc["lat"] = currentLat;
    doc["lon"] = currentLon;
    doc["speed"] = currentSpeed;
    doc["rain_sensor"] = rainScore;
    doc["ldr"] = ldrScore;
    doc["harsh_brake"] = harshBrakeDetected;

    String jsonPayload;
    serializeJson(doc, jsonPayload);

    // Reset one-shot brake flag
    harshBrakeDetected = 0;

    // Dispatch HTTP POST to FastAPI /ingest
    HTTPClient http;
    String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + SERVER_ENDPOINT;
    
    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonPayload);

    if (httpResponseCode > 0) {
      Serial.printf("[HTTP %d] Ingested: Lat=%.5f Lon=%.5f Spd=%.1f Rain=%.0f LDR=%.0f\n",
                    httpResponseCode, currentLat, currentLon, currentSpeed, rainScore, ldrScore);
    } else {
      Serial.printf("[HTTP ERROR] Ingest failed: %s\n", http.errorToString(httpResponseCode).c_str());
    }

    http.end();
  }
}

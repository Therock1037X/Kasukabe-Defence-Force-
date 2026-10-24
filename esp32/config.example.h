#ifndef CONFIG_H
#define CONFIG_H

// Wi-Fi Credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// KDF Navigation Backend Server Configuration
// Replace with the IP address of the machine running FastAPI (e.g., 192.168.1.100)
const char* SERVER_HOST = "192.168.1.100";
const int SERVER_PORT = 8000;
const char* SERVER_ENDPOINT = "/ingest";

// Hardware Pin Definitions
#define PIN_RAIN_ANALOG 34   // ADC1 Channel 6
#define PIN_LDR_ANALOG  35   // ADC1 Channel 7
#define PIN_GPS_RX      16   // ESP32 RX2 connects to GPS TX
#define PIN_GPS_TX      17   // ESP32 TX2 connects to GPS RX
#define PIN_STATUS_LED   2   // Onboard Blue LED

// Sensor Calibration & Thresholds
#define HARSH_BRAKE_G_THRESHOLD 0.40f  // 0.4g deceleration trigger
#define CRASH_IMPACT_G_THRESHOLD 3.00f // 3.0g collision impact trigger
#define TELEMETRY_INTERVAL_MS    2000  // Post every 2 seconds

#endif // CONFIG_H

# ESP32 Vehicle Telemetry Firmware — KDF Navigation

This directory contains the production Arduino firmware for the in-vehicle IoT node that continuously samples physical sensors and dispatches real-time telemetry to the KDF Navigation FastAPI backend (`POST /ingest`).

---

## 1. Hardware Bill of Materials (BOM)

| Component | Interface | ESP32 Pin | Purpose |
| :--- | :--- | :--- | :--- |
| **ESP32 DevKit V1** | Micro-USB / 3.3V | — | Main 240 MHz dual-core MCU with Wi-Fi |
| **NEO-6M GPS Module** | UART (Serial2) | TX $\to$ RX2 (GPIO 16)<br>RX $\to$ TX2 (GPIO 17) | Global positioning, ground speed ($km/h$) |
| **MPU6050 6-Axis IMU** | $I^2C$ (Wire) | SDA $\to$ GPIO 21<br>SCL $\to$ GPIO 22 | Acceleration, harsh braking ($>0.4g$), collision ($>3g$) |
| **Rain Drop Sensor Module** | Analog (ADC1) | AO $\to$ GPIO 34 | Precipitation detection & surface wetness proxy |
| **LDR Photoresistor Module** | Analog (ADC1) | AO $\to$ GPIO 35 | Ambient light & unlit corridor proxy |
| **Onboard Blue LED** | GPIO Output | GPIO 2 | Wi-Fi connection indicator |

> [!NOTE]
> GPIO 34 and GPIO 35 belong to ADC1 on the ESP32. They are input-only pins without internal pull-ups, which makes them ideal for analog sensor reading while Wi-Fi is actively transmitting.

---

## 2. Wiring & Pinout Diagram

```text
       +---------------------------------------------+
       |                  ESP32 DevKit               |
       |                                             |
       |  [GPIO 16] <--- GPS TX (NEO-6M)             |
       |  [GPIO 17] ---> GPS RX (NEO-6M)             |
       |                                             |
       |  [GPIO 21] <--> MPU6050 SDA                 |
       |  [GPIO 22] <--> MPU6050 SCL                 |
       |                                             |
       |  [GPIO 34] <--- Rain Sensor Analog AO       |
       |  [GPIO 35] <--- LDR Module Analog AO        |
       |                                             |
       |  [3.3V]    ---> VCC (All Sensors)           |
       |  [GND]     ---> GND (Common Ground)         |
       +---------------------------------------------+
```

---

## 3. Required Arduino Libraries

Install the following libraries via the Arduino IDE Library Manager (**Sketch** $\to$ **Include Library** $\to$ **Manage Libraries...**):

1. **TinyGPSPlus** by Mikal Hart
2. **Adafruit MPU6050** by Adafruit
3. **Adafruit Sensor** by Adafruit
4. **ArduinoJson** (v6.x) by Benoit Blanchon

---

## 4. Setup & Flashing Instructions

1. Copy `config.example.h` to `config.h`:
   ```bash
   cp esp32/config.example.h esp32/config.h
   ```
2. Open `esp32/config.h` in your editor and configure your local Wi-Fi SSID, password, and the IP address of your host machine running the FastAPI backend:
   ```c
   const char* WIFI_SSID = "MyHomeWiFi";
   const char* WIFI_PASS = "SecretPassword";
   const char* SERVER_HOST = "192.168.1.100"; // Host running uvicorn
   const int SERVER_PORT = 8000;
   ```
3. In Arduino IDE:
   - Select Board: **ESP32 Dev Module**
   - Upload Speed: **921600**
   - Flash Frequency: **80MHz**
   - Select Port: `/dev/ttyUSB0` or `COMx`
4. Click **Upload**.
5. Open Serial Monitor at **115200 baud** to monitor initialization, Wi-Fi handshake, and HTTP ingestion logs.

---

## 5. JSON Telemetry Specification

The firmware dispatches an HTTP POST payload every 2 seconds to `http://<SERVER_HOST>:8000/ingest`:

```json
{
  "lat": 21.1442,
  "lon": 79.0664,
  "speed": 34.2,
  "rain_sensor": 620.0,
  "ldr": 180.0,
  "harsh_brake": 1
}
```

- `rain_sensor > 500`: The backend infers `"rain"` and adjusts road surface friction and visibility.
- `ldr < 350`: The backend infers `"dark"` and checks road lighting (`lit_tag` / `ntl_percentile`).
- `harsh_brake == 1`: Logged in SQLite `sensor_logs.db` for incident pattern detection.

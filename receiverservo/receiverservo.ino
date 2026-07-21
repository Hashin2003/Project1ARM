#include <ESP32Servo.h>
#include <esp_now.h>
#include <WiFi.h>
#include "esp_wifi.h"

// Servo objects
Servo myServo1, myServo2, myServo3, myServo4, myServo5;

// Servo pins
#define SERVO1_PIN 4
#define SERVO2_PIN 5
#define SERVO3_PIN 6
#define SERVO4_PIN 7
#define SERVO5_PIN 15

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Servo + ESP-NOW Receiver");

  // Attach servos
  myServo1.attach(SERVO1_PIN);
  myServo2.attach(SERVO2_PIN);
  myServo3.attach(SERVO3_PIN);
  myServo4.attach(SERVO4_PIN);
  myServo5.attach(SERVO5_PIN);

  // Wi-Fi + ESP-NOW setup
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE); // Match sender channel

  // Serial.print("My MAC: ");
  // Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    return;
  }

  // Register receive callback
  esp_now_register_recv_cb(onReceive);
}

void loop() {
  // Nothing here; messages handled in callback
}

// ESP-NOW receive callback
void onReceive(const esp_now_recv_info *info, const uint8_t *data, int len) {
  String incomingLine = String((char *)data).substring(0, len);
  incomingLine.trim();

  Serial.print("Received from: ");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", info->src_addr[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.print(" | Message: ");
  Serial.println(incomingLine);

  // Parse servo values: val1@val2@val3@val4@val5
  int split1 = incomingLine.indexOf('@');
  int split2 = incomingLine.indexOf('@', split1 + 1);
  int split3 = incomingLine.indexOf('@', split2 + 1);
  int split4 = incomingLine.indexOf('@', split3 + 1);

  if (split1 > 0 && split2 > split1 && split3 > split2 && split4 > split3) {
    int val1 = incomingLine.substring(0, split1).toInt();
    int val2 = incomingLine.substring(split1 + 1, split2).toInt();
    int val3 = incomingLine.substring(split2 + 1, split3).toInt();
    int val4 = incomingLine.substring(split3 + 1, split4).toInt();
    int val5 = incomingLine.substring(split4 + 1).toInt();

    int angle1 = getAngleServo1(val1);
    int angle2 = getAngleServo2(val2);
    int angle3 = getAngleServo3(val3);
    int angle4 = getAngleServo4(val4);
    int angle5 = getAngleServo5(val5);

    myServo1.write(angle1);
    myServo2.write(angle2);
    myServo3.write(angle3);
    myServo4.write(angle4);
    myServo5.write(angle5);

    Serial.printf(
      "val1=%d angle1=%d | val2=%d angle2=%d | val3=%d angle3=%d | val4=%d angle4=%d | val5=%d angle5=%d\n",
      val1, angle1, val2, angle2, val3, angle3, val4, angle4, val5, angle5
    );
  } else {
    Serial.println("Invalid input format.");
  }
}

// Servo threshold functions
int getAngleServo1(int val) {
  if (val > 140) return 180;
  else if (val > 130) return 153;
  else if (val > 120) return 115;
  else if (val > 110) return 98;
  else return 70;
}

int getAngleServo2(int val) {
  if (val > 150) return 180;
  else if (val > 135) return 150;
  else if (val > 105) return 90;
  else if (val > 90) return 30;
  else return 0;
}

int getAngleServo3(int val) {
  if (val > 150) return 180;
  else if (val > 135) return 150;
  else if (val > 105) return 90;
  else if (val > 90) return 30;
  else return 0;
}

int getAngleServo4(int val) {
  if (val > 150) return 0;
  else if (val > 135) return 30;
  else if (val > 105) return 90;
  else if (val > 90) return 150;
  else return 180;
}

int getAngleServo5(int val) {
  if (val > 150) return 0;
  else if (val > 135) return 21;
  else if (val > 105) return 42;
  else if (val > 90) return 64;
  else return 85;
}

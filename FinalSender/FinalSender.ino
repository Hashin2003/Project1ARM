#include <esp_now.h>
#include <WiFi.h>
#include "esp_wifi.h"

// Receiver MAC address (change to your receiver ESP32 MAC)
uint8_t receiverMac[] = {0x10, 0x20, 0xBA, 0x4C, 0x5B, 0x14};

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 ESP-NOW Sender");

  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

  Serial.print("My MAC: ");
  Serial.println(WiFi.macAddress());

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    return;
  }

  // Register send callback
  esp_now_register_send_cb(onSent);

  // Register receiver as peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 1;  
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }
  Serial.println("Peer added successfully");
}

void loop() {
  // Read Serial input
  if (Serial.available()) {
    String incomingLine = Serial.readStringUntil('\n');
    incomingLine.trim();

    // Expecting format: val1@val2@val3@val4@val5
    int split1 = incomingLine.indexOf('@');
    int split2 = incomingLine.indexOf('@', split1 + 1);
    int split3 = incomingLine.indexOf('@', split2 + 1);
    int split4 = incomingLine.indexOf('@', split3 + 1);

    if (split1 > 0 && split2 > split1 && split3 > split2 && split4 > split3) {
      // Send the raw string via ESP-NOW
      esp_err_t result = esp_now_send(receiverMac, (uint8_t *)incomingLine.c_str(), incomingLine.length() + 1);

      if (result == ESP_OK) {
        Serial.println("Message sent successfully");
      } else {
        Serial.print("Send error: ");
        Serial.println(result);
      }
    } else {
      Serial.println("Invalid input format. Use val1@val2@val3@val4@val5");
    }
  }
  delay(50); // Small delay to avoid flooding
}

// ESP-NOW send callback
void onSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  Serial.print("Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

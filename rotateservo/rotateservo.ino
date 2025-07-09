#include <ESP32Servo.h>

Servo myServo;
String incomingLine = "";

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 is ready...");

  // Attach the servo to GPIO19 with pulse width range
  // myServo.setPeriodHertz(50);             // 50 Hz servo control signal
  myServo.attach(19);          // Pin 19, pulse width from 500µs to 2500µs
}

void loop() {
  // Check if there's incoming serial data
  if (Serial.available()) {
    incomingLine = Serial.readStringUntil('\n');
    incomingLine.trim();  // Remove whitespace or newline

    // Parse the line formatted like: "val1@val2@val3@val4@val5"
    int idx1 = incomingLine.indexOf('@');
    if (idx1 > 0) {
      int val1 = incomingLine.substring(0, idx1).toInt();
      int angle = 90;  // Default angle

      // Apply thresholds
      if (val1 > 130) {
        angle = 150;
      } else if (val1 >= 90 && val1 <= 130) {
        angle = 90;
      } else if (val1 < 90) {
        angle = 30;
      }

      // Write angle to servo
      myServo.write(angle);

      // Debug print
      Serial.print("Received val1: ");
      Serial.print(val1);
      Serial.print(" -> Servo angle: ");
      Serial.println(angle);
    } else {
      Serial.println("Invalid input format.");
    }
  }
}

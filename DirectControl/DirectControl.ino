#include <ESP32Servo.h>

Servo myServo1, myServo2, myServo3, myServo4, myServo5;

#define SERVO1_PIN 4
#define SERVO2_PIN 5
#define SERVO3_PIN 6
#define SERVO4_PIN 7
#define SERVO5_PIN 15

void setup() {
  Serial.begin(115200);
  myServo1.attach(SERVO1_PIN);
  myServo2.attach(SERVO2_PIN);
  myServo3.attach(SERVO3_PIN);
  myServo4.attach(SERVO4_PIN);
  myServo5.attach(SERVO5_PIN);
  Serial.println("ESP32 Servo Controller Ready");
}

void loop() {
  if (Serial.available()) {
    String incomingLine = Serial.readStringUntil('\n');
    incomingLine.trim();

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

      myServo1.write(getAngleServo1(val1));
      myServo2.write(getAngleServo2(val2));
      myServo3.write(getAngleServo3(val3));
      myServo4.write(getAngleServo4(val4));
      myServo5.write(getAngleServo5(val5));

      Serial.printf("val1=%d val2=%d val3=%d val4=%d val5=%d\n",
                    val1, val2, val3, val4, val5);
    }
  }
}

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
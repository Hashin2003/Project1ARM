import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# --- Serial Setup ---
ser = serial.Serial('COM7', 115200, timeout=1)  # Update to your correct COM port
time.sleep(2)  # Allow time for ESP32 to reset

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            # Get wrist coords
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = np.array([wrist.x * w, wrist.y * h])

            # Fingertips to measure
            fingertips = {
                "Thumb": mp_hands.HandLandmark.THUMB_TIP,
                "Index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                "Middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                "Ring": mp_hands.HandLandmark.RING_FINGER_TIP,
                "Pinky": mp_hands.HandLandmark.PINKY_TIP
            }

            distances = {}
            for finger_name, landmark_id in fingertips.items():
                tip = hand_landmarks.landmark[landmark_id]
                tip_pos = np.array([tip.x * w, tip.y * h])
                dist = np.linalg.norm(tip_pos - wrist_pos)
                distances[finger_name] = int(dist)

                # Display distances on the frame
                cv2.putText(frame, f"{finger_name}: {int(dist)} px", 
                            (10, 30 + 30 * list(fingertips.keys()).index(finger_name)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Prepare and send raw values only, separated by '@'
            send_str = "@".join([str(int(v)) for v in distances.values()]) + "\n"
            ser.write(send_str.encode())
            print(f"Sent to ESP32: {send_str.strip()}")

            # Optional: Read back from ESP32
            if ser.in_waiting:
                reply = ser.readline().decode().strip()
                if reply:
                    print(f"ESP32 replied: {reply}")

            # time.sleep(0.2)

    cv2.imshow('Finger Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()

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

# --- Hand Selection Variables ---
selected_hand = None  # Will store the selected hand's wrist position
selection_mode = "auto"  # "auto", "left", "right", "manual", "largest"
manual_selected_hand = None
frame_count = 0

def get_hand_label(hand_landmarks, results):
    """Determine if hand is left or right"""
    # Get the classification from MediaPipe
    if results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            if hand_handedness.classification[0].label == "Left":
                return "Right"  # MediaPipe returns mirrored labels
            else:
                return "Left"
    return "Unknown"

def calculate_hand_area(hand_landmarks, w, h):
    """Calculate the bounding box area of a hand"""
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    return (max_x - min_x) * (max_y - min_y)

def select_hand_by_method(results, frame, method="auto"):
    """Select hand based on specified method"""
    if not results.multi_hand_landmarks:
        return None, None
    
    h, w, _ = frame.shape
    
    if method == "auto" or method == "largest":
        # Select the largest hand (closest to camera)
        largest_area = 0
        selected_hand = None
        selected_label = None
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            area = calculate_hand_area(hand_landmarks, w, h)
            if area > largest_area:
                largest_area = area
                selected_hand = hand_landmarks
                if results.multi_handedness:
                    label = results.multi_handedness[idx].classification[0].label
                    selected_label = "Right" if label == "Left" else "Left"
        
        return selected_hand, selected_label
    
    elif method == "left" or method == "right":
        # Select based on hand type
        target_label = "Left" if method == "right" else "Right"  # Mirrored
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness:
                label = results.multi_handedness[idx].classification[0].label
                if label == target_label:
                    actual_label = "Right" if label == "Left" else "Left"
                    return hand_landmarks, actual_label
        
        return None, None
    
    elif method == "manual":
        # Manual selection will be handled separately
        return None, None

def manual_hand_selection(results, frame, mouse_pos):
    """Select hand based on mouse click position"""
    if not results.multi_hand_landmarks or mouse_pos is None:
        return None, None
    
    h, w, _ = frame.shape
    click_x, click_y = mouse_pos
    
    min_distance = float('inf')
    selected_hand = None
    selected_label = None
    
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # Use wrist position for selection
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        
        distance = np.sqrt((click_x - wrist_x)**2 + (click_y - wrist_y)**2)
        
        if distance < min_distance and distance < 100:  # Within 100 pixels
            min_distance = distance
            selected_hand = hand_landmarks
            if results.multi_handedness:
                label = results.multi_handedness[idx].classification[0].label
                selected_label = "Right" if label == "Left" else "Left"
    
    return selected_hand, selected_label

# Mouse callback for manual selection
mouse_pos = None
def mouse_callback(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos = (x, y)

cv2.namedWindow('Finger Tracking')
cv2.setMouseCallback('Finger Tracking', mouse_callback)

print("Hand Selection Controls:")
print("1 - Auto/Largest hand")
print("2 - Left hand only")
print("3 - Right hand only") 
print("4 - Manual selection (click on hand)")
print("Q - Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    frame_count += 1
    
    # Handle keyboard input for selection mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        selection_mode = "auto"
        mouse_pos = None
        print("Mode: Auto (Largest hand)")
    elif key == ord('2'):
        selection_mode = "left"
        mouse_pos = None
        print("Mode: Left hand only")
    elif key == ord('3'):
        selection_mode = "right"
        mouse_pos = None
        print("Mode: Right hand only")
    elif key == ord('4'):
        selection_mode = "manual"
        print("Mode: Manual (Click on hand to select)")
    elif key == ord('q'):
        break

    # Draw all detected hands first (in light gray)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1),
                                    mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1))
            
            # Label each hand
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            
            if results.multi_handedness:
                label = results.multi_handedness[idx].classification[0].label
                actual_label = "Right" if label == "Left" else "Left"
                cv2.putText(frame, f"{actual_label}", (wrist_x - 30, wrist_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # Select the target hand
    if selection_mode == "manual":
        selected_hand_landmarks, selected_label = manual_hand_selection(results, frame, mouse_pos)
        if selected_hand_landmarks:
            manual_selected_hand = selected_hand_landmarks
            mouse_pos = None  # Reset after selection
        elif manual_selected_hand:
            # Try to track the previously selected hand
            selected_hand_landmarks = manual_selected_hand
            selected_label = "Selected"
    else:
        selected_hand_landmarks, selected_label = select_hand_by_method(results, frame, selection_mode)

    # Process the selected hand
    if selected_hand_landmarks:
        # Draw the selected hand in bright colors
        mp_drawing.draw_landmarks(frame, selected_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))

        h, w, _ = frame.shape

        # Get wrist coords
        wrist = selected_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
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
            tip = selected_hand_landmarks.landmark[landmark_id]
            tip_pos = np.array([tip.x * w, tip.y * h])
            dist = np.linalg.norm(tip_pos - wrist_pos)
            distances[finger_name] = int(dist)

        # Display distances on the frame
        for i, (finger_name, dist) in enumerate(distances.items()):
            cv2.putText(frame, f"{finger_name}: {int(dist)} px", 
                        (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display selected hand info
        cv2.putText(frame, f"Selected: {selected_label if selected_label else 'Unknown'}", 
                    (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Prepare and send raw values only, separated by '@'
        send_str = "@".join([str(int(v)) for v in distances.values()]) + "\n"
        ser.write(send_str.encode())
        print(f"Sent to ESP32: {send_str.strip()}")

        # Optional: Read back from ESP32
        if ser.in_waiting:
            reply = ser.readline().decode().strip()
            if reply:
                print(f"ESP32 replied: {reply}")

    # Display current mode
    cv2.putText(frame, f"Mode: {selection_mode}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display hand count
    hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    cv2.putText(frame, f"Hands detected: {hand_count}", (10, frame.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Finger Tracking', frame)

cap.release()
ser.close()
cv2.destroyAllWindows()
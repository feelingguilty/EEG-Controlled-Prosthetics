import serial
import cv2
import mediapipe as mp
import math

debug = False
# ser = serial.Serial('COM6', 115200)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]
    dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def map_angle_to_servo(angle):
    # Map the calculated finger angle to a range between 0 and 90 degrees
    return max(0, min(90, angle))

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger Landmarks: Using PIP, MCP for each finger
            INDEX_TIP = hand_landmarks.landmark[8]
            INDEX_PIP = hand_landmarks.landmark[6]
            INDEX_MCP = hand_landmarks.landmark[5]

            MIDDLE_TIP = hand_landmarks.landmark[12]
            MIDDLE_PIP = hand_landmarks.landmark[10]
            MIDDLE_MCP = hand_landmarks.landmark[9]

            RING_TIP = hand_landmarks.landmark[16]
            RING_PIP = hand_landmarks.landmark[14]
            RING_MCP = hand_landmarks.landmark[13]

            PINKY_TIP = hand_landmarks.landmark[20]
            PINKY_PIP = hand_landmarks.landmark[18]
            PINKY_MCP = hand_landmarks.landmark[17]

            # Thumb: Use CMC, MCP, IP for better angle detection
            THUMB_TIP = hand_landmarks.landmark[4]
            THUMB_IP = hand_landmarks.landmark[3]
            THUMB_MCP = hand_landmarks.landmark[2]
            THUMB_CMC = hand_landmarks.landmark[1]

            # Calculate the angles for each finger
            index_angle = calculate_angle(INDEX_TIP, INDEX_PIP, INDEX_MCP)
            middle_angle = calculate_angle(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
            ring_angle = calculate_angle(RING_TIP, RING_PIP, RING_MCP)
            pinky_angle = calculate_angle(PINKY_TIP, PINKY_PIP, PINKY_MCP)

            # For the thumb, we'll calculate angle between CMC, MCP, and IP
            thumb_angle = calculate_angle(THUMB_TIP, THUMB_IP, THUMB_MCP)

            # Map the angles to the 0-90 range for servo control
            index_servo = map_angle_to_servo(index_angle)
            middle_servo = map_angle_to_servo(middle_angle)
            ring_servo = map_angle_to_servo(ring_angle)
            pinky_servo = map_angle_to_servo(pinky_angle)
            thumb_servo = map_angle_to_servo(thumb_angle)

            # Create a list of servo angles
            servo_angles = [thumb_servo, index_servo, middle_servo, ring_servo, pinky_servo]

            # Print angles (for testing)
            print(f"Thumb: {thumb_servo:.2f}, Index: {index_servo:.2f}, Middle: {middle_servo:.2f}, Ring: {ring_servo:.2f}, Pinky: {pinky_servo:.2f}")

            # Send servo angles to ESP32 if needed (uncomment if using with hardware)
            # if not debug:
            #     ser.write(bytearray(int(angle) for angle in servo_angles))

            # Display angles on the image before flipping
            text = f"Thumb: {thumb_servo:.2f}, Index: {index_servo:.2f}, Middle: {middle_servo:.2f}, Ring: {ring_servo:.2f}, Pinky: {pinky_servo:.2f}"
            # Display the array of angles in black
            angles_text = f"Angles: {servo_angles}"
            cv2.putText(image, angles_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Flip the image horizontally for a selfie-view display after the text is added
        image = cv2.flip(image, 1)
        cv2.imshow('MediaPipe Hands (Servo Angles)', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()

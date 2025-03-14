from operator import index
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from PIL.ImageChops import screen

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

# Configure MediaPipe hands
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# Screen settings
screen_width, screen_height = pyautogui.size()
index_y = 0
smoothing = 5
previous_x, previous_y = 0, 0

# Mode tracking
MODE = "Mouse"  # Default mode
last_gesture_time = time.time()
scroll_speed = 0
is_dragging = False

# Status display settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
status_color = (255, 255, 255)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    # Draw mode indicator
    cv2.putText(frame, f"Mode: {MODE}", (10, 30), font, font_scale, status_color, font_thickness)
    
    # Create a status bar at the bottom
    status_bar = np.zeros((50, frame_width, 3), dtype=np.uint8)
    frame = np.vstack([frame, status_bar])
    
    if hands:
        for hand in hands:
            # Draw hand landmarks with better styling
            drawing_utils.draw_landmarks(
                frame, 
                hand, 
                mp.solutions.hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
            
            landmarks = hand.landmark
            
            # Get finger positions
            index_finger_tip = landmarks[8]
            index_finger_mcp = landmarks[5]
            middle_finger_tip = landmarks[12] 
            thumb_tip = landmarks[4]
            ring_finger_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Calculate positions
            index_x = int(index_finger_tip.x * frame_width)
            index_y = int(index_finger_tip.y * frame_height)
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)
            middle_y = int(middle_finger_tip.y * frame_height)
            
            # Smooth mouse movement
            mouse_x = screen_width / frame_width * index_x
            mouse_y = screen_height / frame_height * index_y
            
            if previous_x == 0:
                previous_x, previous_y = mouse_x, mouse_y
            
            smoothed_x = previous_x + (mouse_x - previous_x) / smoothing
            smoothed_y = previous_y + (mouse_y - previous_y) / smoothing
            
            previous_x, previous_y = smoothed_x, smoothed_y
            
            # Movement visualization
            cv2.circle(img=frame, center=(index_x, index_y), radius=10, color=(0, 255, 255), thickness=-1)
            cv2.circle(img=frame, center=(thumb_x, thumb_y), radius=10, color=(255, 0, 255), thickness=-1)
            
            # Calculate distances
            thumb_index_distance = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5
            index_middle_distance = abs(index_y - middle_y)
            
            # Move cursor
            pyautogui.moveTo(smoothed_x, smoothed_y)
            
            # Left click - thumb and index finger pinch
            if thumb_index_distance < 40:
                cv2.circle(img=frame, center=(index_x, index_y), radius=10, color=(0, 255, 0), thickness=-1)
                if not is_dragging and time.time() - last_gesture_time > 0.3:
                    pyautogui.click()
                    last_gesture_time = time.time()
                    cv2.putText(frame, "Left Click", (frame_width - 150, frame_height + 30), font, font_scale, (0, 255, 0), font_thickness)
            
            # Right click - middle finger down, index finger up
            if middle_finger_tip.y > index_finger_mcp.y and index_finger_tip.y < index_finger_mcp.y:
                if time.time() - last_gesture_time > 0.5:
                    cv2.putText(frame, "Right Click", (frame_width - 150, frame_height + 30), font, font_scale, (0, 0, 255), font_thickness)
                    pyautogui.rightClick()
                    last_gesture_time = time.time()
            
            # Drag mode - pinky finger up
            if pinky_tip.y < landmarks[17].y and ring_finger_tip.y > landmarks[13].y:
                if not is_dragging and time.time() - last_gesture_time > 0.5:
                    pyautogui.mouseDown()
                    is_dragging = True
                    MODE = "Dragging"
                    last_gesture_time = time.time()
                cv2.putText(frame, "Dragging", (frame_width - 150, frame_height + 30), font, font_scale, (255, 165, 0), font_thickness)
            # Release drag
            elif is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                MODE = "Mouse"
                last_gesture_time = time.time()
                
            # Scroll mode - index and middle finger extended
            if index_finger_tip.y < index_finger_mcp.y and middle_finger_tip.y < landmarks[9].y:
                MODE = "Scroll"
                # Calculate scroll direction and speed
                if abs(index_finger_tip.y - previous_y/screen_height*frame_height) > 5:
                    scroll_speed = (index_finger_tip.y - previous_y/screen_height*frame_height) / 10
                    pyautogui.scroll(-int(scroll_speed))
                    cv2.putText(frame, f"Scrolling: {int(-scroll_speed)}", (frame_width - 200, frame_height + 30), font, font_scale, (255, 255, 0), font_thickness)
            else:
                if MODE == "Scroll":
                    MODE = "Mouse"
                    
    # Add instructions at the bottom
    cv2.putText(frame, "Pinch: Left Click | Middle Down: Right Click | Pinky Up: Drag | Two Fingers Up: Scroll", 
                (10, frame_height + 30), font, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Virtual Mouse', frame)
    
    # Break loop with Esc key
    if cv2.waitKey(1) == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
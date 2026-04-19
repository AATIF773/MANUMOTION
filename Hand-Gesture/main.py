import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
from datetime import datetime
from pynput.mouse import Button, Controller
import util  # util.py is in the same directory

# HANDLING pyautogui
import sys
from unittest.mock import MagicMock

# Trick the app into thinking pyautogui exists on the cloud server
try:
    import pyautogui
except ImportError:
    sys.modules['pyautogui'] = MagicMock()
# ─────────────────────────────────────────────────────────────────
# GLOBAL STATE FOR GESTURE LOGIC
# ─────────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
mouse          = Controller()
screen_w, screen_h = pyautogui.size()
plocX, plocY    = 0, 0
clocX, clocY    = 0, 0
smoothening     = 5
margin          = 0.15

last_action_time   = 0
COOLDOWN           = 0.6
last_play_state    = False
last_pause_state   = False
current_gesture    = None
gesture_text       = ""

def find_finger_tip(processed):
    mp_h = mp.solutions.hands
    if processed.multi_hand_landmarks:
        return processed.multi_hand_landmarks[0].landmark[mp_h.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(index_finger_tip):
    global plocX, plocY, clocX, clocY
    if index_finger_tip is not None:
        adj_x = (index_finger_tip.x - margin) / (1.0 - 2 * margin)
        adj_y = (index_finger_tip.y - margin) / (1.0 - 2 * margin)
        x = np.clip(int(adj_x * screen_w), 0, screen_w - 1)
        y = np.clip(int(adj_y * screen_h), 0, screen_h - 1)
        clocX = plocX + (x - plocX) / smoothening
        clocY = plocY + (y - plocY) / smoothening
        pyautogui.moveTo(clocX, clocY, _pause=False)
        plocX, plocY = clocX, clocY

def is_left_click(lml, tid):
    return (util.get_angle(lml[5], lml[6], lml[8]) < 50 and
            util.get_angle(lml[9], lml[10], lml[12]) > 90 and tid > 50)

def is_right_click(lml, tid):
    return (util.get_angle(lml[9], lml[10], lml[12]) < 50 and
            util.get_angle(lml[5], lml[6], lml[8]) > 90 and tid > 50)

def is_double_click(lml):
    ia = util.get_angle(lml[5], lml[6], lml[8])
    ma = util.get_angle(lml[9], lml[10], lml[12])
    td = util.get_distance([lml[4], lml[8]])
    return ia < 50 and ma < 50 and td > 100

def is_fist_closed(lml):
    return all(lml[i][1] > lml[i-2][1] for i in [8, 12, 16, 20])

def is_palm_open(lml):
    return all(lml[t][1] < lml[t-2][1] for t in [8, 12, 16, 20])

def is_volume_up(lml):
    return util.get_distance([lml[4], lml[8]]) > 150

def is_volume_down(lml):
    return util.get_distance([lml[4], lml[8]]) < 40

def is_thumb_open(lml):
    return abs(lml[4][0]-lml[2][0]) > 45 and abs(lml[4][0]-lml[5][0]) > 45

def are_two_fingers_up(lml):
    return (lml[8][1] < lml[6][1] and lml[12][1] < lml[10][1] and
            lml[16][1] > lml[14][1] and lml[20][1] > lml[18][1])

def is_screenshot(lml):
    def up(t, p):  return lml[t][1] < lml[p][1] - 10
    def dn(t, p):  return lml[t][1] > lml[p][1] + 10
    return up(8,6) and up(12,10) and up(16,14) and dn(20,18) and lml[4][0] < lml[3][0]

def take_screenshot(frame):
    folder = os.path.expanduser("~/Pictures/Screenshots")
    os.makedirs(folder, exist_ok=True)
    fname = f"Screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    pyautogui.screenshot().save(os.path.join(folder, fname))

def detect_gesture(frame, lml, processed):
    global last_action_time, last_play_state, last_pause_state
    global current_gesture, gesture_text

    if len(lml) < 21:
        current_gesture = None
        gesture_text = ""
        return gesture_text

    tip = find_finger_tip(processed)
    tid = util.get_distance([lml[4], lml[5]])

    # Mouse movement
    if (are_two_fingers_up(lml) and not is_thumb_open(lml)
            and not is_palm_open(lml) and not is_fist_closed(lml)):
        move_mouse(tip)

    # Gesture detection
    new_g = None
    if   is_palm_open(lml):               new_g = "play"
    elif is_fist_closed(lml):             new_g = "pause"
    elif is_left_click(lml, tid):         new_g = "left_click"
    elif is_right_click(lml, tid):        new_g = "right_click"
    elif is_double_click(lml):            new_g = "double_click"
    elif is_screenshot(lml):              new_g = "screenshot"
    elif is_volume_up(lml):               new_g = "volume_up"
    elif is_volume_down(lml):             new_g = "volume_down"
    
    if new_g != current_gesture:
        current_gesture = new_g

    now = time.time()
    if current_gesture and (now - last_action_time > COOLDOWN):
        if current_gesture == "play" and not last_play_state:
            pyautogui.press("playpause"); gesture_text = "▶  Play"
            last_play_state = True; last_pause_state = False; last_action_time = now
        elif current_gesture == "pause" and not last_pause_state:
            pyautogui.press("playpause"); gesture_text = "⏸  Pause"
            last_pause_state = True; last_play_state = False; last_action_time = now
        elif current_gesture == "left_click":
            mouse.click(Button.left); gesture_text = "Left Click"; last_action_time = now
        elif current_gesture == "right_click":
            mouse.click(Button.right); gesture_text = "Right Click"; last_action_time = now
        elif current_gesture == "double_click":
            pyautogui.doubleClick(); gesture_text = "Double Click"; last_action_time = now
        elif current_gesture == "screenshot":
            take_screenshot(frame); gesture_text = "Screenshot"; last_action_time = now
        elif current_gesture == "volume_up" and (now - last_action_time > 0.2):
            pyautogui.press("volumeup", presses=2); gesture_text = "Vol Up"; last_action_time = now
        elif current_gesture == "volume_down" and (now - last_action_time > 0.2):
            pyautogui.press("volumedown", presses=2); gesture_text = "Vol Down"; last_action_time = now

    h, w = frame.shape[:2]
    if gesture_text:
        cv2.putText(frame, gesture_text, (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
    cv2.rectangle(frame,
                  (int(w*margin), int(h*margin)),
                  (int(w*(1-margin)), int(h*(1-margin))),
                  (0, 255, 120), 1)
                  
    return gesture_text

# ─────────────────────────────────────────────────────────────────
# LOCAL RUN LOOP (Ignored by Streamlit)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                           min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    print("Local System Active: Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        h, w = frame.shape[:2]
        lml = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks[0].landmark:
                lml.append([int(lm.x * w), int(lm.y * h)])
                
        detect_gesture(frame, lml, results)
        
        cv2.imshow("Local ManuMotion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

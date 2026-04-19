import cv2
import pandas as pd
import os
import mediapipe as mp

OUTPUT_FILE = "E:/ISL-CSV/dataset/raw/isl_landmarks.csv"

os.makedirs("E:/ISL-CSV/dataset/raw", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 126 features for 2 hands
columns = []

for hand in ["L","R"]:
    for i in range(21):

        columns.append(f"{hand}_x{i}")
        columns.append(f"{hand}_y{i}")
        columns.append(f"{hand}_z{i}")

columns.append("label")

# create csv file if not exists
if not os.path.exists(OUTPUT_FILE):

    df = pd.DataFrame(columns=columns)

    df.to_csv(OUTPUT_FILE, index=False)

print("Saving data to:", OUTPUT_FILE)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # create empty row for 2 hands
    row = [0]*126

    if results.multi_hand_landmarks:

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # limit to max 2 hands
            if hand_index >= 2:
                continue

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for i, lm in enumerate(hand_landmarks.landmark):

                base_index = hand_index*63 + i*3

                if base_index+2 < 126:

                    row[base_index] = lm.x
                    row[base_index+1] = lm.y
                    row[base_index+2] = lm.z

    cv2.putText(
        frame,
        "Press A-Z to save",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("ISL Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key != 255:

        char = chr(key).upper()

        if char.isalpha():

            df = pd.DataFrame([row+[char]], columns=columns)

            df.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=False,
                index=False
            )

            print("Saved:", char)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Data collection finished")
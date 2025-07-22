'''import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- Load Dino and Cactus Images ---
dino_img = cv2.imread("assets/dino.png", cv2.IMREAD_UNCHANGED)
cactus_img = cv2.imread("assets/cactus.png", cv2.IMREAD_UNCHANGED)
dino_img = cv2.resize(dino_img, (50, 50))
cactus_img = cv2.resize(cactus_img, (50, 50))

# --- Overlay Helper ---
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background_slice = background[y:y+h, x:x+w]
    blended = (1 - mask) * background_slice + mask * overlay_img
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    return background

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Game Settings ---
dino_y = 300
dino_x = 100
jump = False
velocity = 0
gravity = 1
jump_strength = -20  # faster jump
obstacles = []
score = 0
font = cv2.FONT_HERSHEY_SIMPLEX
last_obstacle_time = 0
obstacle_interval = 1.8  # more spacing between cacti
obstacle_speed = 5  # slower movement

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
frame_w, frame_h = 640, 480
cap.set(3, frame_w)
cap.set(4, frame_h)

game_over = False
game_over_time = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not game_over:
        # Hand Tracking
        finger_count = 0
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                lm = hand_landmark.landmark
                finger_tip_ids = [8, 12, 16, 20]
                if lm[4].x < lm[3].x:
                    finger_count += 1
                for tip in finger_tip_ids:
                    if lm[tip].y < lm[tip - 2].y:
                        finger_count += 1
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

        # Jump
        if finger_count == 1 and not jump:
            jump = True
            velocity = jump_strength

        # Dino Physics
        if jump:
            dino_y += velocity
            velocity += gravity
            if dino_y >= 300:
                dino_y = 300
                jump = False

        # Draw Dino
        frame = overlay_transparent(frame, dino_img, dino_x, dino_y)

        # Move and Draw Obstacles
        for obs in obstacles[:]:
            obs[0] -= obstacle_speed
            if obs[0] < -50:
                obstacles.remove(obs)
                score += 1

            # Collision
            if dino_x + 50 > obs[0] and dino_x < obs[0] + 50:
                if dino_y + 50 > obs[1]:
                    game_over = True
                    game_over_time = time.time()

            frame = overlay_transparent(frame, cactus_img, obs[0], obs[1])

        # Spawn Obstacles
        if time.time() - last_obstacle_time > obstacle_interval:
            obstacles.append([frame_w, 300])
            last_obstacle_time = time.time()

        # Display Score
        cv2.putText(frame, f"Score: {score}", (10, 40), font, 1, (0, 0, 0), 2)

    else:
        # Game Over Screen
        cv2.putText(frame, "GAME OVER", (frame_w//4, frame_h//2 - 30), font, 2, (0, 0, 255), 4)
        cv2.putText(frame, f"Final Score: {score}", (frame_w//4, frame_h//2 + 30), font, 1.2, (0, 0, 0), 3)

        if time.time() - game_over_time > 4:  # Show for 4 seconds
            break

    cv2.imshow("Dino Hand Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''


'''import cv2
import mediapipe as mp
import numpy as np
import time
import random

# === Load Dino & Cactus Images ===
dino_img = cv2.imread("assets/dino.png", cv2.IMREAD_UNCHANGED)
cactus_img = cv2.imread("assets/cactus.png", cv2.IMREAD_UNCHANGED)

# Resize for game
dino_img = cv2.resize(dino_img, (50, 50))
cactus_img = cv2.resize(cactus_img, (50, 50))

# === Hand Tracking Setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

# === Overlay Transparent Image ===
def overlay_transparent(background, overlay_img, x, y):
    h, w = overlay_img.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0] or x < 0 or y < 0:
        return background  # Prevent overlay out of bounds

    overlay_rgb = overlay_img[:, :, :3]
    mask = overlay_img[:, :, 3:] / 255.0
    background_slice = background[y:y+h, x:x+w]
    blended = (1 - mask) * background_slice + mask * overlay_rgb
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    return background

# === Game Variables ===
dino_x, dino_y = 100, 300
ground_y = 300
is_jumping = False
jump_velocity = 0

obstacles = [[640, ground_y]]  # Start with one obstacle
obstacle_speed = 6
obstacle_timer = 0
obstacle_interval = random.randint(80, 150)

score = 0
game_over = False

# === Video Capture ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # === Detect Hand & Control Jump ===
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist_y = hand_landmarks.landmark[0].y
            if wrist_y < 0.5 and not is_jumping and not game_over:
                is_jumping = True
                jump_velocity = -25

    # === Dino Jumping ===
    if is_jumping:
        dino_y += jump_velocity
        jump_velocity += 2  # gravity
        if dino_y >= ground_y:
            dino_y = ground_y
            is_jumping = False

    # === Obstacle Movement ===
    if not game_over:
        for obs in obstacles:
            obs[0] -= obstacle_speed

        if obstacles and obstacles[0][0] < -50:
            obstacles.pop(0)
            score += 1

        obstacle_timer += 1
        if obstacle_timer > obstacle_interval:
            obstacles.append([640, ground_y])
            obstacle_timer = 0
            obstacle_interval = random.randint(80, 150)

    # === Collision Detection ===
    for obs in obstacles:
        if (dino_x < obs[0] + 50 and dino_x + 50 > obs[0] and
            dino_y < obs[1] + 50 and dino_y + 50 > obs[1]):
            game_over = True

    # === Draw Dino ===
    frame = overlay_transparent(frame, dino_img, dino_x, dino_y)

    # === Draw Obstacles ===
    for obs in obstacles:
        frame = overlay_transparent(frame, cactus_img, obs[0], obs[1])

    # === Draw Score ===
    cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    # === Game Over ===
    if game_over:
        cv2.putText(frame, "Game Over!", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(frame, f"Final Score: {score}", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # === Display Frame ===
    cv2.imshow("Dino Game - Hand Gesture", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()'''


import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- Load Dino and Cactus Images ---
dino_img = cv2.imread("assets/dino.png", cv2.IMREAD_UNCHANGED)
cactus_img = cv2.imread("assets/cactus.png", cv2.IMREAD_UNCHANGED)
if dino_img is None or cactus_img is None:
    print("Error: Image files not found in assets folder.")
    exit()
dino_img = cv2.resize(dino_img, (50, 50))
cactus_img = cv2.resize(cactus_img, (50, 50))

# --- Overlay Helper ---
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background_slice = background[y:y+h, x:x+w]
    blended = (1 - mask) * background_slice + mask * overlay_img
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    return background

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Game Settings ---
dino_y = 300
dino_x = 100
ground_y = 300
jump = False
velocity = 0
gravity = 1
jump_strength = -20
obstacles = []
score = 0
font = cv2.FONT_HERSHEY_SIMPLEX
last_obstacle_time = 0
obstacle_interval = 1.8
obstacle_speed = 5
game_over = False
game_over_time = None

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
frame_w, frame_h = 640, 480
cap.set(3, frame_w)
cap.set(4, frame_h)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not game_over:
        # Hand Tracking: Count fingers up (jump if only index finger up)
        finger_count = 0
        if result.multi_hand_landmarks:
            hand_landmark = result.multi_hand_landmarks[0]
            lm = hand_landmark.landmark
            fingers = []
            # Thumb
            if lm[4].x < lm[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
            # Four fingers
            for tip_id in [8, 12, 16, 20]:
                if lm[tip_id].y < lm[tip_id - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            finger_count = sum(fingers)
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
        else:
            finger_count = 0

        # Jump: Only when 1 finger is up (typically the index for "point")
        if finger_count == 1 and not jump:
            jump = True
            velocity = jump_strength

        # Dino Physics
        if jump:
            dino_y += velocity
            velocity += gravity
            if dino_y >= ground_y:
                dino_y = ground_y
                jump = False

        # Draw Dino
        frame = overlay_transparent(frame, dino_img, dino_x, dino_y)

        # Move and Draw Obstacles
        for obs in obstacles[:]:
            obs[0] -= obstacle_speed
            if obs[0] < -50:
                obstacles.remove(obs)
                score += 1
            # Collision
            if dino_x + 50 > obs[0] and dino_x < obs[0] + 50:
                if dino_y + 50 > obs[1]:
                    game_over = True
                    game_over_time = time.time()
            frame = overlay_transparent(frame, cactus_img, obs[0], obs[1])

        # Spawn Obstacles
        if time.time() - last_obstacle_time > obstacle_interval:
            obstacles.append([frame_w, ground_y])
            last_obstacle_time = time.time()

        # Display Score
        cv2.putText(frame, f"Score: {score}", (10, 40), font, 1, (0, 0, 0), 2)

    else:
        # Game Over Screen
        cv2.putText(frame, "GAME OVER", (frame_w//4, frame_h//2 - 30), font, 2, (0, 0, 255), 4)
        cv2.putText(frame, f"Final Score: {score}", (frame_w//4, frame_h//2 + 30), font, 1.2, (0, 0, 0), 3)
        if time.time() - game_over_time > 4:
            break

    cv2.imshow("Dino Hand Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



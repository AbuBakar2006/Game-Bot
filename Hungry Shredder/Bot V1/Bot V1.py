import subprocess
import cv2
import numpy as np
import os
import time

TEMPLATE_DIR = "templates"

# Thresholds
FOOD_THRESHOLD = 0.7
SHREDDER_THRESHOLD = 0.6

# Movement control
last_move_time = 0
MOVE_DELAY = 0.3   # seconds

def get_screen():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )
    img = np.frombuffer(result.stdout, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return frame

def load_templates(prefix):
    templates = []
    for file in os.listdir(TEMPLATE_DIR):
        if file.startswith(prefix):
            path = os.path.join(TEMPLATE_DIR, file)
            img = cv2.imread(path, 0)
            if img is not None:
                templates.append(img)
                templates.append(cv2.flip(img, 1))  # flipped version
    return templates

def swipe(direction):
    if direction == "left":
        subprocess.run(["adb", "shell", "input", "swipe", "500", "1500", "100", "1500", "100"])
    elif direction == "right":
        subprocess.run(["adb", "shell", "input", "swipe", "100", "1500", "500", "1500", "100"])

# Load templates
food_templates = load_templates("food")[:2]
shredder_templates = load_templates("shredder")[:1]

frame_count = 0

while True:
    frame = get_screen()
    if frame is None:
        continue

    # Skip frames for speed
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    food_positions = []
    shredder_pos = None

    # 🍔 FOOD detection
    for template in food_templates:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= FOOD_THRESHOLD:
            food_positions.append(max_loc)

    # 🔩 SHREDDER detection
    for template in shredder_templates:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= SHREDDER_THRESHOLD:
            shredder_pos = max_loc
            break

    # 🎯 Choose closest food
    target_food = None
    if food_positions and shredder_pos:
        target_food = min(
            food_positions,
            key=lambda f: abs(f[0] - shredder_pos[0])
        )

    # 🧠 Decide movement
    move = None
    if target_food and shredder_pos:
        if target_food[0] < shredder_pos[0] - 20:
            move = "left"
        elif target_food[0] > shredder_pos[0] + 20:
            move = "right"

    # 🚀 Execute movement with cooldown
    current_time = time.time()
    if move and (current_time - last_move_time > MOVE_DELAY):
        print(f"Moving: {move}")
        swipe(move)
        last_move_time = current_time

    # 🧪 Debug output
    print(f"Food: {len(food_positions)}, Shredder: {shredder_pos}")

    time.sleep(0.05)

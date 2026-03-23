import subprocess
import cv2
import numpy as np
import os
import time

TEMPLATE_DIR = "templates"
THRESHOLD = 0.8

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

# Load templates
food_templates = load_templates("food")
monster_templates = load_templates("monster")
shredder_templates = load_templates("shredder")

while True:
    frame = get_screen()
    if frame is None:
        continue

    # 🔥 Resize early → faster + more stable
    frame = cv2.resize(frame, (540, 960))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    food_positions = []
    monster_positions = []
    shredder_pos = None

    # 🔍 Detect FOOD (ONLY BEST MATCH PER TEMPLATE)
    for template in food_templates:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= THRESHOLD:
            pt = max_loc
            food_positions.append(pt)

            cv2.rectangle(frame, pt,
                          (pt[0] + template.shape[1],
                           pt[1] + template.shape[0]),
                          (0, 255, 0), 2)

    # 🔍 Detect MONSTERS (ONLY BEST MATCH)
    for template in monster_templates:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= THRESHOLD:
            pt = max_loc
            monster_positions.append(pt)

            cv2.rectangle(frame, pt,
                          (pt[0] + template.shape[1],
                           pt[1] + template.shape[0]),
                          (0, 0, 255), 2)

    # 🔍 Detect SHREDDER (FIRST GOOD MATCH)
    for template in shredder_templates:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= THRESHOLD:
            shredder_pos = max_loc

            cv2.rectangle(frame, shredder_pos,
                          (shredder_pos[0] + template.shape[1],
                           shredder_pos[1] + template.shape[0]),
                          (255, 0, 0), 2)
            break

    # 🖥️ Display (smaller)
    display = cv2.resize(frame, (400, 800))
    cv2.imshow("Detection", display)

    # ⏳ Small delay → prevents CPU crash
    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

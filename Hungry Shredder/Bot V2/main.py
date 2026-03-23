import subprocess
import cv2
import numpy as np
import os
import time

TEMPLATE_DIR = "templates"

# ── Screen geometry ──────────────────────────────────────────────────────────
SCREEN_CENTER_X = 353
SCREEN_MID_Y    = 1600   # Y row used for swipe gestures (bottom of screen)

# ── Virtual wall thresholds ──────────────────────────────────────────────────
# When the tracked shredder position crosses these, the bot reverses direction.
#
#      LEFT_WALL          RIGHT_WALL
#          |<─── play area ───>|
#   150   230                 480   580
#
LEFT_WALL  = 230
RIGHT_WALL = 480

# ── How much (in pixels) the shredder moves per single swipe ─────────────────
# Tune this if the bot reverses too early (lower it) or too late (raise it).
# With SWEEP_DISTANCE=430 and the user reporting ~2 swipes to cross the screen,
# one swipe moves roughly half the play area ≈ 215px.
SWIPE_STEP = 215

# ── Sweep settings ───────────────────────────────────────────────────────────
SWEEP_DISTANCE = 430   # px distance of each swipe gesture
SWEEP_DELAY    = 0.05  # seconds between swipes

# ── Shredder optional template snap ─────────────────────────────────────────
# If the shredder IS detected with good confidence, snap est_cx to it.
# This corrects drift over time. Set SHREDDER_THRESHOLD high enough to
# avoid false snaps.
SHREDDER_ZONE_TOP_FRAC = 0.65
SHREDDER_THRESHOLD     = 0.52

# ── Monster detection ────────────────────────────────────────────────────────
MONSTER_THRESHOLD      = 0.60
MONSTER_Y_MAX_FRAC     = 0.65
MONSTER_DANGER_DIST    = 200
FLEE_DISTANCE          = 400
MONSTER_CONFIRM_FRAMES = 2


def get_screen():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )
    img = np.frombuffer(result.stdout, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def load_templates(prefix, with_flip=True):
    templates = []
    for file in sorted(os.listdir(TEMPLATE_DIR)):
        if file.startswith(prefix) and file.endswith(".png"):
            img = cv2.imread(os.path.join(TEMPLATE_DIR, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append(img)
                if with_flip:
                    templates.append(cv2.flip(img, 1))
    return templates


def try_detect_shredder_cx(gray, templates, threshold, y_start):
    """
    Attempt to detect the shredder's horizontal centre in the bottom zone.
    Returns cx (int) if confident, else None.
    """
    region = gray[y_start:, :]
    best_val, best_cx = -1, None
    for tpl in templates:
        if tpl.shape[0] > region.shape[0] or tpl.shape[1] > region.shape[1]:
            continue
        res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_cx  = max_loc[0] + tpl.shape[1] // 2
    return best_cx if best_val >= threshold else None


def find_monsters(gray, templates, threshold, y_max):
    """Return list of (cx, cy) for monsters found above y_max."""
    hits = []
    for tpl in templates:
        if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            h, w = tpl.shape[:2]
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            if cy > y_max:
                continue
            if not any(abs(cx - hx) < 50 and abs(cy - hy) < 50 for hx, hy in hits):
                hits.append((cx, cy))
    return hits


def swipe(direction, distance):
    LEFT_SAFE, RIGHT_SAFE = 150, 580
    if direction == "right":
        x1 = LEFT_SAFE
        x2 = min(RIGHT_SAFE, LEFT_SAFE + distance)
    else:
        x1 = RIGHT_SAFE
        x2 = max(LEFT_SAFE, RIGHT_SAFE - distance)
    subprocess.run([
        "adb", "shell", "input", "swipe",
        str(x1), str(SCREEN_MID_Y), str(x2), str(SCREEN_MID_Y), "30"
    ])


# ── Load templates ───────────────────────────────────────────────────────────
shredder_templates = load_templates("shredder", with_flip=True)
monster_templates  = load_templates("monster",  with_flip=True)
print(f"Shredder: {len(shredder_templates)} templates | Monster: {len(monster_templates)} templates")
print("Strategy: software-tracked sweep with virtual walls. Monster overrides.")

# ── State ────────────────────────────────────────────────────────────────────
sweep_direction     = "right"
est_cx              = SCREEN_CENTER_X   # software-tracked shredder X position
frame_count         = 0
last_swipe_time     = 0
monster_seen_frames = 0

while True:
    now = time.time()

    frame = get_screen()
    if frame is None:
        time.sleep(0.05)
        continue

    frame_count += 1
    h, _ = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Optional: snap est_cx to real detection if confident ─────────────────
    shredder_y0  = int(h * SHREDDER_ZONE_TOP_FRAC)
    detected_cx  = try_detect_shredder_cx(
        gray, shredder_templates, SHREDDER_THRESHOLD, shredder_y0
    )
    if detected_cx is not None:
        est_cx = detected_cx   # correct any drift
        snap_tag = f"snap→{est_cx}"
    else:
        snap_tag = f"est={est_cx}"

    # ── Wall check on estimated position ─────────────────────────────────────
    if est_cx <= LEFT_WALL and sweep_direction == "left":
        print(f"  🧱 Left wall ({snap_tag}) → reversing RIGHT")
        sweep_direction = "right"
    elif est_cx >= RIGHT_WALL and sweep_direction == "right":
        print(f"  🧱 Right wall ({snap_tag}) → reversing LEFT")
        sweep_direction = "left"

    # ── Monster detection ─────────────────────────────────────────────────────
    monster_y_max = int(h * MONSTER_Y_MAX_FRAC)
    monsters  = find_monsters(gray, monster_templates, MONSTER_THRESHOLD, monster_y_max)
    dangerous = [m for m in monsters if abs(m[0] - SCREEN_CENTER_X) < MONSTER_DANGER_DIST]

    if dangerous:
        monster_seen_frames += 1
    else:
        monster_seen_frames = 0

    confirmed_danger = dangerous and monster_seen_frames >= MONSTER_CONFIRM_FRAMES

    # ── Execute swipe ─────────────────────────────────────────────────────────
    if now - last_swipe_time >= SWEEP_DELAY:

        if confirmed_danger:
            nearest = min(dangerous, key=lambda m: abs(m[0] - SCREEN_CENTER_X))
            flee    = "right" if nearest[0] < SCREEN_CENTER_X else "left"
            print(f"  ⚠️  Monster {nearest} (×{monster_seen_frames}) → flee {flee}")
            swipe(flee, FLEE_DISTANCE)
            # Update est_cx for the flee move
            est_cx += SWIPE_STEP * (1 if flee == "right" else -1)
            est_cx  = max(LEFT_WALL, min(RIGHT_WALL, est_cx))
            sweep_direction = flee

        else:
            print(f"  ↔  {sweep_direction}  [{snap_tag}]  [F{frame_count}]")
            swipe(sweep_direction, SWEEP_DISTANCE)
            # Advance estimated position
            est_cx += SWIPE_STEP * (1 if sweep_direction == "right" else -1)
            est_cx  = max(LEFT_WALL, min(RIGHT_WALL, est_cx))

        last_swipe_time = now

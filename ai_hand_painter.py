import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import deque

# AI Hand Painter
# Controls: Q = quit, C = clear, Z = undo, S = save

MIN_BRUSH = 4
MAX_BRUSH = 30
ERASER_THICKNESS = 45
SMOOTHING = 5
UNDO_STACK_SIZE = 15
AUTO_STOP_SECONDS = 20
WHEEL_RADIUS = 70
WHEEL_TOGGLE_COOLDOWN = 0.6

COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Eraser": (0, 0, 0),
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def make_color_wheel(radius):
    size = radius * 2
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - radius, yy - radius
    r = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)

    hue = (theta * 179).astype(np.uint8)
    saturation = np.clip((r / radius) * 255, 0, 255).astype(np.uint8)
    value = np.full_like(hue, 255)

    hsv = cv2.merge([hue, saturation, value])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mask = r <= radius
    return bgr, mask


def smoothed_point(x, y, history):
    history.append((x, y))
    sx = int(np.mean([point[0] for point in history]))
    sy = int(np.mean([point[1] for point in history]))
    return sx, sy


def draw_button_bar(frame, current_color, show_wheel):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (35, 35, 35), -1)

    margin = 8
    item_count = len(COLORS) + 1
    button_width = max((w - margin * (item_count + 1)) // item_count, 40)
    font_scale = 0.55 if button_width >= 90 else 0.42

    x = margin
    boxes = {}

    for name, color in COLORS.items():
        x2 = min(x + button_width, w - 1)
        cv2.rectangle(frame, (x, 10), (x2, 60), color, -1)
        border = (255, 255, 255) if color == current_color else (90, 90, 90)
        cv2.rectangle(frame, (x, 10), (x2, 60), border, 2)

        text_color = (0, 0, 0) if name == "Yellow" else (255, 255, 255)
        cv2.putText(frame, name, (x + 5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 2)
        boxes[name] = (x, x2)
        x += button_width + margin

    x2 = min(x + button_width, w - 1)
    fill = (90, 60, 120) if show_wheel else (60, 60, 60)
    border = (255, 255, 255) if show_wheel else (120, 120, 120)
    cv2.rectangle(frame, (x, 10), (x2, 60), fill, -1)
    cv2.rectangle(frame, (x, 10), (x2, 60), border, 2)
    cv2.putText(frame, "Wheel", (x + 5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2)

    return boxes, (x, x2)


def main():
    os.makedirs("paintings", exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera access.")
        return

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    canvas = None
    draw_color = COLORS["Red"]
    show_wheel = False
    last_wheel_toggle = 0.0
    last_seen = time.time()
    previous_x, previous_y = 0, 0
    stroke_in_progress = False

    point_history = deque(maxlen=SMOOTHING)
    undo_stack = deque(maxlen=UNDO_STACK_SIZE)
    frame_times = deque(maxlen=30)
    wheel_bgr, wheel_mask = make_color_wheel(WHEEL_RADIUS)

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            boxes, wheel_button = draw_button_bar(frame, draw_color, show_wheel)

            wheel_x1 = width - 2 * WHEEL_RADIUS - 15
            wheel_y1 = 80
            wheel_cx = wheel_x1 + WHEEL_RADIUS
            wheel_cy = wheel_y1 + WHEEL_RADIUS

            if show_wheel:
                wheel_x2 = wheel_x1 + 2 * WHEEL_RADIUS
                wheel_y2 = wheel_y1 + 2 * WHEEL_RADIUS

                # Only draw the wheel if it fits in the camera frame.
                if (wheel_x1 >= 0 and wheel_y1 >= 70 and
                        wheel_x2 <= width and wheel_y2 <= height):
                    roi = frame[wheel_y1:wheel_y2, wheel_x1:wheel_x2]
                    roi[wheel_mask] = wheel_bgr[wheel_mask]
                    cv2.circle(frame, (wheel_cx, wheel_cy),
                               WHEEL_RADIUS, (255, 255, 255), 2)
                    cv2.putText(frame, "Color Wheel",
                                (max(0, wheel_x1 - 5), wheel_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

            if result.multi_hand_landmarks:
                last_seen = time.time()

                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    thumb_tip = hand_landmarks.landmark[4]

                    x1 = int(index_tip.x * width)
                    y1 = int(index_tip.y * height)
                    x2 = int(middle_tip.x * width)
                    y2 = int(middle_tip.y * height)
                    xt = int(thumb_tip.x * width)
                    yt = int(thumb_tip.y * height)

                    selection_distance = np.hypot(x2 - x1, y2 - y1)
                    pinch_distance = np.hypot(xt - x1, yt - y1)

                    if y1 < 70:
                        if wheel_button[0] < x1 < wheel_button[1]:
                            if time.time() - last_wheel_toggle > WHEEL_TOGGLE_COOLDOWN:
                                show_wheel = not show_wheel
                                last_wheel_toggle = time.time()
                        else:
                            for name, (left, right) in boxes.items():
                                if left < x1 < right:
                                    draw_color = COLORS[name]

                        previous_x, previous_y = 0, 0
                        point_history.clear()
                        stroke_in_progress = False

                    elif show_wheel and np.hypot(x1 - wheel_cx, y1 - wheel_cy) <= WHEEL_RADIUS:
                        local_x = x1 - wheel_x1
                        local_y = y1 - wheel_y1

                        if (0 <= local_x < wheel_bgr.shape[1] and
                                0 <= local_y < wheel_bgr.shape[0] and
                                wheel_mask[local_y, local_x]):
                            picked = wheel_bgr[local_y, local_x]
                            draw_color = tuple(int(v) for v in picked)

                        previous_x, previous_y = 0, 0
                        point_history.clear()
                        stroke_in_progress = False

                    elif selection_distance > 40:
                        sx, sy = smoothed_point(x1, y1, point_history)

                        thickness = int(np.interp(
                            pinch_distance, [20, 150],
                            [MIN_BRUSH, MAX_BRUSH]
                        ))
                        thickness = max(MIN_BRUSH, min(MAX_BRUSH, thickness))

                        if draw_color == COLORS["Eraser"]:
                            thickness = ERASER_THICKNESS

                        if not stroke_in_progress:
                            undo_stack.append(canvas.copy())
                            stroke_in_progress = True

                        cv2.circle(frame, (sx, sy), thickness // 2 + 2,
                                   draw_color, -1)

                        if previous_x == 0 and previous_y == 0:
                            previous_x, previous_y = sx, sy

                        cv2.line(canvas, (previous_x, previous_y),
                                 (sx, sy), draw_color, thickness)
                        previous_x, previous_y = sx, sy

                    else:
                        previous_x, previous_y = 0, 0
                        point_history.clear()
                        stroke_in_progress = False

            elif time.time() - last_seen > AUTO_STOP_SECONDS:
                print(f"No hand detected for {AUTO_STOP_SECONDS} seconds. Exiting.")
                break

            # Composite drawing canvas over the webcam feed.
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, inverse_mask = cv2.threshold(
                gray, 15, 255, cv2.THRESH_BINARY_INV
            )
            inverse_mask = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inverse_mask)
            frame = cv2.bitwise_or(frame, canvas)

            # Current colour swatch and keyboard hints.
            cv2.rectangle(frame, (10, height - 60), (70, height - 10),
                          draw_color, -1)
            cv2.rectangle(frame, (10, height - 60), (70, height - 10),
                          (255, 255, 255), 2)
            cv2.putText(frame, "Color", (10, height - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Q: Quit | C: Clear | Z: Undo | S: Save",
                        (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (220, 220, 220), 1)

            frame_times.append(time.time() - start_time)
            fps = 1.0 / (sum(frame_times) / len(frame_times) + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (width - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("AI Hand Painter", frame)
            cv2.imshow("Canvas", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit command received.")
                break
            elif key == ord("c"):
                undo_stack.append(canvas.copy())
                canvas = np.zeros_like(canvas)
                previous_x, previous_y = 0, 0
                point_history.clear()
                stroke_in_progress = False
                print("Canvas cleared.")
            elif key == ord("z"):
                if undo_stack:
                    canvas = undo_stack.pop()
                    previous_x, previous_y = 0, 0
                    point_history.clear()
                    stroke_in_progress = False
                    print("Undo successful.")
                else:
                    print("Nothing to undo.")
            elif key == ord("s"):
                filename = os.path.join(
                    "paintings", f"painting_{int(time.time())}.png"
                )
                if cv2.imwrite(filename, canvas):
                    print(f"Painting saved to: {filename}")
                else:
                    print("Could not save painting.")

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

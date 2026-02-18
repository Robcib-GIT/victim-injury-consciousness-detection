import json
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from websocket import create_connection
import os

from bbbox import draw_bbox
from evaluator import ask_one_question
from go2_tts import say_on_go2, WebRTCConnectionMethod


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIC_PATH = os.path.join(BASE_DIR, "assesment.json")

YOLO_MODEL_PATH = "yolov8n.pt"
WS_URL = "ws://192.168.0.129:8765/stream"


latest_result = None
processing = False


# ---------- PROCESS IMAGE ----------
def process_image(img, logic, yolo, img_index):
    print(f"\nProcessing image {img_index}")

    results = yolo(img, classes=[0], conf=0.45, iou=0.25, agnostic_nms=True)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No person detected")
        return None, []

    boxes = sorted(boxes, key=lambda b: int(b.xyxy[0][0]))

    labeled_img = img.copy()
    coords = []
    all_descriptions = []

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        coords.append((i, x1, y1, x2, y2))

        labeled_img = draw_bbox(
            labeled_img, x1, y1, x2, y2,
            label=f"person_{i}",
            triage=None
        )

    context_img = labeled_img.copy()
    result_img = labeled_img.copy()

    ok, buf = cv2.imencode(".jpg", context_img)
    context_bytes = buf.tobytes()

    # --- decision tree ---
    for (i, x1, y1, x2, y2) in coords:
        current = logic["start_condition"]
        triage_result = None

        while True:
            node = logic[current]

            answer = ask_one_question(
                crop_bytes=context_bytes,
                question=f"""
You are analyzing a rescue scene.
The person of interest is clearly marked with a bounding box label: person_{i}.
Look at the bounding box around person_{i} in the image.
{node['question']}
""",
                choices=node["choices"]
            )

            all_descriptions.append(
                f"person_{i} | {node['question']} -> {answer}"
            )

            rule = node["outcomes"][answer]

            if "triage" in rule:
                if triage_result is None:
                    triage_result = rule["triage"]

                if "next" in rule:
                    current = rule["next"]
                    continue
                else:
                    break
            else:
                current = rule["next"]

        result_img = draw_bbox(
            result_img, x1, y1, x2, y2,
            label=f"person_{i}",
            triage=triage_result
        )

        all_descriptions.append(
            f"person_{i} >>> TRIAGE (final) = {triage_result}"
        )

    return result_img, all_descriptions



print("Loading logic and model...")
with open(LOGIC_PATH, "r", encoding="utf-8") as f:
    logic = json.load(f)

yolo = YOLO(YOLO_MODEL_PATH)



print("Connecting to robot stream...")
ws = create_connection(WS_URL, timeout=5)
print("Connected to WebSocket!")
ws.settimeout(0.01)

cv2.namedWindow("Triage Result", cv2.WINDOW_NORMAL)

img_index = 1


def run_processing(frame, index):
    global latest_result, processing

    result_img, descriptions = process_image(frame, logic, yolo, index)

    if result_img is not None:
        latest_result = result_img

        print("\n--- DECISION LOG ---")
        for line in descriptions:
            print(line)

    processing = False


while True:
    try:
        last_frame = None

        # --- clear buffer ---
        while True:
            try:
                message = ws.recv()
                if isinstance(message, bytes):
                    last_frame = message
            except:
                break

        if last_frame is None:
            continue

        nparr = np.frombuffer(last_frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # --- SHOW IMAGE (MAIN THREAD ONLY) ---
        if latest_result is not None:
            cv2.imshow("Triage Result", latest_result)
        else:
            cv2.imshow("Triage Result", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        if not processing:
            processing = True

            threading.Thread(
                target=run_processing,
                args=(img.copy(), img_index),
                daemon=True
            ).start()

            img_index += 1

    except Exception as e:
        print("Connection error:", e)
        break

ws.close()
cv2.destroyAllWindows()

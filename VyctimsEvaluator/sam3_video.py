import os
import asyncio
import multiprocessing as mp
from collections import deque
from typing import Dict, List, Optional, Tuple
import threading
import uuid
import shutil

import cv2
import numpy as np
import websockets

# -------------- CONFIG --------------
WS_URL = "ws://192.168.0.129:8765/stream"
INFER_DOWNSCALE_WIDTH = 512
DETECT_EVERY = 5
DEVICE = "cuda"
MAX_WS_SIZE = None

# prompts editables
current_prompts = ["person"]
prompts_lock = threading.Lock()

RAM_TMP = "/dev/shm/sam3_frames"
os.makedirs(RAM_TMP, exist_ok=True)
# -----------------------------------


def update_prompts_from_terminal():
    global current_prompts
    while True:
        new = input("\nNew prompts (comma separated): ")
        new_list = [p.strip() for p in new.split(",") if p.strip()]
        if not new_list:
            print("No prompts entered.")
            continue
        with prompts_lock:
            current_prompts = new_list
            print("Updated prompts ->", current_prompts)


def downscale(frame: np.ndarray, target_w: Optional[int]) -> np.ndarray:
    if target_w is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    s = target_w / float(w)
    return cv2.resize(frame, (target_w, int(h * s)), interpolation=cv2.INTER_AREA)


def outputs_to_masks(outputs: Optional[dict], target_shape_hw: Tuple[int, int]) -> List[np.ndarray]:
    if outputs is None:
        return []
    masks = outputs.get("out_binary_masks", None)
    if masks is None:
        return []
    if not isinstance(masks, np.ndarray):
        # a veces puede venir como lista
        try:
            masks = np.array(masks)
        except Exception:
            return []

    if masks.ndim == 2:
        masks = masks[None, ...]

    th, tw = target_shape_hw
    out: List[np.ndarray] = []
    for m in masks:
        mu8 = (m.astype(np.uint8) * 255)
        if mu8.shape[:2] != (th, tw):
            mu8 = cv2.resize(mu8, (tw, th), interpolation=cv2.INTER_NEAREST)
        out.append(mu8)
    return out


def overlay_masks(frame: np.ndarray, masks: List[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    if not masks:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, m)
    mbool = combined.astype(bool)
    if not mbool.any():
        return out
    color = np.array([0, 255, 0], dtype=np.float32)  # green
    out_m = out[mbool].astype(np.float32)
    out[mbool] = ((1 - alpha) * out_m + alpha * color).astype(np.uint8)
    return out


# ---------------- WORKER ----------------
def sam3_worker(in_q, out_q, device: str):
    import torch
    import torch.nn as nn
    from sam3.model_builder import build_sam3_video_predictor

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    predictor = build_sam3_video_predictor(async_loading_frames=False, gpus_to_use=[0] if device == "cuda" else [])

    # FP16 excepto texto/LayerNorm
    if device == "cuda":
        model = predictor.model.cuda().half()
        keep_fp32 = ("text", "clip", "prompt", "token", "language", "bert", "transformer", "encoder")
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm) or any(k in name.lower() for k in keep_fp32):
                module.float()
        predictor.model = model
        torch.cuda.empty_cache()

    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu")

    while True:
        item = in_q.get()
        if item is None:
            break

        frame_idx, prompt, infer_frame = item

        session_dir = os.path.join(RAM_TMP, uuid.uuid4().hex)
        os.makedirs(session_dir, exist_ok=True)
        jpg_path = os.path.join(session_dir, "000000.jpg")

        try:
            cv2.imwrite(jpg_path, infer_frame)

            with torch.inference_mode(), autocast_ctx:
                sid = predictor.handle_request({"type": "start_session", "resource_path": session_dir})["session_id"]

                # IMPORTANT: frame_index=0 always, because we have 1 frame.
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": prompt
                })

                outputs = None
                for r in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
                    if r.get("frame_index") == 0:
                        outputs = r.get("outputs")

                predictor.handle_request({"type": "close_session", "session_id": sid})

            # DEBUG: report what we got
            if outputs is None:
                out_q.put((frame_idx, prompt, None, "NO_OUTPUTS"))
            else:
                keys = list(outputs.keys())
                masks = outputs.get("out_binary_masks", None)
                nm = -1
                if isinstance(masks, np.ndarray):
                    nm = (1 if masks.ndim == 2 else masks.shape[0])
                elif masks is not None:
                    try:
                        arr = np.array(masks)
                        nm = (1 if arr.ndim == 2 else arr.shape[0])
                    except Exception:
                        nm = -2
                probs = outputs.get("out_probs", None)
                out_q.put((frame_idx, prompt, outputs, f"keys={keys}, num_masks={nm}, probs={probs}"))

        except Exception as e:
            out_q.put((frame_idx, prompt, None, f"EXC: {e}"))

        finally:
            shutil.rmtree(session_dir, ignore_errors=True)


# ---------------- MAIN LOOP ----------------
async def main_loop():
    mp.set_start_method("spawn", force=True)
    threading.Thread(target=update_prompts_from_terminal, daemon=True).start()

    in_q, out_q = mp.Queue(16), mp.Queue(16)
    worker = mp.Process(target=sam3_worker, args=(in_q, out_q, DEVICE), daemon=True)
    worker.start()

    frame_buffer = deque(maxlen=240)
    last_masks_by_prompt: Dict[str, List[np.ndarray]] = {}
    last_det = -10**9

    async with websockets.connect(WS_URL, max_size=MAX_WS_SIZE) as ws:
        frame_idx = 0
        while True:
            data = await ws.recv()
            arr = np.frombuffer(data, np.uint8)
            raw_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if raw_frame is None:
                continue

            frame_buffer.append((frame_idx, raw_frame.copy()))
            infer_frame = downscale(raw_frame, INFER_DOWNSCALE_WIDTH)

            with prompts_lock:
                prompts_snapshot = list(current_prompts)

            # enqueue
            if frame_idx - last_det >= DETECT_EVERY and in_q.qsize() <= 4:
                for p in prompts_snapshot:
                    in_q.put((frame_idx, p, infer_frame))
                last_det = frame_idx

            # collect
            while not out_q.empty():
                idx, prompt, outputs, info = out_q.get()
                print(f"[sam3] frame={idx} prompt='{prompt}' -> {info}")

                if outputs is None:
                    last_masks_by_prompt[prompt] = []
                    continue

                # find synced raw frame
                synced_raw = None
                for f_idx, f in reversed(frame_buffer):
                    if f_idx == idx:
                        synced_raw = f
                        break
                if synced_raw is None:
                    continue

                last_masks_by_prompt[prompt] = outputs_to_masks(outputs, synced_raw.shape[:2])

            # display (single prompt overlay for simplicity)
            disp = raw_frame.copy()
            # overlay all prompts combined
            all_masks = []
            for p in prompts_snapshot:
                all_masks.extend(last_masks_by_prompt.get(p, []))
            disp = overlay_masks(disp, all_masks)

            cv2.imshow("SAM3 debug", disp)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            frame_idx += 1

    in_q.put(None)
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main_loop())

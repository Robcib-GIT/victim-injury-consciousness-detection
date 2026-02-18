import os

# 🔇 Intenta silenciar tqdm / logs internos (según build)
os.environ["TQDM_DISABLE"] = "1"
os.environ["SAM3_DISABLE_TQDM"] = "1"

# -------------------------------------------------
# SILENCE SAM3 INTERNAL LOGGER (the real fix)
# -------------------------------------------------
import logging

logging.getLogger("sam3").setLevel(logging.ERROR)
logging.getLogger("sam3_video_predictor").setLevel(logging.ERROR)
logging.getLogger("sam3.model").setLevel(logging.ERROR)
logging.getLogger("sam3.utils").setLevel(logging.ERROR)

# también quita handlers existentes (muy importante)
for name in list(logging.root.manager.loggerDict.keys()):
    if "sam3" in name:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = False


import asyncio
import multiprocessing as mp
from collections import deque
from typing import Dict, List, Optional, Tuple
import threading

import cv2
import numpy as np
import websockets

# ---------------- CONFIG ----------------
WS_URL = "ws://192.168.0.129:8766/stream"

# Solo para INFERENCIA (el display será full-res)
INFER_DOWNSCALE_WIDTH = 384

DETECT_EVERY = 5          # subir = menos carga
DEVICE = "cuda"            # "cpu" si lo necesitas
MAX_WS_SIZE = None
SHOW = True
# ----------------------------------------

# 🔥 prompts editables en caliente
current_prompts = ["person", "cup"]
prompts_lock = threading.Lock()


# =========================================================
# PROMPT EDITOR THREAD
# =========================================================
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


# =========================================================
# UTILS
# =========================================================
def downscale(frame: np.ndarray, target_w: Optional[int]) -> np.ndarray:
    if target_w is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    s = target_w / float(w)
    return cv2.resize(frame, (target_w, int(h * s)), interpolation=cv2.INTER_AREA)


def outputs_to_masks(outputs: Optional[dict], target_shape_hw: Tuple[int, int]) -> List[np.ndarray]:
    """
    Convierte outputs SAM3 -> lista de máscaras uint8 (0/255) reescaladas a target_shape_hw.
    Nunca crashea.
    """
    if outputs is None:
        return []

    masks = outputs.get("out_binary_masks", None)
    if masks is None or not isinstance(masks, np.ndarray):
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


def overlay_masks_by_prompt(
    frame: np.ndarray,
    masks_by_prompt: Dict[str, List[np.ndarray]],
    prompts_snapshot: List[str],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay por prompt (colores/labels). Evita saturación: combina instancias por prompt.
    Mezcla SOLO donde máscara=True (no “cuadrados blancos”).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    palette = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 255, 128),
        (255, 128, 128),
    ]
    prompt_colors = {p: palette[i % len(palette)] for i, p in enumerate(prompts_snapshot)}

    for prompt in prompts_snapshot:
        masks = masks_by_prompt.get(prompt, [])
        if not masks:
            continue

        # combinar instancias de ese prompt en una sola máscara
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = cv2.bitwise_or(combined, m)

        mbool = combined.astype(bool)
        if not mbool.any():
            continue

        b, g, r = prompt_colors[prompt]
        color = np.array([b, g, r], dtype=np.float32)

        out_m = out[mbool].astype(np.float32)
        out[mbool] = ((1 - alpha) * out_m + alpha * color).astype(np.uint8)

    # etiquetas
    y = 24
    for prompt in prompts_snapshot:
        if masks_by_prompt.get(prompt):
            b, g, r = prompt_colors[prompt]
            cv2.putText(out, prompt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (b, g, r), 2, cv2.LINE_AA)
            y += 26

    return out


# =========================================================
# SAM3 WORKER
# =========================================================
def sam3_worker(in_q, out_q, device: str):
    # --- silenciar TODO en el proceso hijo (robusto) ---
    import os
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["SAM3_DISABLE_TQDM"] = "1"

    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    # (1) Importa sam3 y DESPUÉS desactiva loggers concretos que ya existen
    import tempfile, shutil
    import torch
    import torch.nn as nn
    import cv2

    from sam3.model_builder import build_sam3_video_predictor

    # ⚠️ IMPORTANTE: desactivar el logger del módulo que imprime "removed session ..."
    try:
        # dependiendo del paquete, el módulo puede llamarse así o similar
        import sam3.sam3_video_predictor as svp
    except Exception:
        svp = None

    # Desactiva todos los loggers que contengan "sam3"
    for name in list(logging.root.manager.loggerDict.keys()):
        if "sam3" in name:
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.propagate = False
            lg.setLevel(logging.CRITICAL)
            lg.disabled = True

    # Y si pudimos importar el módulo, apagamos SU logger global directamente
    if svp is not None and hasattr(svp, "logger"):
        try:
            svp.logger.handlers = []
            svp.logger.propagate = False
            svp.logger.setLevel(logging.CRITICAL)
            svp.logger.disabled = True
        except Exception:
            pass

    # --- construir predictor ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    predictor = build_sam3_video_predictor(
        async_loading_frames=False,
        gpus_to_use=[0] if device == "cuda" else [],
    )

    # FP16 excepto texto/LayerNorm en FP32
    if device == "cuda":
        model = predictor.model.cuda().half()
        keep_fp32 = ("text", "clip", "prompt", "token", "language", "bert", "transformer", "encoder")
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm) or any(k in name.lower() for k in keep_fp32):
                module.float()
        for p in model.parameters():
            p.requires_grad_(False)
        predictor.model = model
        torch.cuda.empty_cache()

    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu")

    # --- loop ---
    while True:
        item = in_q.get()
        if item is None:
            break

        frame_idx, prompt, infer_frame = item
        tmp = tempfile.mkdtemp()
        try:
            cv2.imwrite(f"{tmp}/000000.jpg", infer_frame)

            with torch.inference_mode(), autocast_ctx:
                sid = predictor.handle_request({"type": "start_session", "resource_path": tmp})["session_id"]
                predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})

                outputs = None
                for r in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
                    if r.get("frame_index") == 0:
                        outputs = r.get("outputs")

                predictor.handle_request({"type": "close_session", "session_id": sid})

            if device == "cuda":
                torch.cuda.empty_cache()

            out_q.put((frame_idx, prompt, outputs, None))

        except Exception as e:
            out_q.put((frame_idx, prompt, None, str(e)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)



# =========================================================
# MAIN LOOP
# =========================================================
async def main_loop():
    mp.set_start_method("spawn", force=True)

    # hilo para prompts en caliente
    threading.Thread(target=update_prompts_from_terminal, daemon=True).start()

    in_q, out_q = mp.Queue(16), mp.Queue(16)
    worker = mp.Process(target=sam3_worker, args=(in_q, out_q, DEVICE), daemon=True)
    worker.start()

    # buffer RAW (full-res)
    frame_buffer = deque(maxlen=240)

    last_masks_by_prompt: Dict[str, List[np.ndarray]] = {}
    last_det = -10**9

    async with websockets.connect(WS_URL, max_size=MAX_WS_SIZE) as ws:
        frame_idx = 0
        while True:
            data = await ws.recv()
            if not isinstance(data, (bytes, bytearray)):
                continue

            arr = np.frombuffer(data, np.uint8)
            raw_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if raw_frame is None:
                continue

            frame_buffer.append((frame_idx, raw_frame.copy()))

            # frame pequeño SOLO para inferencia
            infer_frame = downscale(raw_frame, INFER_DOWNSCALE_WIDTH)

            # snapshot prompts
            with prompts_lock:
                prompts_snapshot = list(current_prompts)

            # limpiar prompts eliminados
            for k in list(last_masks_by_prompt.keys()):
                if k not in prompts_snapshot:
                    del last_masks_by_prompt[k]

            # enqueue jobs (1 por prompt)
            if frame_idx - last_det >= DETECT_EVERY and in_q.qsize() <= 4:
                for p in prompts_snapshot:
                    in_q.put((frame_idx, p, infer_frame))
                last_det = frame_idx

            # recoger resultados (reescalando máscaras a RAW)
            while not out_q.empty():
                idx, prompt, outputs, err = out_q.get()
                if err:
                    last_masks_by_prompt[prompt] = []
                    continue

                synced_raw = None
                for f_idx, f in reversed(frame_buffer):
                    if f_idx == idx:
                        synced_raw = f
                        break
                if synced_raw is None:
                    continue

                last_masks_by_prompt[prompt] = outputs_to_masks(outputs, synced_raw.shape[:2])

            # display: FULL-RES + overlays
            display = overlay_masks_by_prompt(raw_frame, last_masks_by_prompt, prompts_snapshot, alpha=0.45)
            cv2.imshow("SAM3 live prompts (silent)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1

    try:
        in_q.put(None)
    except Exception:
        pass
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main_loop())

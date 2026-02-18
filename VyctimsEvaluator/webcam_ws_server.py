import asyncio
import signal
from typing import Set

import cv2
import websockets
from websockets.asyncio.server import ServerConnection

HOST = "127.0.0.1"
PORT = 8765
PATH = "/stream"

# Webcam + encoding settings
CAM_INDEX = 0          # 0 = default webcam
FPS = 20               # target send fps
JPEG_QUALITY = 80      # 0-100
MAX_QUEUE = 2          # backpressure

clients: Set[ServerConnection] = set()
stop_event = asyncio.Event()


async def stream_frames():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing CAM_INDEX.")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    frame_interval = 1.0 / max(FPS, 1)

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.05)
                continue

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok and clients:
                payload = buf.tobytes()
                dead = []
                for conn in list(clients):
                    try:
                        await conn.send(payload)  # binary JPEG
                    except Exception:
                        dead.append(conn)
                for conn in dead:
                    clients.discard(conn)

            # Sleep but still respond quickly to shutdown
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=frame_interval)
            except asyncio.TimeoutError:
                pass
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


async def handler(conn: ServerConnection):
    # websockets v12+: request path is here
    req_path = conn.request.path if conn.request else None
    if req_path != PATH:
        await conn.close(code=1008, reason="Invalid path")
        return

    clients.add(conn)
    try:
        async for _ in conn:  # ignore incoming messages
            pass
    finally:
        clients.discard(conn)


async def main():
    server = await websockets.serve(
        handler,
        HOST,
        PORT,
        max_queue=MAX_QUEUE,
        ping_interval=20,
        ping_timeout=20,
    )
    print(f"Webcam stream server running at ws://{HOST}:{PORT}{PATH}")
    try:
        await stream_frames()
    finally:
        server.close()
        await server.wait_closed()


def _shutdown(*_):
    stop_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    asyncio.run(main())

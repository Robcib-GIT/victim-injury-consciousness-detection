import logging
import asyncio
import os
import sys
import signal
import contextlib

import edge_tts
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from aiortc.contrib.media import MediaPlayer

logging.basicConfig(level=logging.INFO)
logging.getLogger("aioice").setLevel(logging.INFO)
logging.getLogger("aiortc").setLevel(logging.INFO)
logging.getLogger("unitree_webrtc_connect").setLevel(logging.INFO)


async def tts_to_mp3(texto: str, mp3_path: str, voice: str = "es-ES-AlvaroNeural") -> None:
    """Convierte 'texto' a voz y lo guarda como MP3 en 'mp3_path' usando Edge TTS."""
    out_dir = os.path.dirname(mp3_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    communicate = edge_tts.Communicate(text=texto, voice=voice)
    await communicate.save(mp3_path)


def estimate_seconds_from_text(texto: str) -> float:
    """
    Estimación simple (sin ffprobe):
    - habla normal ~ 12-14 caracteres/segundo
    - añadimos margen para arranque/latencia
    Ajusta si lo necesitas.
    """
    cps = 13.0
    base = max(1, len(texto)) / cps
    return max(2.5, min(base + 1.5, 15.0))  # mínimo 2.5s, máximo 15s


async def safe_shutdown(conn: UnitreeWebRTCConnection | None, player: MediaPlayer | None):
    # 1) Para ffmpeg (MediaPlayer)
    if player is not None:
        with contextlib.suppress(Exception):
            await player.stop()

    # 2) Cierra PeerConnection limpio
    if conn is not None and getattr(conn, "pc", None) is not None:
        with contextlib.suppress(Exception):
            await conn.pc.close()

    # 3) Si el driver tiene disconnect/close, úsalo también
    if conn is not None:
        for m in ("disconnect", "close"):
            if hasattr(conn, m):
                with contextlib.suppress(Exception):
                    res = getattr(conn, m)()
                    if asyncio.iscoroutine(res):
                        await res

    # 4) Deja “respirar” el loop para que no queden tareas pendientes
    await asyncio.sleep(0.3)


async def main():
    conn = None
    player = None

    # Permite terminar limpio con Ctrl+C / SIGTERM
    stop_event = asyncio.Event()

    def _request_stop(*_):
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_stop)

        # 1) Texto
        texto = "Hola, soy César."

        # 2) MP3 output
        mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_output.mp3")

        # 3) TTS -> MP3
        logging.info("Generando MP3 con TTS...")
        await tts_to_mp3(texto, mp3_path, voice="es-ES-AlvaroNeural")
        logging.info(f"MP3 generado: {mp3_path}")

        # 4) Conexión al robot (approach inicial)
        # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
        # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
        # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX",
        #                                username="email@gmail.com", password="pass")
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

        logging.info("Conectando al robot...")
        await conn.connect()
        logging.info("Conectado.")

        # 5) Enviar el MP3 como track de audio (requiere ffmpeg)
        logging.info(f"Enviando audio MP3 al robot: {mp3_path}")
        player = MediaPlayer(mp3_path)
        audio_track = player.audio
        if audio_track is None:
            raise RuntimeError("No se pudo cargar la pista de audio. ¿Tienes ffmpeg instalado y en PATH?")

        conn.pc.addTrack(audio_track)

        # 6) Esperar sólo lo suficiente para que el audio suene UNA vez y salir
        play_seconds = estimate_seconds_from_text(texto)
        logging.info(f"Reproduciendo una vez (~{play_seconds:.1f}s)...")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0 + play_seconds)
        except asyncio.TimeoutError:
            pass

        logging.info("Fin de reproducción. Cerrando...")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)

    finally:
        await safe_shutdown(conn, player)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)

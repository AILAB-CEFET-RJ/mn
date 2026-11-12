"""Fábrica de cliente Qdrant com suporte a servidor Docker (:6333) e fallback local em disco."""
import atexit
import logging
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from qdrant_client import QdrantClient

from app.config import QDRANT_API_KEY, QDRANT_LOCAL_DIR, QDRANT_URL

log = logging.getLogger("qdrant_db")
DEFAULT_URL = QDRANT_URL
LOCAL_PATH = QDRANT_LOCAL_DIR
_CLIENT: QdrantClient | None = None


def _is_server_up(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 6333)
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def get_qdrant_client(timeout: int = 120) -> QdrantClient:
    """Retorna singleton QdrantClient (servidor HTTP/Cloud se ativo, senão persistência local em disco)."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    forced_path = os.getenv("QDRANT_PATH")
    url = os.getenv("QDRANT_URL", DEFAULT_URL)
    api_key = os.getenv("QDRANT_API_KEY", QDRANT_API_KEY)

    if forced_path:
        Path(forced_path).mkdir(parents=True, exist_ok=True)
        log.info("Qdrant em modo local (QDRANT_PATH=%s)", forced_path)
        _CLIENT = QdrantClient(path=forced_path)
    elif api_key or _is_server_up(url):
        log.info("Conectado ao Qdrant Server em %s", url)
        _CLIENT = QdrantClient(url=url, api_key=api_key, timeout=timeout)
    else:
        LOCAL_PATH.mkdir(parents=True, exist_ok=True)
        log.info("Qdrant Server (%s) indisponível; usando persistência local em %s", url, LOCAL_PATH)
        _CLIENT = QdrantClient(path=str(LOCAL_PATH))
    return _CLIENT


def close_qdrant_client() -> None:
    """Fecha explicitamente o cliente Qdrant singleton no encerramento da aplicação."""
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None


atexit.register(close_qdrant_client)



"""
Utilidades: configuración de logging, guardado de registros de ejecución.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
DATA_DIR = Path(__file__).parent.parent / "data"


def setup_logging(level: str = "INFO") -> None:
    """
    Configura logging a consola y archivo rotativo.

    Args:
        level: Nivel de log ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Handler de archivo rotativo (1 MB, guarda 7 archivos)
    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "bot.log",
        maxBytes=1_000_000,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # siempre verbose en archivo

    # Configurar root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silenciar librerías ruidosas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def save_run_record(paper, tweets: list[str]) -> None:
    """
    Guarda un registro JSON de cada ejecución exitosa.
    Útil para evitar duplicados y auditar el historial.

    Args:
        paper: Objeto Paper con los metadatos del artículo.
        tweets: Lista de tweets publicados.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record_file = DATA_DIR / "run_history.json"

    # Cargar historial existente
    history = []
    if record_file.exists():
        try:
            history = json.loads(record_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

    # Agregar nuevo registro
    record = {
        "timestamp": datetime.now().isoformat(),
        "paper_url": paper.url,
        "paper_title": paper.title,
        "tweets_count": len(tweets),
        "tweets": tweets,
    }
    history.append(record)

    record_file.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.getLogger(__name__).info(f"Registro guardado en {record_file}")


def was_paper_posted(paper_url: str) -> bool:
    """
    Verifica si un paper ya fue publicado anteriormente.

    Args:
        paper_url: URL del paper a verificar.

    Returns:
        True si ya fue publicado.
    """
    record_file = DATA_DIR / "run_history.json"
    if not record_file.exists():
        return False
    try:
        history = json.loads(record_file.read_text(encoding="utf-8"))
        posted_urls = {r["paper_url"] for r in history}
        return paper_url in posted_urls
    except Exception:
        return False
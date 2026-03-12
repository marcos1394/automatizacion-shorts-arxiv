"""
youtube_uploader.py — Auto-upload a YouTube Short via Data API v3.

Setup (una sola vez):
  1. Crear proyecto en https://console.cloud.google.com/
  2. Activar YouTube Data API v3
  3. Crear credenciales OAuth 2.0 → Tipo: Desktop app
  4. Descargar client_secret.json → colocar en raíz del proyecto
  5. pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
  6. python -m src.youtube_uploader --auth   ← abre browser, autoriza una vez
     Guarda token en youtube_token.json (no lo commitees)

Uso posterior (automático, sin browser):
  uploader = YouTubeUploader()
  video_id = uploader.upload_short(video_path, paper)
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CLIENT_SECRETS  = "client_secret.json"
TOKEN_FILE      = "youtube_token.json"
SCOPES          = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
CATEGORY_SCIENCE = "28"   # Science & Technology
MAX_RETRIES     = 5
CHUNK_SIZE      = 256 * 1024   # 256 KB chunks — stable on slow connections

# Tags base para todos los Shorts — se combinan con tags específicos del paper
BASE_TAGS = [
    "AI research", "artificial intelligence", "machine learning",
    "deep learning", "arXiv", "science explained", "AI shorts",
    "research paper", "tech explained",
]


def _build_description(paper) -> str:
    """Generate YouTube description with abstract and links."""
    abstract = paper.abstract[:800] + "..." if len(paper.abstract) > 800 else paper.abstract
    return (
        f"📄 Paper: {paper.title}\n\n"
        f"🔗 arXiv: {paper.url}\n\n"
        f"📝 Abstract:\n{abstract}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Generated automatically with local AI (Qwen 2.5 + FLUX.1-schnell + Kokoro-82M)\n"
        f"📚 Daily AI research breakdowns — Subscribe for more!\n"
        f"#AIResearch #MachineLearning #arXiv #ScienceExplained"
    )


def _build_tags(paper) -> list[str]:
    """Combine base tags with paper-specific keywords."""
    import re
    stopwords = {"the", "a", "an", "of", "in", "on", "for", "with", "and",
                 "or", "is", "are", "we", "our", "via", "using", "based",
                 "towards", "new", "novel", "large", "model", "paper"}
    paper_words = re.findall(r'\b[a-zA-Z]{4,}\b', paper.title)
    paper_tags = [w for w in paper_words if w.lower() not in stopwords][:6]
    all_tags = BASE_TAGS + paper_tags
    return list(dict.fromkeys(all_tags))[:30]   # YouTube max 30 tags


class YouTubeUploader:
    """
    Uploads videos to YouTube via Data API v3.
    OAuth token cached in youtube_token.json — solo se autoriza una vez.
    """

    def __init__(self, secrets_file: str = CLIENT_SECRETS,
                 token_file: str = TOKEN_FILE):
        self.secrets_file = Path(secrets_file)
        self.token_file   = Path(token_file)
        self._service     = None

    def _get_service(self):
        """Build authenticated YouTube service. Refreshes token automatically."""
        if self._service:
            return self._service

        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None

        # Load cached token
        if self.token_file.exists():
            with open(self.token_file) as f:
                token_data = json.load(f)
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)

        # Refresh or re-authorize
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing YouTube OAuth token...")
                creds.refresh(Request())
            else:
                if not self.secrets_file.exists():
                    raise FileNotFoundError(
                        f"No encontré {self.secrets_file}.\n"
                        "Descarga el client_secret.json desde Google Cloud Console\n"
                        "y colócalo en la raíz del proyecto."
                    )
                logger.info("Autorizando con YouTube (abre browser)...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.secrets_file), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for future runs
            with open(self.token_file, "w") as f:
                f.write(creds.to_json())
            logger.info(f"Token guardado en {self.token_file}")

        self._service = build("youtube", "v3", credentials=creds)
        return self._service

    def upload_short(self, video_path: Path, paper,
                     privacy: str = "public") -> str | None:
        """
        Upload a Short to YouTube.

        Args:
            video_path: Path to the .mp4 file.
            paper:      Paper metadata object (title, abstract, url, authors).
            privacy:    "public" | "private" | "unlisted"

        Returns:
            YouTube video ID (e.g. "dQw4w9WgXcQ") or None if failed.
        """
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video no encontrado: {video_path}")
            return None

        title = f"🤖 {paper.title[:85]}..." if len(paper.title) > 85 else f"🤖 {paper.title}"
        # YouTube Shorts: añadir #Shorts al título/descripción es clave para el algoritmo
        title = title + " #Shorts"

        body = {
            "snippet": {
                "title":       title,
                "description": _build_description(paper),
                "tags":        _build_tags(paper),
                "categoryId":  CATEGORY_SCIENCE,
                "defaultLanguage": "en",
            },
            "status": {
                "privacyStatus":           privacy,
                "selfDeclaredMadeForKids": False,
                "madeForKids":             False,
            },
        }

        logger.info(f"Subiendo Short: {video_path.name} ({video_path.stat().st_size // 1_000_000} MB)")
        logger.info(f"Título: {title}")
        logger.info(f"Privacy: {privacy}")

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            chunksize=CHUNK_SIZE,
            resumable=True,
        )

        youtube = self._get_service()
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        # Upload with retry + progress
        response = None
        retry    = 0
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    logger.info(f"  Subiendo... {pct}%")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    retry += 1
                    if retry > MAX_RETRIES:
                        logger.error("Demasiados errores — abortando upload.")
                        return None
                    wait = 2 ** retry
                    logger.warning(f"Error {e.resp.status} — reintentando en {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP error {e.resp.status}: {e.content}")
                    return None

        video_id = response.get("id")
        if video_id:
            url = f"https://www.youtube.com/shorts/{video_id}"
            logger.info(f"✅ Short publicado: {url}")
            return video_id
        else:
            logger.error(f"Upload falló — respuesta: {response}")
            return None

    def verify_channel(self) -> bool:
        """Check that credentials work and show channel name."""
        try:
            youtube = self._get_service()
            resp = youtube.channels().list(part="snippet", mine=True).execute()
            items = resp.get("items", [])
            if items:
                name = items[0]["snippet"]["title"]
                logger.info(f"✅ Canal verificado: {name}")
                return True
            logger.warning("No se encontró canal en esta cuenta.")
            return False
        except Exception as e:
            logger.error(f"Error verificando canal: {e}")
            return False


# ── CLI para setup y test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    # Fix path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import setup_logging
    setup_logging("INFO")

    parser = argparse.ArgumentParser(description="YouTube uploader CLI")
    parser.add_argument("--auth",    action="store_true", help="Autorizar cuenta (abre browser)")
    parser.add_argument("--verify",  action="store_true", help="Verificar credenciales")
    parser.add_argument("--upload",  type=str,            help="Path al .mp4 a subir (test)")
    parser.add_argument("--privacy", default="private",   choices=["public", "private", "unlisted"])
    args = parser.parse_args()

    uploader = YouTubeUploader()

    if args.auth or args.verify:
        uploader.verify_channel()

    elif args.upload:
        # Test upload con paper falso
        from dataclasses import dataclass
        @dataclass
        class FakePaper:
            title    = "Test Short — AI Research Explained"
            abstract = "This is a test upload to verify the YouTube API integration."
            url      = "https://arxiv.org/abs/test"
            authors  = ["Test Author"]

        vid_id = uploader.upload_short(Path(args.upload), FakePaper(), privacy=args.privacy)
        if vid_id:
            print(f"\n✅ Video ID: {vid_id}")
            print(f"   URL: https://www.youtube.com/shorts/{vid_id}")
        else:
            print("\n❌ Upload falló")
            sys.exit(1)
    else:
        parser.print_help()
"""
image_generator.py — FLUX.1-schnell via mflux (Apple Silicon MLX).

API correcta para mflux >= 0.4:
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.common.config.model_config import ModelConfig

Primera ejecución descarga FLUX.1-schnell (~17GB) a ~/.cache/huggingface/
Ejecuciones posteriores usan caché — carga en ~30s.
Generación: ~60-120s por imagen 1080x1920 en Mac Air M1 16GB.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

W, H = 1080, 1920        # tamaño final (upscale con Pillow)
GEN_W, GEN_H = 576, 1024  # tamaño de generación — múltiplos de 16, cabe en 16GB
NUM_STEPS = 4             # FLUX schnell: 1-4 pasos óptimo

NEGATIVE_PROMPT = (
    "text, letters, words, watermark, signature, blurry, low quality, "
    "ugly, deformed, cartoon, anime"
)

# Paletas de color rotativas — cada paper usa una diferente
COLOR_PALETTES = [
    ("purple and blue",  "violet indigo",    "mint green and cyan",  "amber and gold",   "purple and cyan"),
    ("red and orange",   "crimson magenta",  "emerald and lime",     "blue and silver",  "orange and pink"),
    ("cyan and white",   "teal and amber",   "pink and purple",      "green and yellow", "blue and white"),
    ("gold and white",   "rose and coral",   "blue and turquoise",   "purple and mint",  "gold and orange"),
]

# Estilos visuales rotativas — varía la estética base
VISUAL_STYLES = [
    "biopunk organic neural tendrils",
    "crystalline geometric low-poly",
    "quantum particle field simulation",
    "deep space nebula cosmic",
    "holographic data stream matrix",
    "fractal mathematical infinite zoom",
]

BASE_QUALITY = "ultra detailed, cinematic lighting, 8k, no text, no letters, dark background"

def build_prompts(paper_title: str = "", paper_keywords: list = None,
                   palette_idx: int = 0, style_idx: int = 0) -> dict:
    """Build dynamic prompts unique to each paper."""
    kw = paper_keywords or []
    topic = " ".join(kw[:2]) if kw else "artificial intelligence"
    palette = COLOR_PALETTES[palette_idx % len(COLOR_PALETTES)]
    style   = VISUAL_STYLES[style_idx % len(VISUAL_STYLES)]
    c_title, c_prob, c_find, c_impact, c_cta = palette

    return {
        "title": (
            f"{style} visualization of {topic}, glowing {c_title} neon nodes "
            f"connected by light threads, floating geometric shapes, particles, {BASE_QUALITY}"
        ),
        "problem": (
            f"dark {style} labyrinth representing {topic} complexity, "
            f"tangled {c_prob} glowing circuits, chaotic energy field, {BASE_QUALITY}"
        ),
        "finding": (
            f"{style} crystal structures glowing {c_find}, "
            f"sacred geometry patterns emerging from {topic} data, {BASE_QUALITY}"
        ),
        "impact": (
            f"futuristic {style} cityscape with {c_impact} neural pathways "
            f"representing {topic} breakthroughs, aerial view, {BASE_QUALITY}"
        ),
        "cta": (
            f"abstract {style} portal of {c_cta} light rings, "
            f"{topic} data streams flowing inward, sense of discovery, {BASE_QUALITY}"
        ),
    }


class ImageGenerator:
    """
    FLUX.1-schnell wrapper. Singleton — modelo cargado una sola vez.
    """

    _flux = None

    def __init__(self):
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if ImageGenerator._flux is not None:
            return
        try:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config.model_config import ModelConfig

            logger.info("Cargando FLUX.1-schnell via mflux...")
            logger.info("(Primera vez descarga ~17GB a ~/.cache/huggingface/)")
            ImageGenerator._flux = Flux1(
                model_config=ModelConfig.schnell(),
                quantize=8,   # 8-bit quantization: ~8GB RAM, calidad excelente
            )
            logger.info("FLUX.1-schnell listo ✓")
        except Exception as e:
            logger.error(f"Error cargando FLUX: {e}", exc_info=True)
            raise

    def generate(self, prompt: str, output_path: Path, seed: int = 42) -> Path | None:
        """Genera una imagen y la guarda como PNG en 1080x1920."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Generando: {output_path.name}...")

            result = ImageGenerator._flux.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                height=GEN_H,    # generamos pequeño para no agotar GPU
                width=GEN_W,
                guidance=0.0,    # FLUX schnell usa guidance=0
            )

            # Obtener imagen PIL y upscale a 1080x1920
            from PIL import Image as PILImage
            if hasattr(result, "image"):
                pil_img = result.image
            else:
                pil_img = result
            pil_img = pil_img.resize((W, H), PILImage.LANCZOS)
            pil_img.save(str(output_path), "PNG")

            size_kb = output_path.stat().st_size // 1024
            logger.info(f"Guardado: {output_path.name} ({size_kb} KB)")
            return output_path

        except Exception as e:
            logger.error(f"Error generando imagen: {e}", exc_info=True)
            return None

    def generate_all_backgrounds(self, output_dir: Path,
                                   paper_id: str = "",
                                   paper_title: str = "",
                                   paper_keywords: list = None) -> dict[str, Path | None]:
        """
        Genera 5 fondos únicos por paper.
        - Seed derivado de arxiv_id → misma corrida produce mismas imágenes
        - Prompts dinámicos con keywords del paper → cada Short se ve diferente
        - Paleta y estilo visual rotan por paper → variedad visual garantizada
        - Caché por carpeta de paper → no regenera si ya existen
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        import hashlib
        # Seed único por paper
        base_seed = int(hashlib.md5(paper_id.encode()).hexdigest()[:8], 16) % 100000 if paper_id else 42
        # Paleta y estilo varían por paper
        palette_idx = base_seed % 4
        style_idx   = (base_seed // 4) % 6
        offsets = {"title": 0, "problem": 1, "finding": 2, "impact": 3, "cta": 4}

        # Extraer keywords del título si no se pasan
        if not paper_keywords and paper_title:
            import re
            stopwords = {"the","a","an","of","in","on","for","with","and","or",
                         "is","are","we","our","this","that","from","to","by",
                         "new","novel","large","model","paper","study","via","using"}
            words = re.findall(r"[a-zA-Z]{4,}", paper_title.lower())
            paper_keywords = [w for w in words if w not in stopwords][:4]

        prompts = build_prompts(
            paper_title=paper_title,
            paper_keywords=paper_keywords,
            palette_idx=palette_idx,
            style_idx=style_idx,
        )

        logger.info(f"  Style: {VISUAL_STYLES[style_idx % len(VISUAL_STYLES)]}")
        logger.info(f"  Palette: {COLOR_PALETTES[palette_idx % len(COLOR_PALETTES)][0]}")

        results = {}
        for slide_type, prompt in prompts.items():
            path = output_dir / f"bg_{slide_type}.png"
            seed = base_seed + offsets[slide_type]
            if path.exists():
                logger.info(f"  Caché: bg_{slide_type}.png")
                results[slide_type] = path
            else:
                results[slide_type] = self.generate(prompt, path, seed=seed)

        return results

    @staticmethod
    def cleanup() -> None:
        if ImageGenerator._flux is not None:
            try:
                del ImageGenerator._flux
                ImageGenerator._flux = None
                import gc
                gc.collect()
                logger.debug("FLUX liberado de memoria.")
            except Exception:
                pass
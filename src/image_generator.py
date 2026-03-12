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

PROMPTS = {
    "title": (
        "abstract neural network visualization, glowing purple and blue neon nodes "
        "connected by light threads, floating geometric shapes, particles, "
        "dark space background, ultra detailed, cinematic lighting, 8k, no text"
    ),
    "problem": (
        "dark labyrinth of tangled glowing circuits and wires, amber orange neon glow, "
        "complex interconnected nodes, chaotic energy field, dark background, "
        "ultra detailed, 8k, cinematic, no text"
    ),
    "finding": (
        "crystalline geometric structures glowing mint green and cyan, "
        "sacred geometry patterns, particles emerging from center point, "
        "dark deep space background, ultra detailed, 8k, no text"
    ),
    "impact": (
        "futuristic cityscape aerial view, glowing amber neural pathways "
        "overlaid on city grid, warm neon light, sense of discovery and connection, "
        "dark background, ultra detailed, 8k, cinematic, no text"
    ),
    "cta": (
        "abstract portal of light, concentric glowing rings in purple and cyan, "
        "particles streaming inward, deep space background, sense of invitation, "
        "ultra detailed, 8k, cinematic, no text"
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

    def generate_all_backgrounds(self, output_dir: Path) -> dict[str, Path | None]:
        """
        Genera un fondo por tipo de slide.
        Usa caché si ya existen — no regenera en runs siguientes.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = {"title": 42, "problem": 7, "finding": 13, "impact": 99, "cta": 55}
        results = {}

        for slide_type, prompt in PROMPTS.items():
            path = output_dir / f"bg_{slide_type}.png"
            if path.exists():
                logger.info(f"  Caché: bg_{slide_type}.png")
                results[slide_type] = path
            else:
                results[slide_type] = self.generate(prompt, path, seed=seeds[slide_type])

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
"""
slides_generator.py — High-quality 9:16 slides with AI-generated backgrounds.

Each slide = FLUX.1-schnell background + text overlay via Pillow.
Falls back to gradient-only if image generation is unavailable.

Install: pip install pillow diffusionkit
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

W, H = 1080, 1920

PALETTE = {
    "text":   "#f0f0ff",
    "white":  "#ffffff",
    "muted":  "#a0a0c0",
    "accent": "#c8b8ff",   # soft purple for overlay text
    "amber":  "#ffd080",
    "mint":   "#80ffcc",
}

ACCENT_COLORS = ["#c8b8ff", "#ffd080", "#c8b8ff", "#80ffcc", "#ffd080", "#80ffcc"]


def _hex(c: str):
    h = c.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    paths_bold    = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                     "/System/Library/Fonts/Helvetica.ttc",
                     "/Library/Fonts/Arial Bold.ttf"]
    paths_regular = ["/System/Library/Fonts/Supplemental/Arial.ttf",
                     "/System/Library/Fonts/Helvetica.ttc",
                     "/System/Library/Fonts/SFNS.ttf",
                     "/Library/Fonts/Arial.ttf"]
    for p in (paths_bold if bold else paths_regular):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


def _apply_dark_overlay(img, strength: float = 0.62):
    """Darken the AI image so text is clearly readable."""
    from PIL import Image, ImageEnhance
    darkened = ImageEnhance.Brightness(img).enhance(1.0 - strength)
    # Add a soft central vignette to keep edges very dark
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(vignette)
    # Dark edges
    for margin in range(0, 220, 8):
        alpha = int(180 * (1 - margin / 220))
        draw.rectangle([margin, margin, W - margin, H - margin],
                        outline=(0, 0, 0, alpha), width=8)
    # Central text area: slightly lighter
    draw.rectangle([60, H // 4, W - 60, 3 * H // 4],
                    fill=(0, 0, 0, 30))
    result = Image.alpha_composite(darkened.convert("RGBA"), vignette)
    return result.convert("RGB")


def _gradient_bg(draw, top: str, bottom: str):
    """Fallback gradient if no AI image available."""
    r1, g1, b1 = _hex(top)
    r2, g2, b2 = _hex(bottom)
    for y in range(H):
        t = y / H
        draw.line([(0, y), (W, y)], fill=(
            int(r1 + (r2 - r1) * t),
            int(g1 + (g2 - g1) * t),
            int(b1 + (b2 - b1) * t),
        ))


def _centered_text(draw, text: str, y: int, font, color: str,
                    max_w: int = W - 120, spacing: int = 26) -> int:
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] <= max_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        x = (W - (bb[2] - bb[0])) // 2
        draw.text((x, y), line, font=font, fill=_hex(color))
        y += (bb[3] - bb[1]) + spacing
    return y


def _text_shadow(draw, text: str, x: int, y: int, font, color: str, shadow_offset: int = 3):
    """Draw text with a drop shadow for readability on any background."""
    # Shadow
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font,
               fill=(0, 0, 0, 180))
    # Main text
    draw.text((x, y), text, font=font, fill=_hex(color))


def _pill(draw, text: str, y: int, bg: str, fg: str = "#ffffff",
           font_size: int = 34, pad_x: int = 52, pad_y: int = 18) -> int:
    f = _font(font_size, bold=True)
    bb = draw.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pw, ph = tw + pad_x * 2, th + pad_y * 2
    x = (W - pw) // 2
    # Pill shadow
    draw.rounded_rectangle([x + 3, y + 3, x + pw + 3, y + ph + 3],
                             radius=ph // 2, fill=(0, 0, 0, 120))
    draw.rounded_rectangle([x, y, x + pw, y + ph],
                             radius=ph // 2, fill=_hex(bg))
    draw.text((x + pad_x, y + pad_y), text, font=f, fill=_hex(fg))
    return y + ph


def _divider(draw, y: int, color: str, w: int = 110, h: int = 5):
    x = (W - w) // 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=2, fill=_hex(color))


def _dots(draw, current: int, total: int, y: int, color: str):
    size, gap = 14, 22
    total_w = total * size + (total - 1) * (gap - size)
    sx = (W - total_w) // 2
    for i in range(total):
        cx = sx + i * gap
        fill = _hex(color) if i == current else (80, 80, 110)
        draw.ellipse([cx, y, cx + size, y + size], fill=fill)


def _make_base(bg_path: Path | None, gradient_top: str, gradient_bottom: str):
    """Create base image: AI background or gradient fallback."""
    from PIL import Image, ImageDraw
    if bg_path and bg_path.exists():
        img = Image.open(str(bg_path)).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.LANCZOS)
        img = _apply_dark_overlay(img)
    else:
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)
        _gradient_bg(draw, gradient_top, gradient_bottom)
    return img


# ── Slide builders ────────────────────────────────────────────────────────────

def _title_slide(paper, bg_path: Path | None, path: Path) -> Path:
    from PIL import ImageDraw
    img = _make_base(bg_path, "#07071a", "#140d2e")
    draw = ImageDraw.Draw(img)

    y = 360
    y = _pill(draw, "  AI RESEARCH  ", y, "#7c6af7", font_size=36) + 70

    title = paper.title if len(paper.title) <= 65 else paper.title[:62] + "…"
    y = _centered_text(draw, title, y, _font(74, bold=True), PALETTE["white"],
                        max_w=W - 80, spacing=22) + 44

    _divider(draw, y, PALETTE["amber"], w=130)
    y += 56

    authors = ", ".join(paper.authors[:2]) + (" et al." if len(paper.authors) > 2 else "")
    y = _centered_text(draw, authors, y, _font(42), PALETTE["muted"]) + 12
    _centered_text(draw, paper.published, y, _font(38), PALETTE["muted"])

    _centered_text(draw, "arxiv.org  ·  AI Explained", H - 120, _font(32), PALETTE["muted"])

    img.save(str(path), "PNG")
    return path


def _content_slide(label: str, body: str, bg_path: Path | None,
                    path: Path, accent: str, index: int) -> Path:
    gradients = [("#07071a", "#0a0712"), ("#070d12", "#07120a"),
                  ("#120a07", "#0a0a14"), ("#07121a", "#0a0a14")]
    top, bot = gradients[index % len(gradients)]

    from PIL import ImageDraw
    img = _make_base(bg_path, top, bot)
    draw = ImageDraw.Draw(img)

    y = 330
    y = _pill(draw, f"  {label}  ", y, accent, font_size=32) + 72

    _divider(draw, y, accent, w=90)
    y += 60

    # Large readable body text with shadow
    body_font = _font(66)
    words = body.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        bb = draw.textbbox((0, 0), test, font=body_font)
        if bb[2] <= W - 100:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))

    for line in lines:
        bb = draw.textbbox((0, 0), line, font=body_font)
        x = (W - (bb[2] - bb[0])) // 2
        # Shadow
        draw.text((x + 3, y + 3), line, font=body_font, fill=(0, 0, 0, 160))
        # Text
        draw.text((x, y), line, font=body_font, fill=_hex(PALETTE["text"]))
        y += (bb[3] - bb[1]) + 28

    _dots(draw, index, 6, H - 140, accent)
    img.save(str(path), "PNG")
    return path


def _cta_slide(paper, bg_path: Path | None, path: Path) -> Path:
    from PIL import ImageDraw
    img = _make_base(bg_path, "#0d0a22", "#07071a")
    draw = ImageDraw.Draw(img)

    y = H // 2 - 400
    y = _centered_text(draw, "Read the full paper", y,
                         _font(90, bold=True), PALETTE["white"],
                         max_w=W - 80, spacing=24) + 56

    _divider(draw, y, PALETTE["mint"], w=130)
    y += 66

    short_url = paper.url.replace("http://arxiv.org/abs/", "arxiv.org/abs/")
    y = _centered_text(draw, short_url, y, _font(40), PALETTE["accent"]) + 64

    _centered_text(draw, "Follow for daily AI research breakdowns 🤖",
                    y, _font(48), PALETTE["muted"], spacing=24)

    img.save(str(path), "PNG")
    return path


# ── Public API ────────────────────────────────────────────────────────────────

def generate_slides(paper, slide_content: dict, output_dir: Path,
                     backgrounds: dict | None = None) -> list[dict]:
    """
    Generate 6 slide PNGs. Uses AI backgrounds if provided, else gradient fallback.

    Args:
        paper:        Paper metadata object.
        slide_content: Dict with problem, finding1, finding2, impact text.
        output_dir:   Directory to save PNGs.
        backgrounds:  Dict mapping slide_type → Path from ImageGenerator.
                      Pass None to use gradient fallback.

    Returns:
        List of {"path": Path, "duration": int} dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bg = backgrounds or {}

    configs = [
        ("00_title.png",    "title",   None,                              "title",    8),
        ("01_problem.png",  "content", slide_content.get("problem", ""), "problem",  12),
        ("02_finding1.png", "content", slide_content.get("finding1", ""),"finding",  12),
        ("03_finding2.png", "content", slide_content.get("finding2", ""),"finding",  12),
        ("04_impact.png",   "content", slide_content.get("impact", ""),  "impact",   10),
        ("05_cta.png",      "cta",     None,                              "cta",       6),
    ]
    labels  = ["", "THE PROBLEM", "KEY FINDING", "KEY FINDING", "WHY IT MATTERS", ""]

    slides = []
    for i, (fname, stype, content, bg_key, dur) in enumerate(configs):
        p = output_dir / fname
        bg_img = bg.get(bg_key)

        if stype == "title":
            _title_slide(paper, bg_img, p)
        elif stype == "cta":
            _cta_slide(paper, bg_img, p)
        else:
            _content_slide(labels[i], content or "…", bg_img, p, ACCENT_COLORS[i], i)

        slides.append({"path": p, "duration": dur})
        logger.info(f"  Slide {i+1}/6: {fname} {'(AI bg)' if bg_img else '(gradient)'}")

    return slides


def parse_script_to_slides(script: str, llm_engine) -> dict:
    """Extract 4 short phrases from the narration script for slide cards."""
    logger.info("Extracting slide content from script...")

    prompt = f"""Extract 4 key phrases from this narration script for slide cards.
Each phrase must be SHORT (10-18 words), punchy, and self-contained.

Script:
{script}

Output EXACTLY this format (nothing else):
PROBLEM: <one sentence, 10-18 words>
FINDING1: <first key finding, 10-18 words>
FINDING2: <second key finding, 10-18 words>
IMPACT: <why this matters, 10-18 words>"""

    result = llm_engine.generate_custom(
        system_prompt="Extract slide content. Output only the 4 labeled lines, nothing else.",
        user_message=prompt,
        max_tokens=250,
        temperature=0.3,
    )

    content = {"problem": "", "finding1": "", "finding2": "", "impact": ""}
    if result:
        for line in result.strip().split("\n"):
            line = line.strip()
            for key in ["PROBLEM", "FINDING1", "FINDING2", "IMPACT"]:
                if line.upper().startswith(f"{key}:"):
                    content[key.lower()] = line.split(":", 1)[1].strip()

    logger.info(f"Slide content: {content}")
    return content
"""Pure helpers for reference-image KV seeding.

Prompt shaping, pixel loading, per-entry option resolution and the DCAE-V
encode memory estimate; none of them needs CUDA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# DCAE-V reference encode memory (measured on a meta/dispatch trace)
# ---------------------------------------------------------------------------
DCAEV_ENCODER_PARAMS = 916_700_000
DCAEV_ENCODER_BF16_BYTES = 2 * DCAEV_ENCODER_PARAMS  # ~1.71 GiB
# Activation peak per pixel frame at the measured 704x1280 plane: ~3.0 GiB for
# 4 frames, ~12.1 GiB for 16 frames. Scaled by pixel area for other sizes.
DCAEV_ENCODE_BYTES_PER_PIXEL_FRAME = int(0.75 * 2**30)
DCAEV_ENCODE_REFERENCE_AREA = 704 * 1280
# The causal encoder processes at most one 40-frame chunk at a time.
DCAEV_ENCODE_CHUNK_PIXEL_FRAMES = 40


def reference_encode_peak_bytes(
    *, pixel_frames: int, height: int, width: int, weight_bytes: int = DCAEV_ENCODER_BF16_BYTES,
) -> int:
    """Estimated GPU peak of one DCAE-V encode (weights + activations, no cuDNN workspace)."""
    frames = min(max(1, int(pixel_frames)), DCAEV_ENCODE_CHUNK_PIXEL_FRAMES)
    scale = (int(height) * int(width)) / DCAEV_ENCODE_REFERENCE_AREA
    return int(weight_bytes) + int(DCAEV_ENCODE_BYTES_PER_PIXEL_FRAME * frames * scale)


def choose_reference_full_clip(
    requested: bool,
    *,
    latent_frames: int,
    t_ratio: int,
    height: int,
    width: int,
    free_bytes: Optional[int],
    weight_bytes: int = DCAEV_ENCODER_BF16_BYTES,
) -> Tuple[bool, int]:
    """Keep ``reference_encode_full_clip`` only when its estimated peak fits ``free_bytes``.

    Returns ``(use_full_clip, estimated_full_clip_bytes)``. ``free_bytes=None``
    (no CUDA) keeps the request as is.
    """
    estimate = reference_encode_peak_bytes(
        pixel_frames=int(t_ratio) * int(latent_frames), height=height, width=width,
        weight_bytes=weight_bytes,
    )
    if not requested:
        return False, estimate
    if free_bytes is None:
        return True, estimate
    return estimate <= int(free_bytes), estimate


# ---------------------------------------------------------------------------
# Per-entry reference options
# ---------------------------------------------------------------------------
class ReferenceSpec(NamedTuple):
    image: Optional[str]
    prompt: Optional[str]
    latent_frames: int


def resolve_reference_image_path(
    value: Any, json_dir: Optional[Path], *, label: Any = "",
) -> Optional[str]:
    """Resolve a prompt-file ``reference_image`` (relative to the JSON's directory).

    Returns ``None`` for an empty value and raises ``FileNotFoundError`` when
    the resolved file does not exist.
    """
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and json_dir is not None:
        path = (Path(json_dir) / path).resolve()
    if not path.is_file():
        prefix = "" if label is None or label == "" else f"[{label}] "
        raise FileNotFoundError(f"{prefix}reference_image does not exist: {path}")
    return str(path)


def resolve_reference_entry(
    entry: Optional[Mapping[str, Any]],
    json_dir: Optional[Path],
    cfg_defaults: Any,
    *,
    rollout_type: str = "seg_ar",
    label: Any = "",
) -> ReferenceSpec:
    """Effective reference image / prompt / latent frames for one prompt entry.

    Entry keys override ``cfg_defaults`` (``inference.reference_*``). A relative
    entry image is resolved against ``json_dir`` (pass ``None`` when the loader
    already resolved it). A dict / list prompt is serialized to JSON.
    ``reference_latent_frames`` is parsed only when an image is set (``None``
    means the default 4; numeric strings are accepted; it must be a positive
    integer). Seeding is rejected for ``rollout_type == "one_shot"``.
    """
    entry = entry or {}
    entry_image = entry.get("reference_image")
    if entry_image and json_dir is not None:
        entry_image = resolve_reference_image_path(entry_image, json_dir, label=label)
    image = entry_image or getattr(cfg_defaults, "reference_image", None)
    prompt = entry.get("reference_prompt") or getattr(cfg_defaults, "reference_prompt", None)
    if isinstance(prompt, (dict, list)):
        # A structured JSON prompt given inline in the prompt file.
        prompt = json.dumps(prompt, ensure_ascii=False)
    frames = 4
    if image:
        raw_frames = entry.get("reference_latent_frames")
        if raw_frames is None:
            raw_frames = getattr(cfg_defaults, "reference_latent_frames", None)
        if raw_frames is not None:
            try:
                parsed = float(str(raw_frames).strip())
            except ValueError:
                parsed = float("nan")
            if not parsed.is_integer():
                raise ValueError(
                    f"reference_latent_frames must be a positive integer, got {raw_frames!r}"
                )
            frames = int(parsed)
        if frames <= 0:
            raise ValueError(
                f"reference_latent_frames must be a positive integer, got {raw_frames!r}"
            )
        if rollout_type == "one_shot":
            raise ValueError("reference_image seeding requires rollout_type seg_ar or chunk_ar")
    return ReferenceSpec(image=image, prompt=prompt, latent_frames=frames)


def format_timestamp(seconds: float) -> str:
    """``MM:SS.mmm`` as used by the segment prompt schema."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    minutes, rem = divmod(total_ms, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def default_reference_prompt(
    first_prompt: str, seconds: float, caption: Optional[str] = None,
) -> str:
    """Build the reference segment's prompt in the same schema as segment 0.

    Keeps the global keys (Style, perspective_view, subject_list, ...), drops
    ``contained_actions`` / ``description_change`` and uses either ``caption``
    or segment 0's ``visual_description_long``. A ``caption`` that already is
    a JSON object is used verbatim. Non-JSON prompts fall back to the caption
    or the first prompt as plain text.
    """
    if caption:
        try:
            parsed = json.loads(caption)
            if isinstance(parsed, dict):
                return caption
        except Exception:
            pass
    try:
        obj = json.loads(first_prompt)
    except Exception:
        obj = None
    if not isinstance(obj, dict):
        return caption or first_prompt
    out = {k: v for k, v in obj.items() if k != "segments"}
    segs = obj.get("segments")
    seg0 = segs[0] if isinstance(segs, list) and segs and isinstance(segs[0], dict) else {}
    ref_seg = {}
    if "start_time" in seg0 or "end_time" in seg0:
        ref_seg["start_time"] = format_timestamp(0.0)
        ref_seg["end_time"] = format_timestamp(seconds)
    description = caption or seg0.get("visual_description_long") or seg0.get("description_change")
    if description:
        ref_seg["visual_description_long"] = description
    out["segments"] = [ref_seg]
    return json.dumps(out, ensure_ascii=False)


def load_reference_frames(
    image_path: str, *, height: int, width: int, frames: int,
) -> torch.Tensor:
    """``[1, 3, frames, height, width]`` float32 in ``[-1, 1]``: resize to cover, center crop, repeat.

    The EXIF orientation tag is applied first, so camera photos come out the
    way they are displayed.
    """
    import numpy as np
    from PIL import Image, ImageOps

    if frames <= 0:
        raise ValueError("reference frames must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("reference height/width must be positive")
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        src_w, src_h = image.size
        scale = max(width / src_w, height / src_h)
        new_w = max(width, int(round(src_w * scale)))
        new_h = max(height, int(round(src_h * scale)))
        if (new_w, new_h) != (src_w, src_h):
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        image = image.crop((left, top, left + width, top + height))
        pixels = np.asarray(image, dtype=np.uint8)
    frame = torch.from_numpy(pixels.copy()).to(torch.float32).permute(2, 0, 1)  # [3, H, W]
    frame = frame.mul_(1.0 / 127.5).sub_(1.0)
    return frame[None, :, None].expand(1, 3, frames, height, width).contiguous()

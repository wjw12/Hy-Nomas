"""Pure helpers for reference-image KV seeding (prompt shaping and pixel loading)."""

from __future__ import annotations

import json
from typing import Optional

import torch


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
    """``[1, 3, frames, height, width]`` float32 in ``[-1, 1]``: resize to cover, center crop, repeat."""
    import numpy as np
    from PIL import Image

    if frames <= 0:
        raise ValueError("reference frames must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("reference height/width must be positive")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
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

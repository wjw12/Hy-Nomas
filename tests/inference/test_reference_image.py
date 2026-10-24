"""Reference-image KV seeding: prompt shaping, pixel loading and config plumbing."""

import json
import sys

import pytest
import torch

from hymm.utils.reference_image import (
    default_reference_prompt,
    format_timestamp,
    load_reference_frames,
)

SEGMENT_PROMPT = json.dumps({
    "Style": "Cinematic fantasy",
    "perspective_view": "Third-person view",
    "subject_list": [{"subject_id": "fox", "display_name": "a small orange fox"}],
    "segments": [{
        "start_time": "00:00.000",
        "end_time": "00:05.000",
        "visual_description_long": "A small orange fox stands on a stone path.",
        "contained_actions": [
            {"start_time": "00:00.000", "end_time": "00:05.000", "action_description": "walks"}
        ],
    }],
})


def test_format_timestamp():
    assert format_timestamp(0) == "00:00.000"
    assert format_timestamp(16 / 24) == "00:00.667"
    assert format_timestamp(65.5) == "01:05.500"


def test_default_reference_prompt_keeps_globals_and_drops_actions():
    out = json.loads(default_reference_prompt(SEGMENT_PROMPT, 16 / 24))
    assert out["Style"] == "Cinematic fantasy"
    assert out["subject_list"][0]["subject_id"] == "fox"
    assert len(out["segments"]) == 1
    seg = out["segments"][0]
    assert "contained_actions" not in seg and "description_change" not in seg
    assert seg["visual_description_long"].startswith("A small orange fox")
    assert seg["start_time"] == "00:00.000" and seg["end_time"] == "00:00.667"


def test_default_reference_prompt_uses_plain_caption_inside_schema():
    out = json.loads(default_reference_prompt(SEGMENT_PROMPT, 1.0, caption="A red fox portrait"))
    assert out["segments"][0]["visual_description_long"] == "A red fox portrait"
    assert out["perspective_view"] == "Third-person view"


def test_default_reference_prompt_passes_json_caption_through():
    custom = json.dumps({"Style": "Anime", "segments": [{"visual_description_long": "x"}]})
    assert default_reference_prompt(SEGMENT_PROMPT, 1.0, caption=custom) == custom


def test_default_reference_prompt_non_json_first_prompt():
    assert default_reference_prompt("a plain prompt", 1.0) == "a plain prompt"
    assert default_reference_prompt("a plain prompt", 1.0, caption="cap") == "cap"


def test_load_reference_frames_resizes_to_cover_and_center_crops(tmp_path):
    from PIL import Image

    # 100x60 image: left third red, middle green, right third blue.
    image = Image.new("RGB", (100, 60))
    for x in range(100):
        colour = (255, 0, 0) if x < 33 else (0, 255, 0) if x < 66 else (0, 0, 255)
        for y in range(60):
            image.putpixel((x, y), colour)
    path = tmp_path / "ref.png"
    image.save(path)

    frames = load_reference_frames(str(path), height=64, width=64, frames=3)
    assert frames.shape == (1, 3, 3, 64, 64)
    assert frames.dtype == torch.float32
    assert frames.min() >= -1.0 and frames.max() <= 1.0
    # Every frame is the same still.
    assert torch.equal(frames[:, :, 0], frames[:, :, 2])
    # Center crop of the resized (107x64) image keeps the green middle band.
    centre = frames[0, :, 0, 32, 32]
    assert centre[1] > 0.9 and centre[0] < -0.9 and centre[2] < -0.9

    with pytest.raises(ValueError):
        load_reference_frames(str(path), height=64, width=64, frames=0)


def test_reference_keys_are_captured_as_meta_not_prompt_text():
    try:
        from hymm.inference import _BENCH_META_TOP_LEVEL_KEYS
    except ImportError as exc:  # CUDA attention kernels are not installed on CPU-only hosts
        pytest.skip(f"hymm.inference needs the CUDA environment: {exc}")

    for key in ("reference_image", "reference_prompt", "reference_latent_frames"):
        assert key in _BENCH_META_TOP_LEVEL_KEYS


def test_reference_config_defaults():
    from configs.config import parse_args

    sys.argv = ["test", "+experiment=single_gpu_bf16_16gb_inference"]
    cfg = parse_args()
    assert cfg.inference.reference_image is None
    assert cfg.inference.reference_prompt is None
    assert cfg.inference.reference_latent_frames == 4
    assert cfg.inference.reference_encode_full_clip is False
    assert cfg.inference.reference_vae_checkpoint is None
    # The reference keys must not leak into the text sent to the encoder.
    assert "reference_image" not in list(cfg.inference.keep_keywords)

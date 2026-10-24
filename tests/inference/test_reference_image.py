"""Reference-image KV seeding: prompt shaping, pixel loading and config plumbing."""

import ast
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

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


# ---------------------------------------------------------------------------
# DCAE-V encoder-only loading (B-M5)
# ---------------------------------------------------------------------------
def _small_dcaev_config(real):
    import dataclasses

    cfg = real()
    widths, depths = (8, 8, 16, 16, 16, 16, 16), (0, 1, 1, 1, 1, 1, 1)
    cfg.encoder = dataclasses.replace(cfg.encoder, width_list=widths, depth_list=depths)
    cfg.decoder = dataclasses.replace(cfg.decoder, width_list=widths, depth_list=depths)
    return cfg


@pytest.mark.parametrize("layout", ["model_state_dict", "ema"])
def test_dcaev_encoder_only_matches_full_load(tmp_path, monkeypatch, layout):
    import hymm.models.vae.autoencoder_dcae_v as dcaev

    real = dcaev.dc_ae_v_f32t4c32_config
    monkeypatch.setattr(dcaev, "dc_ae_v_f32t4c32_config", lambda: _small_dcaev_config(real))
    torch.manual_seed(0)
    state = {k: v.to(torch.bfloat16) for k, v in dcaev.DCAEV(dcaev.dc_ae_v_f32t4c32_config()).state_dict().items()}
    payload = {"model_state_dict": state} if layout == "model_state_dict" else {
        "model_state_dict": {k: torch.zeros_like(v) for k, v in state.items()},
        "ema_model_state_dict": {"ema": state},
    }
    path = tmp_path / "dcaev.pt"
    torch.save(payload, path)

    encoder = dcaev.AutoencoderDCAEV.from_checkpoint(str(path), dtype=torch.float32, encoder_only=True)
    full = dcaev.AutoencoderDCAEV.from_checkpoint(str(path), dtype=torch.float32)
    assert encoder.model.decoder is None
    assert not any(name.startswith("model.decoder.") for name, _ in encoder.named_parameters())
    assert not any(p.is_meta for p in encoder.parameters())
    assert encoder.dtype == torch.float32
    assert encoder.model.spatial_compression_ratio == full.model.spatial_compression_ratio == 32
    assert encoder.model.temporal_compression_ratio == full.model.temporal_compression_ratio == 4
    pixels = torch.rand(1, 3, 4, 64, 64) * 2 - 1
    latent = encoder.encode(pixels).latent_dist.mode()
    assert latent.shape == (1, 32, 1, 2, 2)
    full_encoder = dict(full.model.encoder.named_parameters())
    for name, param in encoder.model.encoder.named_parameters():
        assert torch.equal(param, full_encoder[name])
    # Same weights; CPU conv kernels may pick different blockings, so not bit-exact.
    assert torch.allclose(latent, full.encode(pixels).latent_dist.mode(), rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match="encoder_only"):
        encoder.decode(latent)


# ---------------------------------------------------------------------------
# Pipeline pieces, executed from causal_ar.py source without CUDA imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]


class FakeStagedCache(list):
    """Stands in for HostStagedKVCache inside the exec'd pipeline source."""

    def __init__(self, *args):
        super().__init__(*args)
        self.events = []

    def begin_forward(self, **kwargs):
        self.events.append(("begin", kwargs))

    def end_forward(self):
        self.events.append(("end",))


def _causal_ar_source():
    return (ROOT / "hymm/pipeline/causal_ar.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def causal_ar_ns():
    import math
    import os
    import time
    from contextlib import contextmanager
    from typing import Callable, List, Optional

    from loguru import logger
    from omegaconf import ListConfig

    from hymm.distributed.parallel_states import nccl_info
    from hymm.utils.inference_metrics import stage

    tree = ast.parse(_causal_ar_source())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(torch=torch, math=math, os=os, time=time, logger=logger,
                     Callable=Callable, List=List, Optional=Optional, ListConfig=ListConfig,
                     BasePipeline=nn.Module, nccl_info=nccl_info, stage=stage,
                     contextmanager=contextmanager, HostStagedKVCache=FakeStagedCache)
    exec(compile(tree, "causal_ar.py", "exec"), namespace)
    return namespace


def _process_node():
    tree = ast.parse(_causal_ar_source())
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "CPTCausalInferencePipeline")
    return next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
                and node.name == "_process_single_prompt")


class FakeTransformer:
    def __init__(self, out_channels=32):
        self.config = SimpleNamespace(out_channels=out_channels)
        self.calls = []
        self.sparse_optim_counts = {}

    def forward_with_cache(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        tokens = hidden.shape[2] * hidden.shape[3] * hidden.shape[4] + kwargs["text_states"].shape[1]
        return None, torch.tensor(tokens)


def _seed_pipeline(ns, *, sigma=0.0, cache_batch=1, in_channels=32, cache=None):
    cls = ns["CPTCausalInferencePipeline"]
    pipeline = cls.__new__(cls)
    nn.Module.__init__(pipeline)
    pipeline.cache_valid_len = 0
    pipeline.global_frame_counter = 0
    pipeline.segment_token_lengths = []
    pipeline.device = torch.device("cpu")
    pipeline.target_dtype = torch.float32
    pipeline.cache_noise_sigma = sigma
    pipeline.in_channels = in_channels
    pipeline.max_cache_segments = None
    pipeline.cache_at_step = -1
    pipeline.transformer = FakeTransformer()
    pipeline.kv_cache = cache if cache is not None else []
    pipeline.cache_mask_list = [torch.zeros(cache_batch, 64, dtype=torch.bool)]
    return pipeline


def _reference_inputs():
    torch.manual_seed(1)
    latent = torch.randn(1, 32, 3, 2, 2)
    enc = {"text_states": torch.randn(1, 5, 8), "text_mask": torch.ones(1, 5, dtype=torch.long)}
    return latent, enc


def _seed(pipeline, latent, enc, seed=0):
    return pipeline._seed_reference_history(
        reference_latent=latent, reference_enc=enc, batch_size=1, tokens_per_frame=4,
        generator=torch.Generator().manual_seed(seed),
    )


def test_seed_reference_history_writes_clean_reference_as_segment_zero(causal_ar_ns):
    pipeline = _seed_pipeline(causal_ar_ns)
    latent, enc = _reference_inputs()
    assert _seed(pipeline, latent, enc) == (0, 12)
    (call,) = pipeline.transformer.calls
    assert call["timestep"].dtype == torch.long and call["timestep"].tolist() == [0]
    assert torch.equal(call["hidden_states"], latent)
    assert call["cache_valid_len"] == 0 and call["global_frame_counter"] == 0
    assert call["update_cache"] is True and call["cache_txt"] is True
    assert call["kv_cache_list"] is pipeline.kv_cache
    assert call["cache_mask_list"] is pipeline.cache_mask_list
    assert pipeline.cache_valid_len == 17 and pipeline.segment_token_lengths == [17]
    assert pipeline.global_frame_counter == 3
    with pytest.raises(RuntimeError, match="empty KV history"):
        _seed(pipeline, latent, enc)


def test_seed_reference_history_renoises_and_pads_like_cache_write(causal_ar_ns):
    pipeline = _seed_pipeline(causal_ar_ns, sigma=0.25, in_channels=34)
    latent, enc = _reference_inputs()
    _seed(pipeline, latent, enc, seed=7)
    (call,) = pipeline.transformer.calls
    assert call["timestep"].dtype == torch.float32 and call["timestep"].tolist() == [250.0]
    hidden = call["hidden_states"]
    assert hidden.shape == (1, 34, 3, 2, 2)
    eps = torch.randn(latent.shape, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(hidden[:, :32], 0.75 * latent + 0.25 * eps)
    assert torch.count_nonzero(hidden[:, 32:]) == 0


def test_seed_reference_history_fills_every_guidance_slot(causal_ar_ns):
    # chunk_ar CFG / seg_ar without the single-slot cache: one KV slot per branch.
    pipeline = _seed_pipeline(causal_ar_ns, cache_batch=3)
    latent, enc = _reference_inputs()
    _seed(pipeline, latent, enc)
    (call,) = pipeline.transformer.calls
    assert call["hidden_states"].shape[0] == 3 and call["timestep"].shape == (3,)
    assert call["text_states"].shape[0] == 3 and call["text_mask"].shape[0] == 3
    for branch in range(3):
        assert torch.equal(call["hidden_states"][branch], latent[0])
        assert torch.equal(call["text_states"][branch], enc["text_states"][0])
    assert pipeline.cache_valid_len == 17  # tokens per slot, not per branch


def test_seed_reference_history_brackets_host_staged_forward(causal_ar_ns):
    cache = FakeStagedCache()
    pipeline = _seed_pipeline(causal_ar_ns, cache=cache)
    latent, enc = _reference_inputs()
    _seed(pipeline, latent, enc)
    assert cache.events == [
        ("begin", {"valid_len": 0, "extra_tokens": 3 * 4 + 5, "update": True}),
        ("end",),
    ]


def test_seed_reference_history_rejects_wrong_latent_channels(causal_ar_ns):
    pipeline = _seed_pipeline(causal_ar_ns)
    _, enc = _reference_inputs()
    with pytest.raises(RuntimeError, match="latent channels"):
        _seed(pipeline, torch.randn(1, 16, 3, 2, 2), enc)


def test_kv_forward_brackets_and_closes_on_exception(causal_ar_ns):
    pipeline = _seed_pipeline(causal_ar_ns, cache=FakeStagedCache())
    pipeline.cache_valid_len = 7
    with pipeline._kv_forward(update=False):
        pipeline.kv_cache.events.append(("body",))
    assert pipeline.kv_cache.events == [
        ("begin", {"valid_len": 7, "extra_tokens": 0, "update": False}), ("body",), ("end",),
    ]
    pipeline.kv_cache.events.clear()
    with pytest.raises(ValueError, match="boom"):
        with pipeline._kv_forward(update=True, extra_tokens=9):
            raise ValueError("boom")
    assert pipeline.kv_cache.events == [
        ("begin", {"valid_len": 7, "extra_tokens": 9, "update": True}), ("end",),
    ]
    plain = _seed_pipeline(causal_ar_ns)
    with plain._kv_forward(update=True, extra_tokens=3):
        pass  # ordinary list caches are untouched


def _base_text_encode_phase():
    from contextlib import contextmanager

    tree = ast.parse((ROOT / "hymm/pipeline/base.py").read_text(encoding="utf-8"))
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BasePipeline")
    method = next(node for node in base.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_text_encode_phase")
    namespace = {"contextmanager": contextmanager, "torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "base.py", "exec"), namespace)
    return namespace["_text_encode_phase"]


class CountingController:
    def __init__(self):
        self.depth = 0
        self.entered = 0

    def on_device(self):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            self.depth += 1
            self.entered += 1
            try:
                yield
            finally:
                self.depth -= 1

        return scope()


def test_text_encode_phase_is_released_when_encoding_raises(causal_ar_ns):
    process = _process_node()
    assert not any(isinstance(node, ast.Attribute) and node.attr in ("__enter__", "__exit__")
                   for node in ast.walk(process))
    with_node = next(node for node in ast.walk(process) if isinstance(node, ast.With)
                     and ast.unparse(node.items[0].context_expr) == "self._text_encode_phase()")
    function = ast.parse("def run(self, prompts, negative_prompt, needs_guidance):\n    pass").body[0]
    function.body = [with_node]
    namespace = dict(causal_ar_ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "causal_ar.py", "exec"), namespace)

    controller = CountingController()
    fake = SimpleNamespace(text_encoder=SimpleNamespace(_model_cpu_offload=controller))
    fake._text_encode_phase = MethodType(_base_text_encode_phase(), fake)
    seen_depth = []

    def failing_encode(prompt, **kwargs):
        seen_depth.append(controller.depth)
        raise RuntimeError("text encoder OOM")

    fake._encode_prompt = failing_encode
    with pytest.raises(RuntimeError, match="text encoder OOM"):
        namespace["run"](fake, ["a prompt"], "", False)
    assert seen_depth == [1] and controller.entered == 1
    assert controller.depth == 0


def test_fresh_rollout_resets_counters_before_noise_timeline(causal_ar_ns):
    """B-M1: _fixed_initial_noise_timeline reads global_frame_counter before KV init."""
    body = _process_node().body

    def index(predicate):
        return next(i for i, node in enumerate(body) if predicate(node))

    def calls(node, name):
        return any(isinstance(child, ast.Call) and isinstance(child.func, (ast.Attribute, ast.Name))
                   and getattr(child.func, "attr", getattr(child.func, "id", None)) == name
                   for child in ast.walk(node))

    assigned = index(lambda node: isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == "continuing_session")
    reset = index(lambda node: isinstance(node, ast.If)
                  and ast.unparse(node.test) == "not continuing_session")
    noise = index(lambda node: calls(node, "_fixed_initial_noise_timeline"))
    kv_init = index(lambda node: isinstance(node, ast.Expr) and calls(node, "_prepare_persistent_kv_cache"))
    assert assigned < reset < noise < kv_init

    function = ast.parse("def run(self, continuing_session):\n    pass").body[0]
    function.body = [body[reset]]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "causal_ar.py", "exec"), namespace)
    stale = SimpleNamespace(global_frame_counter=9, cache_valid_len=33)
    namespace["run"](stale, False)
    assert stale.global_frame_counter == 0 and stale.cache_valid_len == 0
    kept = SimpleNamespace(global_frame_counter=9, cache_valid_len=33)
    namespace["run"](kept, True)
    assert kept.global_frame_counter == 9 and kept.cache_valid_len == 33


def _entry_checks(causal_ar_ns):
    process = _process_node()
    stop = next(i for i, node in enumerate(process.body) if isinstance(node, ast.If)
                and ast.unparse(node.test).startswith("segment_frame_boundaries is not None"))
    function = ast.parse(
        "def run(self, preserve_kv_cache=False, reference_image=None, reference_latent_frames=4):\n"
        "    pass"
    ).body[0]
    function.body = process.body[1:stop] + [ast.Return(value=ast.Constant("ok"))]
    namespace = dict(causal_ar_ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "causal_ar.py", "exec"), namespace)
    return namespace["run"]


def _entry_pipeline(**inference):
    return SimpleNamespace(
        transformer=FakeTransformer(),
        args=SimpleNamespace(inference=SimpleNamespace(**inference)),
    )


def test_entry_checks_reject_unsupported_requests_before_any_work(causal_ar_ns, tmp_path):
    run = _entry_checks(causal_ar_ns)
    image = tmp_path / "ref.png"
    image.write_bytes(b"not decoded at entry")
    assert run(_entry_pipeline(), reference_image=str(image)) == "ok"
    with pytest.raises(ValueError, match="kv_cache_host_staging does not support preserve_kv_cache"):
        run(_entry_pipeline(kv_cache_host_staging=True), preserve_kv_cache=True)
    with pytest.raises(FileNotFoundError, match="reference_image does not exist"):
        run(_entry_pipeline(), reference_image=str(tmp_path / "missing.png"))
    with pytest.raises(ValueError, match="sliding_window"):
        run(_entry_pipeline(sliding_window={"enabled": True}), reference_image=str(image))
    with pytest.raises(ValueError, match="positive"):
        run(_entry_pipeline(), reference_image=str(image), reference_latent_frames=0)

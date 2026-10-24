"""Validation for the BF16 layer-streaming inference options.

These options let the full-precision (no INT8 / FP8, no sliding window)
pipeline run on GPUs with less memory than any single model: weights are
streamed layer by layer from pinned host memory and the KV history is staged
per layer from the host. Every switch here is output-identical to the
resident configuration; the constraints below only rule out combinations the
streaming code paths do not implement.

The second half of the module is the pure arithmetic of the host-staged KV
pre-flight check (``host_staged_kv_preflight``), kept here so it can be unit
tested without CUDA.
"""

from typing import Optional

from loguru import logger

# Rough GPU working set of the DCAE-V VAE kept fully resident during the DiT
# phase when the decoder is not PixelUMM (which streams). Used for a warning only.
_DCAEV_RESIDENT_DECODER_BYTES = int(2.4 * 10**9)


def _int_option(section, name: str, default: int) -> int:
    """Read an integer option; only a missing / null value takes the default.

    ``0`` must stay ``0`` so the ``>= 2`` slot guard can reject it.
    """
    value = getattr(section, name, None)
    return default if value is None else int(value)


def streaming_options(args) -> dict:
    parallelism = getattr(args, "parallelism", None)
    vae = getattr(args, "vae", None)
    inference = getattr(args, "inference", None)
    return {
        "dit_block_streaming": bool(getattr(parallelism, "dit_block_streaming", False)),
        "dit_streaming_slots": _int_option(parallelism, "dit_streaming_slots", 2),
        "dit_resident_blocks": _int_option(parallelism, "dit_resident_blocks", 0),
        "text_encoder_layer_streaming": bool(
            getattr(parallelism, "text_encoder_layer_streaming", False)
        ),
        "text_encoder_streaming_slots": _int_option(parallelism, "text_encoder_streaming_slots", 2),
        "vae_layer_streaming": bool(getattr(vae, "layer_streaming", False)),
        "vae_streaming_slots": _int_option(vae, "streaming_slots", 2),
        "kv_cache_host_staging": bool(getattr(inference, "kv_cache_host_staging", False)),
        "kv_cache_host_staging_slots": _int_option(inference, "kv_cache_host_staging_slots", 2),
    }


def any_streaming_enabled(args) -> bool:
    options = streaming_options(args)
    return any(
        options[name]
        for name in (
            "dit_block_streaming",
            "text_encoder_layer_streaming",
            "vae_layer_streaming",
            "kv_cache_host_staging",
        )
    )


def validate_streaming_config(args) -> None:
    options = streaming_options(args)
    parallelism = getattr(args, "parallelism", None)
    vae = getattr(args, "vae", None)
    inference = getattr(args, "inference", None)

    for name in ("dit_streaming_slots", "text_encoder_streaming_slots",
                 "vae_streaming_slots", "kv_cache_host_staging_slots"):
        if options[name] < 2:
            raise ValueError(f"{name} must be >= 2 (one unit computing, one in flight)")
    if options["dit_resident_blocks"] < 0:
        raise ValueError("parallelism.dit_resident_blocks must be >= 0")

    if options["dit_block_streaming"]:
        if getattr(parallelism, "dit_cpu_offload", False) or getattr(
            parallelism, "dit_cpu_offload_for_decode", False
        ):
            raise ValueError(
                "parallelism.dit_block_streaming replaces dit_cpu_offload / "
                "dit_cpu_offload_for_decode; disable the whole-model flags"
            )
        if getattr(inference, "dit_weight_quantization", "none") != "none":
            raise ValueError("dit_block_streaming streams the BF16 checkpoint; set dit_weight_quantization=none")
        if getattr(parallelism, "sp_size", 1) != 1:
            raise ValueError("dit_block_streaming is a single-GPU path (parallelism.sp_size=1)")
        if getattr(getattr(args, "dit", None), "master_weight_type", "bf16") != "bf16":
            raise ValueError("dit_block_streaming requires BF16 compute weights")
        if getattr(vae, "name", None) != "pixelumm":
            # Only the PixelUMM decoder streams; any other VAE stays fully
            # resident on the GPU through the DiT phase.
            logger.warning(
                "dit_block_streaming with model.vae.name="
                f"{getattr(vae, 'name', None)!r}: the VAE is not streamed and stays "
                f"resident on the GPU during the DiT phase (~{_DCAEV_RESIDENT_DECODER_BYTES / 1e9:.1f} GB "
                "for the DCAE-V decoder), on top of the streamed DiT working set"
            )

    if options["text_encoder_layer_streaming"]:
        if getattr(parallelism, "text_encoder_cpu_offload", False):
            raise ValueError(
                "parallelism.text_encoder_layer_streaming replaces text_encoder_cpu_offload"
            )
        if getattr(parallelism, "shard_text_encoder", False) and getattr(parallelism, "sp_size", 1) > 1:
            raise ValueError("text_encoder_layer_streaming cannot be combined with FSDP sharding")

    if options["vae_layer_streaming"]:
        if getattr(vae, "name", None) != "pixelumm":
            raise ValueError("model.vae.layer_streaming requires model.vae.name=pixelumm")
        if getattr(vae, "cpu_offload", False):
            raise ValueError("model.vae.layer_streaming replaces model.vae.cpu_offload")
        if getattr(vae, "compile_decoder", False):
            raise ValueError("model.vae.layer_streaming requires compile_decoder=false (fullgraph compile pins weight storage)")
        if getattr(vae, "cutlass_fp8", False) or getattr(vae, "deepgemm_fp8", False):
            raise ValueError("model.vae.layer_streaming streams the BF16 decoder; disable FP8 backends")
        if str(getattr(vae, "parallel_mode", "none")).lower() != "none":
            raise ValueError("model.vae.layer_streaming requires parallel_mode=none")

    if options["kv_cache_host_staging"]:
        if getattr(inference, "kv_cache_quantization", "none") != "none":
            raise ValueError("kv_cache_host_staging keeps exact BF16 history; set kv_cache_quantization=none")
        if getattr(inference, "rollout_type", "seg_ar") not in ("seg_ar", "chunk_ar"):
            raise ValueError("kv_cache_host_staging applies to seg_ar / chunk_ar rollouts")
        window = getattr(inference, "sliding_window", {})
        if window.get("enabled", False):
            raise ValueError(
                "kv_cache_host_staging keeps the full lifelong history; disable "
                "inference.sliding_window (or use the sliding window without host staging)"
            )
        # preserve_kv_cache is a per-call pipeline argument (interactive
        # sessions), not a config key; the pipeline rejects it at entry.
        for name in ("interactive_kv_cache_layerwise_cpu_offload", "seg_ar_segment_vae_decode"):
            if getattr(inference, name, False):
                raise ValueError(f"kv_cache_host_staging does not support inference.{name}")
        if (float(getattr(inference, "cfg_scale_video", 1.0)) != 1.0
                or float(getattr(inference, "camera_cfg_scale", 1.0)) != 1.0):
            # Host staging holds one physical KV slot; chunk_ar CFG and seg_ar
            # without the single-slot optimization keep one slot per branch.
            if getattr(inference, "rollout_type", "seg_ar") == "chunk_ar":
                raise ValueError(
                    "kv_cache_host_staging with CFG requires rollout_type=seg_ar "
                    "(chunk_ar keeps one KV slot per guidance branch)"
                )
            if not bool(getattr(inference, "ar_single_slot_cfg_cache", True)):
                raise ValueError(
                    "kv_cache_host_staging with CFG requires ar_single_slot_cfg_cache=true"
                )
        if getattr(inference, "sparse_attention", {}).get("enabled", False):
            raise ValueError("kv_cache_host_staging currently requires dense attention")
        if getattr(getattr(args, "dit", None), "master_weight_type", "bf16") != "bf16":
            raise ValueError("kv_cache_host_staging requires BF16 compute")


# ---------------------------------------------------------------------------
# Host-staged KV pre-flight (pure arithmetic)
# ---------------------------------------------------------------------------
# Conservative DiT activation peak per token of the longest chunk (img + text)
# and per guidance-batch row, used when no measured profile is available.
# 48 KiB/token is ~12 hidden-size BF16 vectors (hidden=2048): qkv, the MLP's
# 4x expansion and the attention output alive at once, with margin.
KV_PREFLIGHT_ACTIVATION_BYTES_PER_TOKEN = 48 * 1024


def kv_layer_token_bytes(cache_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """Bytes of K + V for one token of one layer (8192 at 16 heads x 128, BF16)."""
    return 2 * int(cache_heads) * int(head_dim) * int(dtype_bytes)


def estimate_host_staged_kv_gpu_bytes(
    *,
    history_tokens: int,
    cache_heads: int,
    head_dim: int,
    slots: int,
    guidance_batch: int,
    extra_tokens: int = 0,
    cache_batch: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """GPU KV working set of a host-staged cache at ``history_tokens`` of history.

    ``slots`` staging slots each hold one layer's whole history (plus the
    current chunk's ``extra_tokens`` on cache writes), and the DiT
    materializes the broadcast history once per guidance-batch row
    (``history_k.expand(B, ...)`` + ``torch.cat`` with the current chunk)::

        bytes = 2 * heads * head_dim * dtype_bytes
                * (slots * cache_batch + guidance_batch) * (history_tokens + extra_tokens)
    """
    per_token = kv_layer_token_bytes(cache_heads, head_dim, dtype_bytes)
    rows = int(slots) * int(cache_batch) + int(guidance_batch)
    return per_token * rows * (int(history_tokens) + int(extra_tokens))


def estimate_host_staged_kv_host_bytes(
    *,
    capacity_tokens: int,
    num_layers: int,
    cache_heads: int,
    head_dim: int,
    cache_batch: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Pinned host bytes of the full history: ``layers * 2 * heads * head_dim * 2 B * tokens``."""
    per_token = kv_layer_token_bytes(cache_heads, head_dim, dtype_bytes)
    return int(num_layers) * per_token * int(cache_batch) * int(capacity_tokens)


def max_history_tokens_for(
    *,
    budget_bytes: int,
    cache_heads: int,
    head_dim: int,
    slots: int,
    guidance_batch: int,
    extra_tokens: int = 0,
    cache_batch: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Largest history (tokens) whose KV working set fits ``budget_bytes`` (inverse of the estimate)."""
    per_token = kv_layer_token_bytes(cache_heads, head_dim, dtype_bytes)
    rows = int(slots) * int(cache_batch) + int(guidance_batch)
    if budget_bytes <= 0:
        return 0
    return max(0, int(budget_bytes) // (per_token * rows) - int(extra_tokens))


def host_staged_kv_preflight(
    *,
    history_tokens: int,
    extra_tokens: int,
    activation_tokens: int,
    history_latent_frames: int,
    cache_heads: int,
    head_dim: int,
    slots: int,
    guidance_batch: int,
    num_layers: int,
    capacity_tokens: int,
    resident_bytes: int = 0,
    free_bytes: Optional[int] = None,
    fps: float = 24.0,
    pixel_frames_per_latent: int = 4,
    cache_batch: int = 1,
    dtype_bytes: int = 2,
    activation_bytes_per_token: int = KV_PREFLIGHT_ACTIVATION_BYTES_PER_TOKEN,
    enforce: bool = True,
) -> dict:
    """Predict the DiT-phase GPU peak of a host-staged request; raise if it cannot fit.

    ``predicted = resident_bytes + kv_gpu_bytes + activation_bytes`` where
    ``kv_gpu_bytes`` is :func:`estimate_host_staged_kv_gpu_bytes` at the
    request's maximum history and ``activation_bytes = activation_bytes_per_token
    * activation_tokens * guidance_batch``. With ``free_bytes`` given and
    ``enforce`` set, a ``ValueError`` names the estimate, the free bytes and
    the longest history (latent frames / seconds at ``fps``) that would fit.
    Returns the estimate as a dict (for logging and the audit).
    """
    kv_gpu = estimate_host_staged_kv_gpu_bytes(
        history_tokens=history_tokens, extra_tokens=extra_tokens, cache_heads=cache_heads,
        head_dim=head_dim, slots=slots, guidance_batch=guidance_batch, cache_batch=cache_batch,
        dtype_bytes=dtype_bytes,
    )
    activation = int(activation_bytes_per_token) * int(activation_tokens) * int(guidance_batch)
    predicted = int(resident_bytes) + kv_gpu + activation
    host = estimate_host_staged_kv_host_bytes(
        capacity_tokens=capacity_tokens, num_layers=num_layers, cache_heads=cache_heads,
        head_dim=head_dim, cache_batch=cache_batch, dtype_bytes=dtype_bytes,
    )
    # Tokens per latent frame of history, text amortized over the frames.
    tokens_per_frame = max(1.0, float(history_tokens) / max(1, int(history_latent_frames)))
    result = {
        "history_tokens": int(history_tokens),
        "kv_gpu_bytes": kv_gpu,
        "activation_bytes": activation,
        "resident_bytes": int(resident_bytes),
        "predicted_bytes": predicted,
        "host_kv_bytes": host,
        "free_bytes": None if free_bytes is None else int(free_bytes),
        "fits": None,
        "max_history_tokens": None,
        "max_latent_frames": None,
        "max_seconds": None,
    }
    if free_bytes is None:
        return result
    budget = int(free_bytes) - int(resident_bytes) - activation
    max_tokens = max_history_tokens_for(
        budget_bytes=budget, cache_heads=cache_heads, head_dim=head_dim, slots=slots,
        guidance_batch=guidance_batch, extra_tokens=extra_tokens, cache_batch=cache_batch,
        dtype_bytes=dtype_bytes,
    )
    max_frames = int(max_tokens // tokens_per_frame)
    max_seconds = max_frames * int(pixel_frames_per_latent) / float(fps or 24.0)
    result.update(
        fits=predicted <= int(free_bytes),
        max_history_tokens=max_tokens,
        max_latent_frames=max_frames,
        max_seconds=max_seconds,
    )
    if enforce and not result["fits"]:
        gib = 2**30
        raise ValueError(
            "kv_cache_host_staging pre-flight: the DiT phase is predicted to need "
            f"{predicted / gib:.2f} GiB of GPU memory (KV staging + broadcast history "
            f"{kv_gpu / gib:.2f} GiB for {int(history_tokens)} history tokens with "
            f"{int(slots)} slots and guidance batch {int(guidance_batch)}, activations "
            f"~{activation / gib:.2f} GiB, resident DiT weights {int(resident_bytes) / gib:.2f} GiB) "
            f"but only {int(free_bytes) / gib:.2f} GiB is free. At most {max_frames} latent "
            f"frames (~{max_seconds:.1f} s at {float(fps or 24.0):g} fps) of history fit. "
            "Shorten the request, set inference.max_cache_segments, use fewer staging "
            "slots / no CFG, or set inference.kv_cache_host_staging_preflight=false to "
            "skip this check."
        )
    return result

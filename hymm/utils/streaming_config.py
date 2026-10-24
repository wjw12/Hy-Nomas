"""Validation for the BF16 layer-streaming inference options.

These options let the full-precision (no INT8 / FP8, no sliding window)
pipeline run on GPUs with less memory than any single model: weights are
streamed layer by layer from pinned host memory and the KV history is staged
per layer from the host. Every switch here is output-identical to the
resident configuration; the constraints below only rule out combinations the
streaming code paths do not implement.
"""


def streaming_options(args) -> dict:
    parallelism = getattr(args, "parallelism", None)
    vae = getattr(args, "vae", None)
    inference = getattr(args, "inference", None)
    return {
        "dit_block_streaming": bool(getattr(parallelism, "dit_block_streaming", False)),
        "dit_streaming_slots": int(getattr(parallelism, "dit_streaming_slots", 2) or 2),
        "dit_resident_blocks": int(getattr(parallelism, "dit_resident_blocks", 0) or 0),
        "text_encoder_layer_streaming": bool(
            getattr(parallelism, "text_encoder_layer_streaming", False)
        ),
        "text_encoder_streaming_slots": int(
            getattr(parallelism, "text_encoder_streaming_slots", 2) or 2
        ),
        "vae_layer_streaming": bool(getattr(vae, "layer_streaming", False)),
        "vae_streaming_slots": int(getattr(vae, "streaming_slots", 2) or 2),
        "kv_cache_host_staging": bool(getattr(inference, "kv_cache_host_staging", False)),
        "kv_cache_host_staging_slots": int(
            getattr(inference, "kv_cache_host_staging_slots", 2) or 2
        ),
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
        for name in ("interactive_kv_cache_layerwise_cpu_offload", "preserve_kv_cache",
                     "seg_ar_segment_vae_decode"):
            if getattr(inference, name, False):
                raise ValueError(f"kv_cache_host_staging does not support inference.{name}")
        if (float(getattr(inference, "cfg_scale_video", 1.0)) != 1.0
                or float(getattr(inference, "camera_cfg_scale", 1.0)) != 1.0):
            if not bool(getattr(inference, "ar_single_slot_cfg_cache", True)):
                raise ValueError(
                    "kv_cache_host_staging with CFG requires ar_single_slot_cfg_cache=true"
                )
        if getattr(inference, "sparse_attention", {}).get("enabled", False):
            raise ValueError("kv_cache_host_staging currently requires dense attention")
        if getattr(getattr(args, "dit", None), "master_weight_type", "bf16") != "bf16":
            raise ValueError("kv_cache_host_staging requires BF16 compute")

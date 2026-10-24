"""Names and enabled-state reporting for inference CPU residency."""



OFFLOAD_OPTIONS = (
    "text_encoder_cpu_offload", "dit_cpu_offload", "dit_cpu_offload_for_decode",
)


STREAMING_OPTIONS = (
    "dit_block_streaming", "text_encoder_layer_streaming",
)

# Names reported by enabled_cpu_offloads for layer streaming. Unlike the
# whole-model flags they are valid under data-parallel prompt sharding.
STREAMING_OFFLOAD_NAMES = STREAMING_OPTIONS + ("vae_layer_streaming",)


def enabled_cpu_offloads(args):
    parallelism = getattr(args, "parallelism", None)
    enabled = [name for name in OFFLOAD_OPTIONS if getattr(parallelism, name, False)]
    if getattr(getattr(args, "vae", None), "cpu_offload", False):
        enabled.append("vae_cpu_offload")
    # Layer streaming is CPU residency at finer granularity; it shares the
    # CUDA-device check of the whole-model offload flags (not the SP topology
    # check, see validate_model_cpu_offload).
    enabled += [name for name in STREAMING_OPTIONS if getattr(parallelism, name, False)]
    if getattr(getattr(args, "vae", None), "layer_streaming", False):
        enabled.append("vae_layer_streaming")
    return enabled

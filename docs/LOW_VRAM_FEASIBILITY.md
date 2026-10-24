# Hy-Nomas on a GPU with less than 24 GB, BF16, no quantization

Static code analysis of `Tencent-Hunyuan/Hy-Nomas` (clone at this directory, 2026-10-23) plus
safetensors headers read from the Hugging Face repo. Nothing was executed on a GPU. Four
independent analyses (DiT internals, pipeline scheduling, decoder + text encoder, external
offload techniques) were cross-checked; where they disagreed, the number derived from the
checkpoint header is used.

> **Status (2026-10-24):** the changes listed in §6 are implemented (commits "Add BF16 layer
> streaming...", "Add reference-image KV-cache seeding...", and two review-iteration commits)
> and reviewed twice statically. How to test them on a real machine:
> [`BF16_16GB_TESTING.md`](BF16_16GB_TESTING.md). One correction to this report came out of
> the review: the implemented per-layer KV staging holds one layer's *whole* history per GPU
> slot and the DiT attention concatenates it with the current tokens, so the GPU KV working set
> is `(slots + B) x 8 KiB x history_tokens` (about 22 MB per 720p latent frame at B = 1), not a
> constant. 16 GB therefore holds roughly 60 to 75 s of 720p history; a pre-flight check refuses
> longer requests. Tiled history attention (§6 of the testing doc) would remove that bound.

## Verdict

| Question | Answer |
|---|---|
| Does any shipped config run BF16 on < 24 GB? | **No.** The only low-memory preset is INT8 DiT + INT8 KV + FP8 decoder. |
| Does BF16 with the shipped offload flags fit even 24 GB? | **No.** DiT weights alone are 17.1 GB; KV cache and activations push the denoise phase past 24 GB. |
| Does the architecture permit it with offload/scheduling only? | **Yes.** All three large models are plain sequential stacks with no cross-layer weight sharing, phases are strictly sequential, and the KV cache is re-materialized per layer anyway. |
| What GPU floor is achievable, output-identical? | **About 4 to 5 GB per phase** with per-block weight streaming and host-resident KV. 12 GB and 16 GB cards both work. |
| What is the cost? | New code (roughly 300 to 600 lines across 5 files), **64 GB host RAM minimum, 96 to 128 GB recommended**, Linux, and PCIe traffic that stays hidden behind compute at 720p. Throughput is bound by the card's TFLOPS, not by PCIe. |

## 1. What has to fit, in BF16

Derived from safetensors headers (FP32 checkpoints, cast to BF16 at load).

| Component | Params | BF16 | Structure | Streaming unit |
|---|---|---|---|---|
| DiT double blocks | 8.157 B | 16.3 GB | 54 identical MMDoubleStreamBlocks, hidden 2048, 16 heads x 128, MLP x4 | 151 M params = 302 MB per block (+ 9 MB camera ActionModule) |
| DiT non-block (txt_in refiner, time_in, final_layer, img_in, camera embeds) | 0.40 B | 0.8 GB | keep resident | |
| **DiT total** | **8.553 B** | **17.1 GB** | | |
| PixelUMM decoder | 6.972 B | 13.9 GB | 36 Qwen3-8B-shaped layers (hidden 4096, GQA 32/8, MLP 12288), 1-step x0-predictor | 193 M params = 386 MB per layer |
| PixelUMM encoder (in the same file) | 0.917 B | skipped at load | | |
| Qwen2.5-VL-7B text encoder, language model only | 7.07 B | 14.1 GB | 28 layers x 466 MB + 1.09 GB embedding | 466 MB per layer |
| Lifelong KV cache | | 442,368 B per token (54 layers x K,V x 16 x 128 x 2 B) | 880 image tokens per latent frame at 720x1280 (22 x 40) + 1280 padded text tokens per segment | 389 MB per latent frame, 566 MB per segment of text |

KV growth: 24 fps / 4x temporal compression = 6 latent frames per second = **about 2.3 GB of KV per second of video**, forever, unless a sliding window is enabled.

## 2. How the shipped pipeline schedules memory

Phases are already mutually exclusive, which is the good news.

1. **Load** (`hymm/pipeline/base.py:78-198`). With `dit_cpu_offload`, `text_encoder_cpu_offload`, `vae.cpu_offload` all models are built on CPU.
2. **KV cache pre-allocated on GPU at full video size** (`hymm/pipeline/causal_ar.py:596-620`, size formula `:1816-1828`), *before* text encoding. For a lossless lifelong cache this alone is the first thing that overflows.
3. **Text encode, all segments up front** (`causal_ar.py:2034-2072`). The whole 14.1 GB encoder is moved H2D and back **per prompt** (`encoder.py:392` + `model_cpu_offload.py:65-90`).
4. **DiT residency for the whole rollout** (`causal_ar.py:2097-3312` via `base.py:200-222`). Whole 17.1 GB model moved once. Per segment: 8 Euler steps + 1 cache-write forward (`:2331-2378`, `:2608-2628`). CFG = 1 by default, so one forward at batch 1, no doubled activations.
5. **Decode once at the end** (`causal_ar.py:3579` -> `pixelumm_decoder/autoencoder.py:525`), DiT already back on CPU. Whole 13.9 GB decoder moved H2D. Internal temporal windows of 10 latents with overlap 5.

The existing `ModelCPUOffload` (`hymm/utils/model_cpu_offload.py`) is **whole-model granularity**, synchronous (`non_blocking=False`), from pageable memory, and refuses mixed-device models (`:30-31`). It cannot reduce the peak below "largest single model + its working set".

Peak per phase, BF16, shipped flags, 3 segments x 30 latent frames at 720p:

| Phase | Resident | Full history | Sliding window 18 frames |
|---|---|---|---|
| Text encode | TE 14.1 + pre-allocated KV | ~60 GB | ~22 GB |
| Denoise / cache write | DiT 17.1 + KV + ~2.5 GB activations | ~64 GB | **~26 GB** |
| Decode | PixelUMM 13.9 + ~3.9 GB activations | ~18 GB | ~18 GB |

Nothing fits 24 GB, let alone 16 or 12.

## 3. Why streaming is architecturally possible

- **DiT block loop is plain sequential** (`hymm/models/dit/nomas_cpt.py:3462-3516`): `for index, block in enumerate(self.double_blocks): img, txt = block(...); self.action_modules[index](...)`. Inter-block state is activations plus `kv_cache_list[index]` only. No weight tying across blocks (`:1772-1788`). Blocks are invoked through `__call__`, deliberately so module hooks fire (comments at `:1079-1082`, `:3455-3457`, written for FSDP2).
- **torch.compile is streaming-friendly.** Only free functions `pre_attention` / `post_attention` are compiled, `fullgraph=True`, `"triton.cudagraphs": False`, and weights are passed as arguments (`hymm/models/dit/modules/nomas_compile_prepost.py:102-109`; call sites `nomas_cpt.py:1351-1363`, `1423-1433`). Swapping which GPU buffer a weight lives in is just a different input tensor. No CUDA graphs, no recompiles expected.
- **KV is already re-materialized per layer on the denoise path.** With `update_cache=False`, every layer builds `full_k = torch.cat([history_k, cur_k])` (`nomas_cpt.py:626-638`). Filling that buffer from pinned host memory instead of from a GPU cache adds nothing beyond one or two per-layer staging buffers. The cache-write pass (`:587-595`, `:615-624`) becomes an async D2H.
- **PixelUMM decoder is a plain `for layer in self.layers`** (`pixelumm_decoder/model.py:344-345`) with no cross-layer state. FlexAttention there is compiled independently of weight placement. Keep `compile_decoder=false` (default), since that one wraps the whole forward in `fullgraph=True`.
- **Text encoder is encoded entirely before the DiT phase opens** (`causal_ar.py:2034-2040` vs `:2095-2097`), so it can be evicted completely, or its outputs cached to disk (9.2 MB per prompt).
- **Precedent in the repo:** `apply_fsdp2_text_encoder` already does per-layer `fully_shard` with `CPUOffloadPolicy(pin_memory=True)`, `reshard_after_forward=True` (`hymm/distributed/fsdp_util.py:701-732`), which on a size-1 mesh is per-layer weight streaming. It is gated to `sp_size > 1` (`base.py:231-237`).

## 4. Bandwidth: does streaming hide behind compute?

Per DiT forward at a 5 s 720p segment (about 28k image + 1.3k text query tokens):

| Quantity | Value |
|---|---|
| Weight bytes streamed per forward | 16.3 GB (54 x 302 MB) |
| History KV streamed per forward | 442 KB x history tokens (7.3 GB at an 18-frame window) |
| PCIe Gen4 x16 pinned, measured | ~25 GB/s -> 0.7 s weights, ~0.3 s KV |
| PCIe Gen3 x16 or Gen4 x8 (4060 Ti, 5060 Ti) | ~12 GB/s -> 1.4 s weights |
| Compute per forward | ~0.5 to 0.7 PFLOP -> 9 to 12 s on a 4080, 15 to 20 s on a 4070 / 5070 at 60 % utilization |
| Per block | ~12 to 18 ms transfer vs ~170 to 250 ms compute |

Transfer is 10 to 30x smaller than compute, so a one-block-ahead prefetch on a side stream from **pinned** memory hides it. The crossover to bandwidth-bound is around 3 to 5k query tokens, i.e. segments shorter than about 1 s at 720p. Expected overhead versus infinite VRAM on the same card: roughly 5 to 15 %. Diffusers measured +6.9 % for Wan 14B with block-level group offload on an H100, and consumer cards have proportionally more slack because they compute slower while PCIe is the same.

Expected wall time per 5 s 720p segment (9 forwards): ~85 to 110 s on a 4080, ~140 to 180 s on a 4070 / 5070, 200 s+ on a 4060 Ti. Compute-dominated. These are first-order estimates, not measurements.

Not viable instead of explicit streaming: CUDA managed memory (PyTorch exposes it only via pluggable allocators and documents it as not recommended, 10 to 100x cliffs under oversubscription), Windows WDDM shared-memory fallback (silent, ~3x slower), `accelerate` sequential `cpu_offload` (synchronous, unpinned, ~10x slowdown on FLUX).

## 5. Target budgets

Assumes CUDA context + allocator slack ~0.5 to 1 GB, activations 1.25 to 2.5 GB depending on segment length (eager path; compiled pre/post lowers it; chunking the MLP over tokens would lower it further and is exact).

| Scenario | DiT phase GPU | 16 GB | 12 GB | Output |
|---|---|---|---|---|
| Shipped code, BF16 | >= 17.1 + KV + act | no | no | |
| Stream 1 to 2 blocks, full lossless KV kept on GPU | 0.9 + 0.5 + act + 442 KB x all tokens | ~13 GB for KV: ~3 segments of 10 latent frames (~5 s total video) | ~9 GB: ~2 such segments | identical |
| Stream blocks, `recent` sliding window 18 frames on GPU | ~7.3 GB KV + ~3 GB | fits (~10 GB) | tight (~10 GB) | **changed** (context truncation, not quantization) |
| Stream blocks, KV in pinned host, per-layer double buffer | 0.9 + 0.5 + act + ~2 GB staging | **~4.7 GB** | **~4.7 GB** (spare ~7 GB can hold ~20 blocks resident to cut PCIe traffic) | identical, video length bound by host RAM |
| PixelUMM decode, whole model resident (shipped `cpu_offload`) | 13.9 + 3.9 | **no** | no | |
| PixelUMM decode, per-layer streamed | 0.05 + 0.77 + 3.9 | **~4.7 GB** | **~4.7 GB** | identical |
| Text encoder, whole model (shipped) | 14.1 + 0.6 | borderline, only if nothing else resident | no | |
| Text encoder, per-layer streamed, or embeddings cached to disk | 1.5 to 2.6 GB, or 0 | yes | yes | identical |

Host RAM for the lossless lifelong configuration: 17.1 + 13.9 + 14.1 = 45 GB of pinned weights (or load each model only during its phase: 17.1 GB peak), plus 2.3 GB of KV per second of generated video, plus transient load spikes (the DiT loader materializes the whole FP32 checkpoint dict, ~34 GB, `hymm/models/checkpointing/load.py:46`; the text encoder most likely loads FP32 first, ~31 GB). A 60 s 720p video has ~140 GB of KV. On a host with 128 GB or less that, not VRAM, is the ceiling of a lossless lifelong cache; with more host RAM the GPU KV working set (see the status note at the top) becomes the bound at roughly 60 to 75 s on a 16 GB card.

## 6. Code changes required (none exist today)

1. **Per-block DiT weight streaming.** Hook `double_blocks[i]` + `action_modules[i]` as one unit in `forward_with_cache` (`nomas_cpt.py:3462-3516`): pinned host master copy, 2 to 3 fixed GPU slot buffers (all 54 blocks share shapes), side-stream `copy_(non_blocking=True)` prefetch of block i+1, CUDA events, swap `param.data` to slot views. Keep `img_in`, `txt_in`, `time_in`, `final_layer`, camera embeds resident (0.8 GB). Install in `base.py:145-146` behind a new flag; rework `_dit_denoise_residency` (`base.py:200-222`) to allocate slots on entry and free on exit. Do not install `ModelCPUOffload` on the DiT (its single-device check fails). Load on CPU and pin (`base.py:88-89, 122`). Quick prototype: `diffusers.hooks.apply_group_offloading(transformer, offload_type="block_level", num_blocks_per_group=1, use_stream=True, record_stream=True)` applied to `double_blocks`, but note the pipeline calls `forward_with_cache` directly (`causal_ar.py:2357, 2608`) so a top-level hook never fires; non-block modules must be placed manually.
2. **KV cache on pinned host, per-layer staging.** Move `_initialize_kv_cache` (`causal_ar.py:1906-1912`) after the encode loop (`:2072`); allocate pinned instead of GPU (`:596-620`), or generalize `PhaseSwappedKVCache` (`:58-301`) from whole-cache phase swap to per-layer prefetch. In `causal_attention_with_cache_mp` relax the same-device assertion (`nomas_cpt.py:446-451`), build `full_k`/`full_v` in a prefetched staging buffer instead of the `torch.cat` at `:629-638`, make the in-place write an async D2H. Keep bool masks on GPU (~6 MB). Lift the Flex-backend guard (`hymm/utils/range_attention_config.py:65-67`), since Flex is mandatory on SM89 and below. Adapt the per-layer pipeline loops at `causal_ar.py:1068-1080`, `2603-2605`, `2689-2757`, `3349`. Also fix the allocation formula which reserves one extra segment that the final segment never writes (`:1816-1828`, `:2165`).
3. **Per-layer PixelUMM streaming.** Hook `DCAEVPixelUMMDecoder.layers` (`pixelumm_decoder/model.py:344-345`) the same way. Keep `compile_decoder=false`, `parallel_mode=none` (the default `window` raises without a process group, `loader.py:95-100`), `seg_ar_segment_vae_decode=false` (per-segment decode runs inside the DiT residency and is strictly worse for memory). Optional: reorder windows-inside-layers so weights stream once per clip instead of once per window (`_predict_x0`).
4. **Text encoder.** Either per-layer streaming hooks on the HF decoder layers (wrap the encode loop in one `on_device()` so the model is not moved per prompt), or bypass it entirely: load only the tokenizer (still needed for action offsets, `base.py:474-672`) and override `_encode_single_prompt` (`base.py:268-295`) to read cached `[1,1280,3584]` BF16 hidden states + mask from disk.
5. **Optional exact savings.** Chunk the DiT MLP over tokens (`mlp_layers.py:14-34` already has `fixed_m_linear`) to cut the ~32 KB-per-token fc1/GELU peak; drop padded text tokens from the cache (always masked, RoPE is image-only, so exact; requires remapping the absolute action-tail indices at `nomas_cpt.py:3356-3363`, `3545-3565`).

Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already set in `scripts/inference/inference_from_config.sh:31`). Keep `inference.sparse_attention.enabled=false` or cap `router_workspace_gib` (default 16 GB).

## 7. Settings that are *not* quantization but *do* change output

- `inference.sliding_window` (`recent`, `uniform`, `temporal_context`, `recent_mix_uniform`): bounds the attended history. Exact only while history <= `past_frames`. The official 4090 preset uses 18 frames.
- `model.vae.tile_pixels > 0`: spatial tiling with blending in the decoder; the shipped decoder is trained to run untiled.
- `model.vae.temporal_window_latents != 10`: the code deliberately avoids untrained window lengths.
- DCAE-V as decoder instead of PixelUMM: selectable (`model.vae.name=dcae-v-f32t4c32`), 2.4 GB of decoder weights, but its full-plane 40-frame conv tiles need an estimated 14 to 23 GB of activations at 720p with no spatial tiling implemented, and `vae.cpu_offload` would break it (`decode` lacks `@cpu_offloaded`). Not a low-memory option without a patch, and quality relative to PixelUMM is unknown.

## 8. How to verify on a real GPU

`inference.profile_memory=true` records allocated / reserved / peak at every stage boundary (`model_load`, `text_encode`, `*_H2D`, `*_D2H`, `denoise`, `cache_write`, `vae_decode`, `postprocess`) into `<video>.stages.rank0.jsonl` and `<video>.performance.json`. `profiling.profiler_record_memory` is training-only; wrap `pipeline.inference` in `torch.cuda.memory._record_memory_history` yourself for an allocator timeline. Check the negotiated PCIe link with `nvidia-smi -q` and measure pinned H2D bandwidth before trusting the overlap estimates. Use native Linux: the kernels (FlashAttention 3, MagiAttention, CUTLASS) are Linux-only and WSL2 limits pinned memory.

## Things that could not be determined statically

Real segment lengths (`video_length: event` takes them from prompt timestamps), valid vs padded text token counts, CUDA context and Inductor workspace size, FlexAttention workspace, whether `transformers==4.57.1` loads the text encoder in FP32 first, achieved PCIe bandwidth and pinned-memory limits on the target host, whether in-place copies into pre-allocated buffers behave under `inference_mode: true`, and whether MagiAttention runs on consumer Blackwell (SM120).

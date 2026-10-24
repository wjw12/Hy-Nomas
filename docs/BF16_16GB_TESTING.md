# Testing the BF16 16 GB streaming path on a real machine

These commits add two features that have only been verified statically (no GPU
was available while writing them; the CPU unit tests pass, see §2.1):

1. **BF16 layer streaming** (`configs/experiment/single_gpu_bf16_16gb_inference.yaml`):
   the full-precision DiT, PixelUMM decoder and text encoder are streamed layer
   by layer from pinned host memory, and the KV history is kept on the host
   and staged per layer to the GPU. No INT8, no FP8, no sliding window. Same
   model, sampling schedule and BF16 precision as the official preset; not
   bitwise identical (attention backend, sequence-parallel degree and
   `torch.compile` change floating-point rounding). The history is still not
   free on the GPU: see the capacity formula in §3.2.
2. **Reference-image KV seeding** (`inference.reference_*`): a still image is
   encoded with DCAE-V, repeated as a short static clip, and written into the
   KV history as "segment 0" before generation.

Everything below is what to run, in which order, what to look at, and what to
try when something fails or is slow. Design background:
[`LOW_VRAM_FEASIBILITY.md`](LOW_VRAM_FEASIBILITY.md).

---

## 1. Machine requirements

| Item | Requirement | Why |
|---|---|---|
| OS | Native Linux (the Docker image from the README is fine) | FlashAttention / MagiAttention / CUTLASS kernels are Linux-only. WSL2 caps pinned memory and breaks the async copies. |
| GPU | 12 to 16 GB, SM80 or newer (Ampere+). `flex` backend on SM89 and below, `magi` on SM90+ | FlexAttention is the only cached-attention backend on consumer cards. |
| PCIe | x16 slot, Gen4 preferred. Check with `nvidia-smi -q \| grep -A3 "GPU Link Info"` | Streaming moves ~17 GB of weights per DiT forward; Gen3 or x8 doubles the transfer time (still hidden at 720p, not at low resolution). |
| Host RAM | 64 GB minimum for short clips, 96 to 128 GB for 20 s+ rollouts | Pinned: 17.1 GB (DiT) + 14.1 GB (text encoder: Qwen2.5-VL-7B language model only, 28 layers + embeddings, no vision tower) + 13.9 GB (PixelUMM) + 389 MB of KV per generated latent frame at 720p (about 2.3 GB per second of video). Plus transient load spikes (see §6). |
| Container | `--ipc=host --ulimit memlock=-1 --shm-size 32g` | Weights and KV host buffers are pinned with `cudaHostRegister` at their exact size (no power-of-two rounding). If registration fails the code falls back to `pin_memory=True`, then to pageable memory with a warning; the log then shows `pinned=False` and copies become synchronous. |
| Assets | `DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors` (FP32, cast to BF16 on load), `VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors` (not the FP8 file), `Qwen/Qwen2.5-VL-7B-Instruct`, and for the reference image `VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt` | The INT8 / FP8 files are rejected by the streaming validation on purpose. |

Environment variables (the launcher already sets the first one):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:8
export TORCH_INDUCTOR_FX_GRAPH_CACHE=1
```

The streaming code registers its own host buffers with `cudaHostRegister`
(`allocate_pinned_exact` in `hymm/utils/module_streaming.py`), so the
`pinned_use_cuda_host_register` setting only affects PyTorch's own pinned
allocations (the fallback path and any `pin_memory=True` tensors elsewhere).
Keep it on; it is harmless and makes the fallback as fast as the primary path.

---

## 2. Test ladder (run in this order)

Stop at the first failing rung and look at §4 before continuing.

### 2.1 Unit tests and config composition (no models needed)

```bash
python -m pytest tests/inference/test_module_streaming.py tests/inference/test_reference_image.py -q
```

Expected: 108 passed (on a CPU-only host without `flash_attn` the one
`hymm.inference` import test is skipped instead). Then compose the preset once:

```bash
python -c "import sys; sys.argv=['x','+experiment=single_gpu_bf16_16gb_inference']; from configs.config import parse_args; c=parse_args(); from hymm.utils.streaming_config import validate_streaming_config; validate_streaming_config(c); print('ok')"
```

### 2.2 Model load only

Run any prompt with `inference.max_iters=0` is not supported, so use a tiny
clip instead (next rung) and read `model_load.rank0.json` in the save path
first. Expect:

- `dit` / `text_encoder` / `vae` parameter bytes all on `cpu`, `policy: layer_streaming`, `pinned: true`.
- Log lines `DiT block streaming: 54 units x ~299 MiB (313 MB), slots=2, resident_blocks=0, resident_other=~295 MiB, pinned=True, pack_seconds=...`, `Text encoder layer streaming: 28 layers x 466 MiB ...`, `PixelUMM layer streaming: 36 layers x 386 MiB ...`.
- Host RAM after load: about 45 GB resident, GPU: only the CUDA context.

If `pinned=False` appears, fix memlock limits before measuring anything else.

### 2.3 Smoke clip: 480p, one segment, 2 seconds

Create `prompts_smoke.json` with one entry, one segment, `start_time 00:00.000`
and `end_time 00:02.000`, then:

```bash
export Hy_Nomas_Model_Path=/path/to/Hy-Nomas
bash scripts/inference/inference_from_config.sh configs/experiment/single_gpu_bf16_16gb_inference.yaml \
  dit_weight="${Hy_Nomas_Model_Path}/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" \
  model.vae.checkpoint="${Hy_Nomas_Model_Path}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors" \
  inference.video_size="[480,832]" \
  inference.prompt=./prompts_smoke.json \
  inference.save_path=./outputs/bf16_smoke
```

Goal: the whole pipeline runs end to end. The video quality is irrelevant here.

### 2.4 720p, one segment of 5 seconds

Same command with `inference.video_size="[720,1280]"` and a 5 s segment. This is
the main memory test for the DiT phase (about 28k tokens per forward).

### 2.5 720p, three segments (15 s)

Three segments of 5 s. Exercises the growing host-staged KV (write-back,
staging regrowth at each segment boundary) and the per-segment transfer cost.

### 2.6 Reference image

Add to the prompt entry:

```json
"reference_image": "ref.png",
"reference_prompt": "A small orange fox sits on a stone path in a luminous forest.",
"reference_latent_frames": 4
```

and pass `inference.reference_vae_checkpoint="${Hy_Nomas_Model_Path}/VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt"`.
Only the DCAE-V encoder is loaded (0.92 B params = 1.7 GiB BF16, memory-mapped
from the checkpoint, no decoder). Activations are about 0.75 GiB per pixel
frame at 720p (static estimate, capped at the 40-frame chunk): the default
single-chunk encode (4 pixel frames) should peak near 4.7 GiB plus cuDNN
workspace (budget 6.5 GiB); `reference_encode_full_clip=true` at
`reference_latent_frames=4` encodes 16 pixel frames and needs about 14 GiB,
so on a 16 GB card the pipeline logs `reference_encode_full_clip: ... falling
back to the single-chunk encode` and continues. The encode runs before the DiT
phase, so it does not stack with the streamed blocks. With `sp_size > 1` only
SP rank 0 encodes and broadcasts the latent. Phone JPEGs are rotated according
to their EXIF orientation before cropping. Not allowed together with the
reference image: `preserve_kv_cache`, QVG, `one_shot`, the sliding window.
See §5 for what to compare.

### 2.7 Equivalence check (needs a second machine with >= 40 GB)

Run the same prompt, seed and resolution with the stock `inference.yaml`
(resident, `sp_size=1` on a big GPU, `seg_ar_attention_backend=flex` so the
kernel matches) and with the 16 GB preset, both with `inference.save_latents=true`.
Compare the saved `.npy` latents:

```python
import numpy as np; a=np.load("resident.npy"); b=np.load("streamed.npy"); print(np.abs(a-b).max(), np.abs(a).mean())
```

Streaming moves bytes only, so the difference should be zero or at the level
of allocator / compile nondeterminism (1e-3 relative or less in BF16). A
systematic difference means a wrong write-back range or stale staging data.

---

## 3. What to observe

### 3.1 Files written next to each video (`inference.profile_memory=true` is on in the preset)

| File | What to look at |
|---|---|
| `<video>.stages.rank0.jsonl` | One row per stage boundary with `allocated`, `reserved`, `peak` GiB and wall time. Stages: `text_encode`, `te_H2D`/`te_D2H`, `dit_H2D`/`dit_D2H`, `denoise`, `cache_write`, `vae_decode`, `pixelumm_H2D`/`pixelumm_D2H`, `postprocess`, `output_to_cpu`. The `peak` of `denoise` and `vae_decode` are the two numbers that decide whether the card is big enough. |
| `<video>.performance.json` | `peak_allocated_gib`, `peak_reserved_gib` for the sample, plus per-stage seconds. |
| `<video>.offload.rank0.json` | Per model: `policy: layer_streaming`, `unit_loads`, `bytes_streamed`, `resyncs`, `phases`, `transfer_seconds`, `pinned`. **`resyncs` must be 0.** A non-zero value means the block loop visited units in an unexpected order and the streamer fell back to synchronous reloads (correct but slow). |
| `<video>.kv.rank0.json` | Written when the KV audit is active. With host staging look at `host_staged_kv`: `host_bytes` / `host_kv_storage_bytes` (pinned history on the host), `staging_bytes` (peak GPU staging, reported as the cache's GPU bytes), `staging_capacity_tokens`, `layer_loads`, `bytes_h2d`, `bytes_d2h`, `resyncs` (must be 0), `forwards`, `staging_allocations` (should be a handful per clip: staging grows by 25 % steps, not once per segment). The staging ring is released in a `finally` right after the DiT phase, so decode starts with no KV bytes on the GPU. The file is written whenever host staging is on. |
| `model_load.rank0.json` | Load-time memory and the residency snapshot. |
| `<video>.metrics.rank0.json` | `model_storage_bytes` per model (should be host-resident) and compile statistics when `torch_compile=true` (`compilation.recompiles` should stay at a handful, not grow per block or per step). |

### 3.2 Expected numbers (static estimates, 720p, 5 s segments; confirm them)

| Quantity | Expectation |
|---|---|
| GPU peak, DiT phase | about 4 to 6 GB with `dit_resident_blocks=0` and a short history (0.63 GB slots + 0.31 GB resident modules + 1.3 to 2.5 GB activations + CUDA context), plus the KV working set below |
| GPU KV working set (grows with history) | `(kv_cache_host_staging_slots + B) x 8 KiB x (history_tokens + one chunk)`, B = guidance batch (1 without CFG). At 720p that is about 22 MB per latent frame of history (about 130 MB per second of video; 29 MB per frame with CFG), so roughly 60 to 75 s of 720p history fit on a 16 GB card. The pre-flight check (`inference.kv_cache_host_staging_preflight`, on by default) refuses longer requests before the first segment; the INFO line `Host-staged KV estimate: ...` prints the prediction for every request. Host RAM grows faster (389 MB per latent frame), so on a 128 GB box the host fills first. |
| GPU peak, decode | about 5 GB (0.8 GB slots + ~4 GB activations for a 10-latent window) |
| GPU peak, text encode | about 2 to 3 GB (1.1 GB embedding + 0.93 GB slots + activations) |
| Per DiT forward | 16.3 GB of weights + 442 KB x history tokens streamed; ~0.7 s on Gen4 x16 against 9 to 20 s of compute depending on the card |
| Per 5 s 720p segment (8 steps + 1 cache write) | roughly 85 to 110 s on a 4080-class card, 140 to 180 s on a 4070 / 5070, 200 s+ on a 4060 Ti (x8 link) |
| Streaming overhead vs infinite VRAM | 5 to 15 %. If `transfer_seconds` in the offload file approaches the denoise wall time, the copies are not overlapping (see §4). |
| Host RAM steady state | ~45 GB pinned weights + KV: 389 MB per latent frame (6 latent frames per second at 24 fps) |

### 3.3 Live monitoring

- `nvidia-smi dmon -s pucm` in a second terminal: memory should plateau at the
  numbers above; PCIe RX utilization should be high but not pinned at 100 %
  during denoise (if it is, the run is bandwidth-bound).
- `watch -n1 free -g`: host RAM should not grow during the denoise steps, only
  at segment boundaries (KV write-back) by the expected 389 MB per latent frame.
- For a definitive overlap check record one segment with Nsight Systems
  (`nsys profile -t cuda,nvtx python hymm/inference.py ...`) and look for the
  copy stream's `Memcpy HtoD` ranges sitting under the compute stream's kernels.
  Gaps where the compute stream waits on a copy event mean the prefetch depth
  is too shallow (raise `*_streaming_slots` to 3).

---

## 4. Likely issues and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| `RuntimeError: ... streamed module ran outside an on_device() phase` | A code path called the DiT / decoder / encoder outside the phase scope (demo server, a custom script, `prewarm`). | Wrap the call in `pipeline._dit_denoise_residency()` / `_text_encode_phase()` or call `decode` (which is decorated). Report the call site. |
| `Cannot set data ... inference tensor` or version-counter errors at the first block | `param.data` rebinding under `torch.inference_mode` behaves differently than expected in this torch build. | Set `inference.inference_mode=false` (falls back to `no_grad`, same memory). Report the torch version. |
| `ValueError: kv_cache_host_staging pre-flight: the DiT phase is predicted to need X GiB ... but only Y GiB is free` before the first segment | The requested video length puts the GPU KV working set (formula in §3.2) over the free VRAM. | Shorten the video, set `inference.max_cache_segments`, lower the resolution, or avoid CFG. The message states the maximum number of latent frames and seconds that fit. `inference.kv_cache_host_staging_preflight=false` skips the check (the run will then OOM late, after the earlier segments were computed). If a run that passed the check still OOMs in `denoise`, or one that failed it would clearly have fit, report the predicted and measured numbers: the constants (48 KiB activations per token, the `+ B` broadcast rows) live in `hymm/utils/streaming_config.py`. |
| `RuntimeError: Expected all tensors to be on the same device` inside a DiT block, decoder layer or text-encoder layer | A streamed unit's weights were read outside its own call (a stray access after the unit was released; its parameters are rebound to the host copy on release on purpose, so this fails loudly instead of reading another layer's weights). | Find the call site in the traceback and report it; it must run inside the unit's hooked `__call__`. |
| OOM in `denoise` on 16 GB | Segment too long (activations scale with tokens), or the KV working set larger than the pre-flight estimate (for example CFG batch > 1, or `torch.cat` copies in the attention path). | Check `staging_capacity_tokens` and `staging_bytes` in the KV audit; shorten segments to <= 5 s at 720p; keep `dit_resident_blocks=0`; try `inference.torch_compile=true/false` (compiled pre/post lowers activation peaks). If the estimate was wrong, report the audit numbers so the constants in `hymm/utils/streaming_config.py` can be corrected. |
| `cudaHostRegister(... bytes) failed` followed by `Falling back to torch pin_memory=True` | `cudaHostRegister` is not supported or not permitted in this driver / container mode. | Harmless but slower to load and up to 2x host RAM for the KV buffers (power-of-two rounding). After the first failure registration is disabled for the process. Check `ulimit -l` and the container flags from §1; if `Pinned host allocation failed ...; using pageable memory` also appears, the copies are synchronous and `pinned=False` shows in the offload snapshot. |
| OOM in `vae_decode` | Decode activations (~4 GB per 10-latent window) plus leftovers. | `kv_cache.release_staging()` runs right after the DiT phase; if `torch.cuda.memory_allocated()` is still above ~1 GB when decode starts, something else is resident: print `torch.cuda.memory_summary()` at the start of `vae_decode`. `model.vae.tile_pixels=256` reduces activations but changes output slightly; use only as a diagnostic. |
| `pinned=False` in logs | memlock limit or not enough free host RAM. | `--ulimit memlock=-1`, close other processes. Pageable copies make `transfer_seconds` roughly equal to the denoise time. |
| `resyncs > 0` | Units visited out of order (for example a block skipped on some path, or hooks firing twice). | Open an issue with the log; as a workaround raise `dit_streaming_slots=3`. Output is still correct. |
| `compilation.recompiles` grows every step or every block | Inductor guards on storage or stride of the rebound weights. | Set `inference.torch_compile=false`. Memory rises a little (eager pre/post attention); speed drops a few percent. |
| Slow despite high GPU utilization | PCIe x8 / Gen3, or pageable memory. | Check the link, pin memory, raise `dit_resident_blocks` (each unit is 313 MB) to use spare VRAM. |
| `HostStagedKVCache ... exceeds host capacity` | A chunk wrote more tokens than the capacity formula planned (unexpected segment geometry). | Report prompt timestamps and `video_size`; check the "KV cache initialized" log against `tokens_per_frame x latent frames + text_len x segments`. |
| Text encoder: `self.model.device` errors | HF property reads the first parameter while layers are host-resident. | The pipeline passes `device` explicitly; if a new call path does not, pass `device=pipeline.device`. |
| Reference run: `reference_image needs the DCAE-V encoder` | PixelUMM is decoder-only. | Set `inference.reference_vae_checkpoint`. |
| Reference run: OOM while encoding the image | DCAE-V encoder activations at 704x1280. | Keep `reference_encode_full_clip=false` (one 4-frame chunk) and `reference_latent_frames` small; the encoder is freed before the DiT phase. |

---

## 5. Reference image: what to compare

The model was trained to continue its *own* generated history, never real
frames, so this is an experiment. Keep everything else fixed and vary one knob
at a time:

1. **Baseline**: same prompt and seed without `reference_image`.
2. **Reference, default**: `reference_latent_frames=4`, derived prompt.
3. Vary `reference_latent_frames` (1, 4, 8): longer static clips give the
   model more "history" to copy from but also more static context.
4. `reference_encode_full_clip=true`: lets the chunk-causal encoder see the
   whole static clip instead of repeating one latent.
5. `reference_prompt`: a caption that describes the image closely vs the
   derived segment-0 prompt.
6. `inference.cache_noise_sigma` (default 0.206897): the reference is re-noised
   to the same level as generated history. A lower sigma keeps more detail,
   a higher one hides the real-vs-generated gap.

Look at: identity of the subject in segment 0 and segment 2 (does it drift
less than the baseline?), colour / exposure shift in the first seconds (a real
frame has different statistics than a generated one), and whether the first
generated frames freeze or jump (the model may interpret the static reference
as "the camera is not moving"). The log line `Reference history seeded: N
latent frames, +T cache tokens` confirms the seeding ran; `global_frame_counter`
in the first chunk log should start at `reference_latent_frames`.

---

## 6. Performance work after the first measurements

In order of expected payoff:

0. **Tiled history attention** (removes the history-length bound on VRAM):
   today each staging slot holds one layer's *whole* history and the attention
   code in `hymm/models/dit/nomas_cpt.py` concatenates it with the current
   tokens (`expand` + `torch.cat`), so the GPU working set is
   `(slots + B) x 8 KiB x history_tokens`. Attending to the history in
   fixed-size tiles staged from the host and merging the partial results with
   log-sum-exp (the FlashAttention split-KV trick) makes staging O(tile)
   instead of O(history) and is exact. Passing the batch-1 history to the
   kernel as a broadcast instead of `expand` + `cat` removes the extra copy
   even without tiling. This is the one change that would make 16 GB truly
   independent of video length; host RAM (389 MB per latent frame) would then
   be the only bound.
1. **Resident blocks**: with the measured headroom, set `dit_resident_blocks` so
   the DiT phase uses about 12 GB on a 16 GB card at the longest history you
   plan to generate. Each block saves 313 MB of PCIe traffic per forward.
2. **Prefetch depth**: `dit_streaming_slots=3` / `kv_cache_host_staging_slots=3`
   if Nsight shows the compute stream waiting on copies.
3. **Decoder weight traffic**: `_predict_x0` runs windows outermost, so the 36
   layers stream once per window. Reordering to run every window through
   layer *i* before loading layer *i+1* cuts decoder traffic from
   13.9 GB x windows to 13.9 GB per clip. Only worth it if decode time matters.
4. **Text encoder embedding cache**: encode prompts once and store the
   `[1, 1280, 3584]` BF16 hidden states to disk; generation then never loads
   the 14 GB encoder.
5. **Host RAM at load**: the DiT loader now casts to BF16 while reading. The
   text encoder still loads FP32 first (transformers default); passing
   `torch_dtype=torch.bfloat16` to `from_pretrained` in
   `hymm/models/text_encoder/encoder.py` would halve that spike.
6. **DCAE-V encoder streaming**: the encoder is already loaded without the
   decoder; its ~6.5 GiB activation peak at 720p could be cut further by
   encoding spatial tiles or by streaming its stages like the other models.
7. **MLP chunking** over tokens inside the double block (exact) to lower the
   ~32 KiB per token fc1/GELU peak for long segments.
8. **Padded text tokens** in the KV history are always masked; dropping them
   is exact and saves up to 432 KiB per padded token per segment, but needs
   the absolute action-tail indices remapped.

Please record the measured `peak` per stage, `transfer_seconds`, wall time per
segment, GPU model, PCIe link and host RAM in the issue or PR so the estimates
in §3.2 can be replaced with real numbers.

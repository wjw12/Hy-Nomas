"""Exact BF16 KV history kept in pinned host memory, staged per layer on demand.

The segment-AR DiT reads ``kv_cache_list[index]`` once per double block and
then slices ``cache["k"][:, :cache_valid_len]`` (denoise) or writes
``cache["k"][:, valid:new_end]`` in place and views ``[:, :total]`` (cache
write). :class:`HostStagedKVCache` keeps every layer's full K/V in pinned host
memory and owns a ring of GPU staging slots the size of ONE layer's working
range. Indexing layer ``i`` binds that layer's dict to a prefetched staging
slot and starts the copy of layer ``i + slots - 1`` on a side stream, so the
GPU holds ``slots`` layers of history instead of 54. Values are never
quantized or dropped.

GPU memory still grows with the history length, only 54 / (slots + B) times
more slowly than a resident cache: each staging slot holds one layer's whole
history, and the DiT materializes the broadcast history once more per
guidance branch (``history_k.expand(B, ...)`` + ``torch.cat`` with the
current chunk). Per token of history (``T`` tokens, BF16)::

    GPU  ~= base + (slots + B) * 2 * heads * head_dim * 2 bytes * T
    host ~=   54 *              2 * heads * head_dim * 2 bytes * T

At 720p (16 heads x 128, 880 tokens per latent frame) that is ~22 MB of GPU
per latent frame of history at slots=2, B=1 (~130 MB per second of video) and
~389 MB of host RAM per latent frame. Host RAM usually runs out first; on a
16 GB GPU with a very large host the GPU bound is roughly 60-75 s of 720p
history. ``hymm.utils.streaming_config.host_staged_kv_preflight`` checks a
request against free GPU memory before the cache is built.

Pipeline protocol (see ``CPTCausalInferencePipeline``)::

    cache.begin_forward(valid_len=cvl, extra_tokens=img+txt, update=True)
    transformer.forward_with_cache(..., kv_cache_list=cache, ...)
    cache.end_forward()

``begin_forward`` makes the staging slots hold at least ``valid_len +
extra_tokens`` (``extra_tokens`` is the current chunk's img+txt token count on
cache-write forwards, 0 otherwise; capacity grows geometrically in 256-token
steps, capped at the host capacity), prefetches the first layers, and keeps already
prefetched layers when consecutive forwards share the same geometry (the
eight denoise steps of a segment). On update forwards, each layer's newly
written range is copied back to the host before its slot is reused.

Slot reuse is ordered by CUDA events: a slot is only refilled on the copy
stream after the compute stream has finished every kernel that reads it,
including across forwards (``end_forward`` records the "slot free" event of
the layers still bound when the forward ends). A layer's dict is reset to an
empty placeholder as soon as its slot is released, so no dict ever holds a
view into a slot that another layer may own.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from hymm.utils.module_streaming import RingScheduler, allocate_pinned_exact

_STAGING_GRANULE = 256  # staging capacity grows in multiples of this many tokens


class HostStagedKVCache(list):
    def __init__(
        self,
        *,
        num_blocks: int,
        batch_size: int,
        max_cache_tokens: int,
        cache_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        slots: int = 2,
    ) -> None:
        if dtype != torch.bfloat16:
            raise ValueError("HostStagedKVCache keeps exact BF16 history; set kv_cache_quantization=none")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("HostStagedKVCache requires a CUDA execution device")
        self.num_blocks = int(num_blocks)
        self.batch_size = int(batch_size)
        self.capacity = int(max_cache_tokens)
        self.cache_heads = int(cache_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.slots = int(slots)
        if self.slots < 2:
            raise ValueError("kv_cache_host_staging_slots must be >= 2")
        self._pin_memory = True
        self._host: List[Dict[str, torch.Tensor]] = [
            {"k": self._host_empty(self.capacity), "v": self._host_empty(self.capacity)}
            for _ in range(self.num_blocks)
        ]
        empty_shape = [self.batch_size, 0, self.cache_heads, self.head_dim]
        self._placeholder = {"k": self._device_empty(empty_shape), "v": self._device_empty(empty_shape)}
        super().__init__(
            [{"k": self._placeholder["k"], "v": self._placeholder["v"]} for _ in range(self.num_blocks)]
        )
        self.scheduler = RingScheduler(list(range(self.num_blocks)), self.slots)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._staging: List[Dict[str, torch.Tensor]] = []
        self._staging_capacity = 0
        self._slot_layer: List[Optional[int]] = [None] * self.slots
        self._ready: Dict[int, torch.cuda.Event] = {}
        self._free: Dict[int, torch.cuda.Event] = {}
        self._bound_slot: Dict[int, int] = {}
        self._served: set = set()
        self._in_forward = False
        self._forward_valid = -1
        self._forward_needed = -1
        self._forward_update = False
        self._ring_valid_for: Optional[tuple] = None
        self.layer_loads = 0
        self.bytes_h2d = 0
        self.bytes_d2h = 0
        self.resyncs = 0
        self.forwards = 0
        self.staging_allocations = 0
        self.peak_staging_bytes = 0

    # ------------------------------------------------------------------
    # allocation helpers
    # ------------------------------------------------------------------
    def _host_empty(self, tokens: int) -> torch.Tensor:
        shape = [self.batch_size, int(tokens), self.cache_heads, self.head_dim]
        tensor, pinned = allocate_pinned_exact(shape, self.dtype, pin_memory=self._pin_memory)
        # Copies use non_blocking only while every host tensor is pinned.
        self._pin_memory = self._pin_memory and pinned
        return tensor

    def _device_empty(self, shape) -> torch.Tensor:
        return torch.empty(shape, dtype=self.dtype, device=self.device)

    @property
    def pin_memory_enabled(self) -> bool:
        return self._pin_memory

    def host_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for layer in self._host for t in layer.values())

    def staging_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for slot in self._staging for t in slot.values())

    def _unbind(self, layer: int) -> None:
        entry = list.__getitem__(self, layer)
        entry["k"] = self._placeholder["k"]
        entry["v"] = self._placeholder["v"]

    def _unbind_entries(self) -> None:
        """Point every layer dict back at the empty placeholder.

        Drops the last references to staging views so the staging memory is
        really released when the slots are freed or regrown.
        """
        for layer in range(self.num_blocks):
            self._unbind(layer)

    def _staging_target(self, needed: int) -> int:
        """Capacity to allocate when ``needed`` tokens no longer fit.

        Every update forward needs ``valid + extra`` tokens, more than the last
        one, so an exact-size ring would be reallocated (drain + empty_cache +
        malloc) on every cache write. Grow by >= 25% in 256-token steps,
        capped at the host capacity; slices still use ``valid_len`` / ``extra``.
        """
        target = max(int(needed), int(1.25 * self._staging_capacity))
        target = (target + _STAGING_GRANULE - 1) // _STAGING_GRANULE * _STAGING_GRANULE
        return max(int(needed), min(target, self.capacity))

    def _ensure_staging(self, needed: int) -> None:
        if needed <= self._staging_capacity and self._staging:
            return
        capacity = self._staging_target(needed)
        self._drain()
        self._unbind_entries()
        self._staging = []
        torch.cuda.empty_cache()
        with torch.inference_mode(False), torch.no_grad():
            shape = [self.batch_size, capacity, self.cache_heads, self.head_dim]
            self._staging = [
                {"k": self._device_empty(shape), "v": self._device_empty(shape)}
                for _ in range(self.slots)
            ]
        self._staging_capacity = capacity
        self.staging_allocations += 1
        self._slot_layer = [None] * self.slots
        self._ring_valid_for = None
        self.peak_staging_bytes = max(self.peak_staging_bytes, self.staging_bytes())

    def _drain(self) -> None:
        self._copy_stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self._ready = {}
        self._free = {}
        self._bound_slot = {}
        self._slot_layer = [None] * self.slots

    # ------------------------------------------------------------------
    # forward protocol
    # ------------------------------------------------------------------
    def begin_forward(self, *, valid_len: int, extra_tokens: int, update: bool) -> None:
        if self._in_forward:
            raise RuntimeError("HostStagedKVCache.begin_forward called twice without end_forward")
        valid_len = int(valid_len)
        needed = valid_len + (int(extra_tokens) if update else 0)
        if needed > self.capacity:
            raise RuntimeError(
                f"KV write of {needed} tokens exceeds host capacity {self.capacity}; grow first"
            )
        self._ensure_staging(max(needed, 1))
        key = (valid_len, needed)
        reusable = (
            self._ring_valid_for == key
            and self.scheduler.next_position == 0
            and not self._forward_update
        )
        if not reusable:
            self._drain()
            self.scheduler.counter = 0
            for layer, slot in self.scheduler.initial_prefetch():
                self._issue_load(layer, slot, valid_len)
        self._in_forward = True
        self._served = set()
        self._forward_valid = valid_len
        self._forward_needed = needed
        self._forward_update = bool(update)
        self._ring_valid_for = key
        self.forwards += 1

    def end_forward(self) -> None:
        if not self._in_forward:
            return
        compute = torch.cuda.current_stream(self.device)
        if self._forward_update:
            # Write back the layers still bound and make the host copy visible
            # to the pipeline before it reads or grows the host tensors.
            if self._bound_slot:
                event = torch.cuda.Event()
                event.record(compute)
                with torch.cuda.stream(self._copy_stream):
                    self._copy_stream.wait_event(event)
                    for layer, slot in list(self._bound_slot.items()):
                        self._writeback(layer, slot)
            self._copy_stream.synchronize()
            self._ring_valid_for = None  # history changed; next forward reloads
        else:
            # The next forward may reuse the ring without draining: its
            # prefetch into the slots still bound here must wait until this
            # forward's attention kernels have finished reading them.
            for slot in set(self._bound_slot.values()):
                event = torch.cuda.Event()
                event.record(compute)
                self._free[slot] = event
        self._unbind_entries()
        self._bound_slot = {}
        self._in_forward = False

    def _issue_load(self, layer: int, slot: int, valid_len: int) -> None:
        free_event = self._free.pop(slot, None)
        with torch.cuda.stream(self._copy_stream):
            if free_event is not None:
                self._copy_stream.wait_event(free_event)
            if valid_len > 0:
                for key in ("k", "v"):
                    # Per batch row: ``[b, :valid_len]`` of a [B, cap, H, D]
                    # tensor is contiguous, ``[:, :valid_len]`` is not for B > 1
                    # (a strided copy would stage through a blocking pageable
                    # temporary).
                    staging, host = self._staging[slot][key], self._host[layer][key]
                    for row in range(self.batch_size):
                        staging[row, :valid_len].copy_(
                            host[row, :valid_len], non_blocking=self._pin_memory
                        )
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)
        self._ready[layer] = ready
        self._slot_layer[slot] = layer
        self.layer_loads += 1
        self.bytes_h2d += 2 * valid_len * self.batch_size * self.cache_heads * self.head_dim * 2

    def _writeback(self, layer: int, slot: int) -> None:
        """D2H of the range written during this update forward (copy stream)."""
        start, end = self._forward_valid, self._forward_needed
        if end <= start:
            return
        for key in ("k", "v"):
            host, staging = self._host[layer][key], self._staging[slot][key]
            for row in range(self.batch_size):  # contiguous rows, see _issue_load
                host[row, start:end].copy_(staging[row, start:end], non_blocking=self._pin_memory)
        self.bytes_d2h += 2 * (end - start) * self.batch_size * self.cache_heads * self.head_dim * 2

    def _resync(self, layer: int, compute) -> None:
        """Restart the ring at ``layer`` after an out-of-order arrival."""
        self.resyncs += 1
        if self._forward_update and self._bound_slot:
            # Layers computed in this update forward still hold their newly
            # written range only in staging; save it before the drain forgets
            # the bindings and the slots are refilled.
            event = torch.cuda.Event()
            event.record(compute)
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(event)
                for bound_layer, bound_slot in list(self._bound_slot.items()):
                    self._writeback(bound_layer, bound_slot)
        for bound_layer in list(self._bound_slot):
            self._unbind(bound_layer)
        self._drain()
        for load_layer, load_slot in self.scheduler.resync_prefetch(layer):
            self._issue_load(load_layer, load_slot, self._forward_valid)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice) or not self._in_forward:
            return list.__getitem__(self, index)
        layer = int(index)
        if layer < 0:
            layer += self.num_blocks
        if layer in self._served:
            # Re-reading a layer in the same forward. While it is bound the
            # dict holds its staging views; once its slot was released the dict
            # was reset to the empty placeholder, so a late read (for example
            # the DiT's ``"swa_capacity" in kv_cache_list[0]`` key check after
            # the block loop) can never see another layer's slot.
            return list.__getitem__(self, layer)
        compute = torch.cuda.current_stream(self.device)
        slot, released, prefetch, in_order = self.scheduler.arrive(layer)
        if (not in_order and not self.scheduler.all_fit) or self._slot_layer[slot] != layer:
            self._resync(layer, compute)
            slot, released, prefetch, _ = self.scheduler.arrive(layer)
        ready = self._ready.get(layer)
        if ready is not None:
            compute.wait_event(ready)
        needed = self._forward_needed
        entry = list.__getitem__(self, layer)
        entry["k"] = self._staging[slot]["k"][:, :needed]
        entry["v"] = self._staging[slot]["v"][:, :needed]
        self._bound_slot[layer] = slot
        self._served.add(layer)
        if released is not None and released != layer:
            released_slot = self._bound_slot.pop(released, None)
            if released_slot is not None:
                self._unbind(released)
                event = torch.cuda.Event()
                event.record(compute)
                if self._forward_update:
                    with torch.cuda.stream(self._copy_stream):
                        self._copy_stream.wait_event(event)
                        self._writeback(released, released_slot)
                        done = torch.cuda.Event()
                        done.record(self._copy_stream)
                    self._free[released_slot] = done
                else:
                    self._free[released_slot] = event
        if prefetch is not None:
            prefetch_layer, prefetch_slot = prefetch
            wraps = self.scheduler.position_of[prefetch_layer] <= self.scheduler.position_of[layer]
            if self._forward_update and wraps:
                # The next forward after an update always reloads (the history
                # changed), so a wrap-around prefetch would only be discarded.
                pass
            elif self._slot_layer[prefetch_slot] != prefetch_layer:
                self._issue_load(prefetch_layer, prefetch_slot, self._forward_valid)
        return entry

    # ------------------------------------------------------------------
    # host-side maintenance used by the pipeline between forwards
    # ------------------------------------------------------------------
    def _require_idle(self) -> None:
        if self._in_forward:
            raise RuntimeError("HostStagedKVCache is inside a forward; call end_forward first")
        self._drain()
        self._ring_valid_for = None

    def grow(self, required_tokens: int, valid_len: int) -> None:
        if required_tokens <= self.capacity:
            return
        self._require_idle()
        valid_len = int(valid_len)
        for layer in self._host:
            for key in ("k", "v"):
                grown = self._host_empty(required_tokens)
                grown[:, :valid_len].copy_(layer[key][:, :valid_len])
                layer[key] = grown
        self.capacity = int(required_tokens)

    def evict_front(self, tokens: int, valid_len: int) -> None:
        """Drop the oldest ``tokens`` of history on the host (shift left)."""
        self._require_idle()
        remaining = int(valid_len) - int(tokens)
        if remaining <= 0:
            return
        for layer in self._host:
            for key in ("k", "v"):
                layer[key][:, :remaining].copy_(layer[key][:, tokens:valid_len].clone())

    def write_history(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write K/V for ``layer`` at host positions ``[start, start + len)``."""
        self._require_idle()
        end = start + int(k.shape[1])
        if end > self.capacity:
            raise RuntimeError(f"history write [{start}:{end}) exceeds capacity {self.capacity}")
        self._host[layer]["k"][:, start:end].copy_(k.to(device="cpu", dtype=self.dtype))
        self._host[layer]["v"][:, start:end].copy_(v.to(device="cpu", dtype=self.dtype))

    def release_staging(self) -> None:
        """Free GPU staging between phases (for example before VAE decode)."""
        self._require_idle()
        self._unbind_entries()
        self._staging = []
        self._staging_capacity = 0
        self._slot_layer = [None] * self.slots
        torch.cuda.empty_cache()

    def snapshot(self) -> dict:
        return {
            "storage": "cpu-host_staged",
            "pinned": self._pin_memory,
            "capacity_tokens": self.capacity,
            "host_bytes": self.host_bytes(),
            "staging_bytes": self.staging_bytes(),
            "peak_staging_bytes": self.peak_staging_bytes,
            "staging_capacity_tokens": self._staging_capacity,
            "staging_allocations": self.staging_allocations,
            "slots": self.slots,
            "layer_loads": self.layer_loads,
            "bytes_h2d": self.bytes_h2d,
            "bytes_d2h": self.bytes_d2h,
            "resyncs": self.resyncs,
            "forwards": self.forwards,
        }

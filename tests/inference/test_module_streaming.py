"""CPU-side checks for the BF16 layer-streaming machinery.

The CUDA ring itself needs a GPU; these tests cover the scheduler decisions,
the flat packing/rebinding of weights, resident-module discovery and the
configuration guards, all of which run without CUDA.
"""

import sys

import pytest
import torch
import torch.nn as nn

from hymm.utils.module_streaming import (
    RingScheduler,
    _Packed,
    all_tensors,
    direct_tensors,
    modules_outside,
)


@pytest.mark.parametrize("units,slots", [(54, 2), (54, 3), (28, 3), (36, 2), (5, 4), (1, 2)])
def test_ring_scheduler_keeps_slot_assignment_consistent_across_forwards(units, slots):
    scheduler = RingScheduler(list(range(units)), slots)
    occupant = {}
    for unit, slot in scheduler.initial_prefetch():
        occupant[slot] = unit
    for _forward in range(3):
        for unit in range(units):
            slot, released, prefetch, in_order = scheduler.arrive(unit)
            assert in_order
            # The unit must already be in the slot the scheduler serves it from.
            assert occupant[slot] == unit
            if prefetch is not None:
                prefetch_unit, prefetch_slot = prefetch
                # Never overwrite the slot we are about to compute from.
                assert prefetch_slot != slot
                occupant[prefetch_slot] = prefetch_unit
            if units > 1:
                assert released == (unit - 1) % units


def test_ring_scheduler_reports_out_of_order_and_resyncs():
    scheduler = RingScheduler(list(range(6)), 2)
    scheduler.initial_prefetch()
    scheduler.arrive(0)
    slot, released, prefetch, in_order = scheduler.arrive(4)
    assert not in_order and released is None and prefetch is None
    loads = scheduler.resync_prefetch(4)
    assert loads[0][0] == 4
    slot, released, prefetch, in_order = scheduler.arrive(4)
    assert in_order and loads[0][1] == slot
    assert prefetch == (5, (slot + 1) % 2)


def test_ring_scheduler_rejects_single_slot():
    with pytest.raises(ValueError):
        RingScheduler([0, 1], 1)


class _Unit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.norm = nn.RMSNorm(8)
        self.register_buffer("table", torch.arange(5, dtype=torch.float32))


def _model():
    model = nn.Module()
    model.blocks = nn.ModuleList([_Unit(), _Unit(), _Unit()])
    model.embed = nn.Linear(4, 8)
    model.head = nn.Linear(8, 2)
    model.register_buffer("root_buffer", torch.ones(3))
    return model.eval().requires_grad_(False)


def test_packed_roundtrip_preserves_values_and_rebinding():
    model = _model()
    unit = [model.blocks[0]]
    before = {name: p.clone() for name, p in model.blocks[0].named_parameters()}
    packed = _Packed(all_tensors(unit))
    packed.pack_to_host(pin_memory=False)
    assert packed.nbytes % 256 == 0
    assert packed.host is not None and packed.host.numel() == packed.nbytes
    # Master copy now lives inside the flat host buffer.
    weight = model.blocks[0].linear.weight
    assert packed.host.data_ptr() <= weight.data_ptr() < packed.host.data_ptr() + packed.nbytes
    for name, param in model.blocks[0].named_parameters():
        assert torch.equal(param, before[name])
    # Bind to a "slot" and back; values unchanged, module still callable.
    slot = torch.zeros(packed.nbytes, dtype=torch.uint8)
    views = packed.views_into(slot)
    slot.copy_(packed.host)
    packed.bind(views)
    assert slot.data_ptr() <= model.blocks[0].linear.weight.data_ptr() < slot.data_ptr() + packed.nbytes
    x = torch.randn(2, 8)
    out_slot = model.blocks[0].linear(x)
    packed.bind_host()
    assert torch.equal(model.blocks[0].linear(x), out_slot)
    assert torch.equal(model.blocks[0].table, torch.arange(5, dtype=torch.float32))


def test_modules_outside_lists_only_non_streamed_tensor_owners():
    model = _model()
    units = [[block] for block in model.blocks]
    resident = modules_outside(model, units)
    owners = {type(m).__name__ for m in resident}
    assert owners == {"Module", "Linear"}  # root (root_buffer), embed, head
    tensors = [t for m in resident for t in direct_tensors(m)]
    streamed = {id(t) for group in units for t in all_tensors(group)}
    assert tensors and all(id(t) not in streamed for t in tensors)
    assert any(t.shape == (3,) for t in tensors)  # the root buffer is covered


def _compose(*overrides):
    from configs.config import parse_args

    sys.argv = ["test", *overrides]
    return parse_args()


def test_bf16_16gb_preset_validates_and_enables_every_streaming_path():
    from configs.cpu_offload import enabled_cpu_offloads
    from hymm.utils.streaming_config import streaming_options, validate_streaming_config

    cfg = _compose("+experiment=single_gpu_bf16_16gb_inference")
    validate_streaming_config(cfg)
    options = streaming_options(cfg)
    assert options["dit_block_streaming"] and options["text_encoder_layer_streaming"]
    assert options["vae_layer_streaming"] and options["kv_cache_host_staging"]
    assert cfg.inference.dit_weight_quantization == "none"
    assert cfg.inference.kv_cache_quantization == "none"
    assert not cfg.inference.sliding_window.enabled
    assert cfg.parallelism.sp_size == 1
    assert set(enabled_cpu_offloads(cfg)) == {
        "dit_block_streaming", "text_encoder_layer_streaming", "vae_layer_streaming",
    }


@pytest.mark.parametrize(
    "override,message",
    [
        ("parallelism.dit_cpu_offload=true", "dit_block_streaming replaces"),
        ("parallelism.text_encoder_cpu_offload=true", "text_encoder_layer_streaming replaces"),
        ("model.vae.cpu_offload=true", "layer_streaming replaces"),
        ("model.vae.compile_decoder=true", "compile_decoder=false"),
        ("inference.kv_cache_quantization=int8", "kv_cache_quantization=none"),
        ("inference.sliding_window.enabled=true", "sliding_window"),
        ("parallelism.dit_streaming_slots=1", ">= 2"),
    ],
)
def test_streaming_guards_reject_unsupported_combinations(override, message):
    from hymm.utils.streaming_config import validate_streaming_config

    cfg = _compose("+experiment=single_gpu_bf16_16gb_inference", override)
    with pytest.raises(ValueError, match=message):
        validate_streaming_config(cfg)


def test_existing_presets_do_not_enable_streaming():
    from hymm.utils.streaming_config import any_streaming_enabled, validate_streaming_config

    for experiment in ("inference", "single_gpu_low_mem_inference"):
        cfg = _compose(f"+experiment={experiment}")
        validate_streaming_config(cfg)
        assert not any_streaming_enabled(cfg)


def test_streaming_guards_reject_zero_slots():
    from hymm.utils.streaming_config import streaming_options, validate_streaming_config

    cfg = _compose("+experiment=single_gpu_bf16_16gb_inference", "parallelism.dit_streaming_slots=0")
    # 0 must not silently become the default of 2.
    assert streaming_options(cfg)["dit_streaming_slots"] == 0
    with pytest.raises(ValueError, match=">= 2"):
        validate_streaming_config(cfg)


@pytest.mark.parametrize(
    "overrides,message",
    [
        (("inference.rollout_type=chunk_ar", "inference.cfg_scale_video=6.0"), "rollout_type=seg_ar"),
        (("inference.rollout_type=chunk_ar", "inference.camera_cfg_scale=2.0"), "rollout_type=seg_ar"),
        (("inference.cfg_scale_video=6.0", "inference.ar_single_slot_cfg_cache=false"),
         "ar_single_slot_cfg_cache=true"),
    ],
)
def test_host_staging_rejects_multi_slot_cfg(overrides, message):
    from hymm.utils.streaming_config import validate_streaming_config

    cfg = _compose("+experiment=single_gpu_bf16_16gb_inference", *overrides)
    with pytest.raises(ValueError, match=message):
        validate_streaming_config(cfg)


def test_host_staging_allows_seg_ar_single_slot_cfg():
    from hymm.utils.streaming_config import validate_streaming_config

    validate_streaming_config(
        _compose("+experiment=single_gpu_bf16_16gb_inference", "inference.cfg_scale_video=6.0")
    )


def test_streaming_skips_sp_topology_check_for_data_parallel(monkeypatch):
    from hymm.utils.model_cpu_offload import validate_model_cpu_offload

    monkeypatch.setenv("WORLD_SIZE", "8")  # torchrun DP prompt sharding, sp_size=1
    cfg = _compose("+experiment=single_gpu_bf16_16gb_inference")
    validate_model_cpu_offload(cfg, "cuda:0")
    with pytest.raises(ValueError, match="CUDA execution device"):
        validate_model_cpu_offload(cfg, "cpu")
    whole_model = _compose(
        "+experiment=single_gpu_bf16_16gb_inference",
        "parallelism.dit_block_streaming=false", "parallelism.dit_cpu_offload=true",
    )
    with pytest.raises(ValueError, match="world_size == sp_size"):
        validate_model_cpu_offload(whole_model, "cuda:0")


# ---------------------------------------------------------------------------
# Exact-size pinned host memory (H1)
# ---------------------------------------------------------------------------
def test_allocate_pinned_exact_without_cuda_is_exact_pageable():
    from hymm.utils.module_streaming import allocate_pinned_exact

    tensor, pinned = allocate_pinned_exact([3, 5, 7], torch.bfloat16)
    assert tensor.shape == (3, 5, 7) and tensor.dtype == torch.bfloat16
    assert not pinned
    assert tensor.untyped_storage().nbytes() == 3 * 5 * 7 * 2
    flat, pinned = allocate_pinned_exact(311_000_001, torch.uint8, pin_memory=False)
    assert flat.numel() == 311_000_001 and not pinned


@pytest.fixture(autouse=True)
def _fresh_host_register_state(monkeypatch):
    """allocate_pinned_exact keeps process-wide register state; isolate every test."""
    from hymm.utils import module_streaming

    monkeypatch.setattr(module_streaming, "_HOST_REGISTER_DISABLED", False)
    monkeypatch.setattr(module_streaming, "_HOST_REGISTER_SUCCEEDED", False)
    monkeypatch.setattr(module_streaming, "_FALLBACK_WARNED", False)


class _FakeCudart:
    """cudart stand-in: a failing call leaves a pending error until cudaGetLastError."""

    def __init__(self, error=0, unregister_error=0):
        self.error = error
        self.unregister_error = unregister_error
        self.registered = {}
        self.unregistered = []
        self.register_calls = 0
        self.pending = 0
        self.cleared = 0

    def cudaHostRegister(self, ptr, size, flags):
        self.register_calls += 1
        if self.error == 0:
            self.registered[ptr] = size
        else:
            self.pending = self.error
        return self.error

    def cudaHostUnregister(self, ptr):
        self.unregistered.append(ptr)
        if self.unregister_error:
            self.pending = self.unregister_error
        return self.unregister_error

    def cudaGetLastError(self):
        error, self.pending = self.pending, 0
        self.cleared += 1
        return error


def test_allocate_pinned_exact_registers_exact_page_rounded_range(monkeypatch):
    import gc

    from hymm.utils import module_streaming

    cudart = _FakeCudart()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    nbytes = 3 * 2**20 + 123  # power-of-two rounding would give 4 MiB
    tensor, pinned = module_streaming.allocate_pinned_exact(nbytes, torch.uint8)
    assert pinned and tensor.numel() == nbytes
    (ptr, size), = cudart.registered.items()
    assert ptr == tensor.data_ptr() and ptr % module_streaming._PAGE == 0
    assert nbytes <= size < nbytes + module_streaming._PAGE
    assert tensor.untyped_storage().nbytes() <= nbytes + 2 * module_streaming._PAGE
    del tensor
    gc.collect()
    assert cudart.unregistered == [ptr]


def test_allocate_pinned_exact_falls_back_when_registration_fails(monkeypatch):
    from hymm.utils import module_streaming

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FakeCudart(error=1))
    tensor, pinned = module_streaming.allocate_pinned_exact([4, 6], torch.float32)
    assert tensor.shape == (4, 6)
    if not pinned:  # CPU-only build: pin_memory=True fails too, pageable fallback
        assert tensor.untyped_storage().nbytes() == 4 * 6 * 4


def _fake_cuda_available(monkeypatch, cudart):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)


def test_failed_registration_consumes_the_pending_cuda_error(monkeypatch):
    """A-M1: the error a failed cudaHostRegister leaves behind must not hit the next kernel."""
    from hymm.utils import module_streaming

    cudart = _FakeCudart(error=1)
    _fake_cuda_available(monkeypatch, cudart)
    tensor, _ = module_streaming.allocate_pinned_exact([16], torch.float32)
    assert tensor.shape == (16,)
    assert cudart.register_calls == 1
    assert cudart.pending == 0 and cudart.cleared == 1


def test_pending_error_falls_back_to_a_kernel_probe_without_get_last_error(monkeypatch):
    """Bindings without cudaGetLastError: a trivial CUDA kernel inside try absorbs the error."""
    from hymm.utils import module_streaming

    class NoLastError(_FakeCudart):
        cudaGetLastError = None

    probes = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            probes.append(args)
            raise RuntimeError("CUDA error: invalid argument (pending)")
        return real_empty(*args, **kwargs)

    cudart = NoLastError(error=1)
    _fake_cuda_available(monkeypatch, cudart)
    monkeypatch.setattr(torch, "empty", empty)
    tensor, _ = module_streaming.allocate_pinned_exact([8], torch.float32)
    assert tensor.shape == (8,) and probes == [(1,)]


@pytest.mark.parametrize("error", [801, 1])
def test_unsupported_or_first_failure_disables_registration(monkeypatch, error):
    """A-M1: 801 (not supported) or a failing first allocation skips registration afterwards."""
    from hymm.utils import module_streaming

    cudart = _FakeCudart(error=error)
    _fake_cuda_available(monkeypatch, cudart)
    for _ in range(5):
        tensor, _ = module_streaming.allocate_pinned_exact([32], torch.uint8)
        assert tensor.numel() == 32
    assert cudart.register_calls == 1
    assert module_streaming._HOST_REGISTER_DISABLED


def test_failure_after_a_success_keeps_registration_enabled(monkeypatch):
    from hymm.utils import module_streaming

    cudart = _FakeCudart()
    _fake_cuda_available(monkeypatch, cudart)
    kept, pinned = module_streaming.allocate_pinned_exact([64], torch.uint8)
    assert pinned
    cudart.error = 2  # e.g. out of lockable memory for this one range
    module_streaming.allocate_pinned_exact([64], torch.uint8)
    assert not module_streaming._HOST_REGISTER_DISABLED
    cudart.error = 0
    _, pinned = module_streaming.allocate_pinned_exact([64], torch.uint8)
    assert pinned and cudart.register_calls == 3
    del kept


def test_host_allocation_failure_is_not_reported_as_register_failure(monkeypatch):
    """A-L1: only the cudaHostRegister call itself counts as a registration failure."""
    from hymm.utils import module_streaming

    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("dtype") == torch.uint8 and args and args[0] == 3 * module_streaming._PAGE:
            raise RuntimeError("host out of memory (simulated)")
        return real_empty(*args, **kwargs)

    cudart = _FakeCudart()
    _fake_cuda_available(monkeypatch, cudart)
    monkeypatch.setattr(torch, "empty", empty)
    tensor, _ = module_streaming.allocate_pinned_exact([2 * module_streaming._PAGE], torch.uint8)
    assert tensor.numel() == 2 * module_streaming._PAGE
    assert cudart.register_calls == 0 and not module_streaming._HOST_REGISTER_DISABLED


def test_failed_registration_drops_the_buffer_before_the_fallback(monkeypatch):
    """A-L1: the unregistered staging buffer is freed before the fallback allocates."""
    import weakref

    from hymm.utils import module_streaming

    alive_at_fallback = []
    bases = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            alive_at_fallback.append([ref() is not None for ref in bases])
            return real_empty(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"})
        tensor = real_empty(*args, **kwargs)
        size = args[0] if args else None
        if kwargs.get("dtype") == torch.uint8 and isinstance(size, int) and size > module_streaming._PAGE:
            bases.append(weakref.ref(tensor))
        return tensor

    _fake_cuda_available(monkeypatch, _FakeCudart(error=1))
    monkeypatch.setattr(torch, "empty", empty)
    module_streaming.allocate_pinned_exact([module_streaming._PAGE * 4], torch.uint8)
    assert alive_at_fallback == [[False]]


def test_failed_unregister_consumes_the_pending_cuda_error(monkeypatch):
    import gc

    from hymm.utils import module_streaming

    cudart = _FakeCudart(unregister_error=1)
    _fake_cuda_available(monkeypatch, cudart)
    tensor, pinned = module_streaming.allocate_pinned_exact([4096], torch.uint8)
    assert pinned
    ptr = tensor.data_ptr()
    del tensor
    gc.collect()
    assert cudart.unregistered == [ptr]
    assert cudart.pending == 0 and cudart.cleared == 1


# ---------------------------------------------------------------------------
# ModuleStreamer install / phase safety
# ---------------------------------------------------------------------------
def test_streamer_rejects_tensors_shared_between_units():
    from hymm.utils.module_streaming import ModuleStreamer

    shared = nn.Linear(4, 4)
    first, second = nn.Sequential(shared), nn.Sequential(shared)
    for module in (first, second):
        module.eval().requires_grad_(False)
    with pytest.raises(ValueError, match="share a tensor"):
        ModuleStreamer([[first], [second]], device=torch.device("cuda"), slots=2)


def test_streamer_rejects_resident_tensor_shared_with_unit():
    from hymm.utils.module_streaming import ModuleStreamer

    model = _model()
    tied = nn.Module().eval()
    tied.weight = model.blocks[0].linear.weight
    with pytest.raises(ValueError, match="shares a tensor"):
        ModuleStreamer([[block] for block in model.blocks], resident=[tied],
                       device=torch.device("cuda"), slots=2)


def test_streamer_enter_failure_rebinds_host_and_resets_depth(monkeypatch):
    from hymm.utils.module_streaming import ModuleStreamer

    model = _model()
    units = [[block] for block in model.blocks]
    streamer = ModuleStreamer(units, resident=modules_outside(model, units),
                              device=torch.device("cuda"), slots=2, resident_units=1)
    streamer.device = torch.device("cpu")  # let enter() allocate its "device" buffers on CPU
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    def oom(unit, slot):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(streamer, "_issue_load", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        streamer.enter()
    assert streamer.depth == 0 and not streamer.active
    assert streamer._slot_buffers == [] and streamer._resident_buffers == {}
    # Resident unit 0 and the resident modules were bound to "device" buffers
    # before the failure; every tensor is back on its host copy.
    for packed in streamer.packed + [streamer.resident_packed]:
        host = packed.host
        for tensor, _, _ in packed.entries:
            assert host.data_ptr() <= tensor.data_ptr() < host.data_ptr() + packed.nbytes


def test_ring_scheduler_all_fit_flag():
    assert RingScheduler([0, 1], 3).all_fit
    assert not RingScheduler([0, 1, 2], 3).all_fit


# ---------------------------------------------------------------------------
# HostStagedKVCache scheduling on CPU stand-ins for CUDA streams/events
# ---------------------------------------------------------------------------
class _FakeEvent:
    clock = 0

    def __init__(self, *args, **kwargs):
        self.stream = None
        self.time = None

    def record(self, stream=None):
        _FakeEvent.clock += 1
        self.time = _FakeEvent.clock
        self.stream = stream


class _FakeStream:
    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)

    def synchronize(self):
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    from contextlib import contextmanager

    compute = _FakeStream("compute")

    @contextmanager
    def stream_scope(stream):
        yield

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream("copy"))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: compute)
    monkeypatch.setattr(torch.cuda, "stream", stream_scope)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return compute


def _cpu_staged_cache(num_blocks=5, slots=2, capacity=16, batch_size=1):
    from hymm.models.dit.modules.host_staged_kv_cache import HostStagedKVCache

    class CPUStagedCache(HostStagedKVCache):
        def _device_empty(self, shape):
            return torch.zeros(shape, dtype=self.dtype)

    cache = CPUStagedCache(
        num_blocks=num_blocks, batch_size=batch_size, max_cache_tokens=capacity, cache_heads=2,
        head_dim=4, dtype=torch.bfloat16, device=torch.device("cuda"), slots=slots,
    )
    for layer in range(num_blocks):
        for key in ("k", "v"):
            cache._host[layer][key].fill_(float(layer + 1))
    return cache


def _slot_of(cache, layer):
    return next(slot for slot, occupant in enumerate(cache._slot_layer) if occupant == layer)


def test_host_staged_reused_forward_waits_for_previous_forward_slot_release(fake_cuda):
    """B1: the first prefetch of a reused forward must wait on the last layer's free event."""
    blocks = 5
    cache = _cpu_staged_cache(num_blocks=blocks)
    copy = cache._copy_stream
    for forward in range(3):
        cache.begin_forward(valid_len=4, extra_tokens=0, update=False)
        if forward:
            # Same geometry: the ring is reused, nothing is drained or reloaded.
            assert cache.resyncs == 0
            waits_before = len(copy.waits)
            entry = cache[0]
            assert torch.all(entry["k"][:, :4] == 1)
            # Layer 1 is prefetched into the slot layer 4 used last forward.
            assert cache._slot_layer[last_slot] == 1
            new_waits = copy.waits[waits_before:]
            assert any(event.stream is fake_cuda and event.time > last_layer_time
                       for event in new_waits), "prefetch into a slot still read by the previous forward"
            start = 1
        else:
            start = 0
        for layer in range(start, blocks):
            entry = cache[layer]
            assert torch.all(entry["k"][:, :4] == layer + 1)
        last_slot = cache._bound_slot[blocks - 1]
        last_layer_time = _FakeEvent.clock
        cache.end_forward()
        assert cache._free.get(last_slot) is not None
    assert cache.resyncs == 0


def test_host_staged_entries_never_keep_staging_views(fake_cuda):
    """H2: released / finished layers point at the placeholder, so staging can be freed."""
    cache = _cpu_staged_cache()
    cache.begin_forward(valid_len=4, extra_tokens=0, update=False)
    cache[0]
    cache[1]
    # Layer 0's slot was released when layer 1 arrived; a late re-read (the DiT
    # checks ``"swa_capacity" in kv_cache_list[0]`` after its block loop) must
    # not expose a slot that layer 2 is being prefetched into.
    assert cache[0]["k"].shape[1] == 0
    assert cache[1]["k"].shape[1] == 4
    for layer in range(2, 5):
        cache[layer]
    cache.end_forward()
    assert all(list.__getitem__(cache, i)["k"].numel() == 0 for i in range(5))
    old_storage = cache._staging[0]["k"].untyped_storage().data_ptr()
    cache.begin_forward(valid_len=8, extra_tokens=4, update=True)  # regrows staging
    # 12 needed, rounded up to the 256-token granule, capped at the host capacity.
    assert cache._staging_capacity == 16
    assert all(list.__getitem__(cache, i)["k"].numel() == 0 for i in range(5))
    for layer in range(5):
        cache[layer]
    cache.end_forward()
    for layer in range(5):
        for key in ("k", "v"):
            entry = list.__getitem__(cache, layer)[key]
            assert entry.numel() == 0
            assert entry.untyped_storage().data_ptr() != old_storage
    assert cache.peak_staging_bytes == cache.staging_bytes() > 0
    cache.release_staging()
    assert cache.staging_bytes() == 0 and cache._staging_capacity == 0
    assert cache.snapshot()["peak_staging_bytes"] > 0


def test_host_staged_resync_writes_back_bound_layers(fake_cuda):
    """A-M3: an out-of-order arrival during an update forward keeps written ranges."""
    cache = _cpu_staged_cache()
    cache.begin_forward(valid_len=2, extra_tokens=3, update=True)
    for layer in (0, 1):
        entry = cache[layer]
        entry["k"][:, 2:5].fill_(10.0 + layer)
        entry["v"][:, 2:5].fill_(20.0 + layer)
    cache[3]  # skips layer 2 -> resync while layer 1 is still bound
    assert cache.resyncs == 1
    for layer in (0, 1):
        assert torch.all(cache._host[layer]["k"][:, 2:5] == 10.0 + layer)
        assert torch.all(cache._host[layer]["v"][:, 2:5] == 20.0 + layer)
    # History below valid_len is untouched.
    assert torch.all(cache._host[1]["k"][:, :2] == 2)
    cache[4]
    cache.end_forward()


def test_host_staged_update_forward_skips_wraparound_prefetch(fake_cuda):
    cache = _cpu_staged_cache()
    cache.begin_forward(valid_len=2, extra_tokens=3, update=True)
    loads = []
    original = cache._issue_load
    cache._issue_load = lambda layer, slot, valid: (loads.append(layer), original(layer, slot, valid))
    for layer in range(5):
        cache[layer]
    cache.end_forward()
    # begin_forward loaded layer 0; the wrap-around prefetch of 0 at layer 4 is skipped.
    assert loads == [1, 2, 3, 4]
    assert cache.resyncs == 0


def test_host_staged_update_forward_writes_back_and_next_forward_reloads(fake_cuda):
    cache = _cpu_staged_cache()
    cache.begin_forward(valid_len=2, extra_tokens=3, update=True)
    for layer in range(5):
        entry = cache[layer]
        entry["k"][:, 2:5].fill_(50.0 + layer)
    cache.end_forward()
    for layer in range(5):
        assert torch.all(cache._host[layer]["k"][:, 2:5] == 50.0 + layer)
    cache.begin_forward(valid_len=5, extra_tokens=0, update=False)
    for layer in range(5):
        assert torch.all(cache[layer]["k"][:, 2:5] == 50.0 + layer)
    cache.end_forward()


def _run_forward(cache, valid_len, extra, value=None):
    cache.begin_forward(valid_len=valid_len, extra_tokens=extra, update=extra > 0)
    for layer in range(cache.num_blocks):
        entry = cache[layer]
        assert entry["k"].shape[1] == valid_len + extra  # never the staging capacity
        if value is not None:
            entry["k"][:, valid_len:valid_len + extra].fill_(value + layer)
            entry["v"][:, valid_len:valid_len + extra].fill_(-(value + layer))
    cache.end_forward()


def test_host_staged_staging_grows_geometrically(fake_cuda):
    """A-L4: consecutive growing cache writes do not reallocate the ring every time."""
    cache = _cpu_staged_cache(capacity=2048)
    _run_forward(cache, 0, 100, value=10.0)
    assert cache.staging_allocations == 1 and cache._staging_capacity == 256
    _run_forward(cache, 100, 0)  # denoise on the grown history
    _run_forward(cache, 100, 100, value=20.0)
    assert cache.staging_allocations == 1  # 200 tokens still fit the 256-token ring
    _run_forward(cache, 200, 100, value=30.0)
    # 300 > 256: max(300, 1.25 * 256) rounded up to 512.
    assert cache.staging_allocations == 2 and cache._staging_capacity == 512
    for layer in range(cache.num_blocks):
        host_k, host_v = cache._host[layer]["k"], cache._host[layer]["v"]
        for start, value in ((0, 10.0), (100, 20.0), (200, 30.0)):
            assert torch.all(host_k[:, start:start + 100] == value + layer)
            assert torch.all(host_v[:, start:start + 100] == -(value + layer))
        # Write-back used valid_len/extra, never the staging capacity.
        assert torch.all(host_k[:, 300:512] == layer + 1)


def test_host_staged_staging_is_capped_at_host_capacity(fake_cuda):
    cache = _cpu_staged_cache(capacity=300)
    _run_forward(cache, 0, 290)
    assert cache._staging_capacity == 300


def test_host_staged_batched_rows_round_trip(fake_cuda):
    """A-L6: B=2 rows are copied per contiguous row and stay distinct."""
    cache = _cpu_staged_cache(batch_size=2)
    for layer in range(5):
        for row in range(2):
            cache._host[layer]["k"][row].fill_(10.0 * layer + row + 1)
            cache._host[layer]["v"][row].fill_(-(10.0 * layer + row + 1))
    cache.begin_forward(valid_len=4, extra_tokens=2, update=True)
    for layer in range(5):
        entry = cache[layer]
        assert entry["k"].shape[:2] == (2, 6)
        for row in range(2):
            assert torch.all(entry["k"][row, :4] == 10.0 * layer + row + 1)
            assert torch.all(entry["v"][row, :4] == -(10.0 * layer + row + 1))
            entry["k"][row, 4:6].fill_(100.0 + 10 * layer + row)
            entry["v"][row, 4:6].fill_(-(100.0 + 10 * layer + row))
    cache.end_forward()
    for layer in range(5):
        for row in range(2):
            host_k, host_v = cache._host[layer]["k"][row], cache._host[layer]["v"][row]
            assert torch.all(host_k[:4] == 10.0 * layer + row + 1)
            assert torch.all(host_k[4:6] == 100.0 + 10 * layer + row)
            assert torch.all(host_v[4:6] == -(100.0 + 10 * layer + row))
            assert torch.all(host_k[6:] == 10.0 * layer + row + 1)  # untouched past the write


# ---------------------------------------------------------------------------
# ModuleStreamer ring on CPU stand-ins (A-L2, A-L3)
# ---------------------------------------------------------------------------
def _cpu_streamer(model, units, **kwargs):
    from hymm.utils.module_streaming import ModuleStreamer

    streamer = ModuleStreamer(units, resident=modules_outside(model, units),
                              device=torch.device("cuda"), **kwargs)
    streamer.device = torch.device("cpu")  # "device" buffers are CPU tensors here
    return streamer


def test_all_fit_ring_allocates_only_the_slots_it_uses(fake_cuda):
    """A-L2: one streamed unit left by resident blocks needs one slot, not ``slots``."""
    model = _model()
    units = [[block] for block in model.blocks]
    streamer = _cpu_streamer(model, units, slots=2, resident_units=2)
    expected = streamer.phase_device_bytes()
    streamer.enter()
    try:
        assert len(streamer._slot_buffers) == 1
        assert streamer.device_bytes() == expected
        assert expected == streamer.unit_bytes + sum(p.nbytes for p in streamer.packed[:2]) + (
            streamer.resident_packed.nbytes
        )
    finally:
        streamer.exit()
    other = _model()
    full = _cpu_streamer(other, [[block] for block in other.blocks], slots=2)
    assert full.phase_device_bytes() == 2 * full.unit_bytes + full.resident_packed.nbytes


def _linear_stack(count=4):
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(count)])
    return model.eval().requires_grad_(False)


def _inside(tensor, buffer):
    start = buffer.data_ptr()
    return start <= tensor.data_ptr() < start + buffer.numel() * buffer.element_size()


def test_released_units_are_rebound_to_their_host_copy(fake_cuda):
    """A-L3: a unit that left its slot must not alias the slot being refilled."""
    model = _linear_stack()
    x = torch.randn(3, 4)
    expected = x
    for block in model.blocks:
        expected = block(expected)
    units = [[block] for block in model.blocks]
    streamer = _cpu_streamer(model, units, slots=2)
    with torch.no_grad():
        streamer.enter()
        try:
            out = x
            for index, block in enumerate(model.blocks):
                out = block(out)
                assert any(_inside(block.weight, slot) for slot in streamer._slot_buffers)
                for earlier in range(index):
                    packed = streamer.packed[earlier]
                    weight = model.blocks[earlier].weight
                    assert _inside(weight, packed.host), f"unit {earlier} still bound to a slot"
                    assert not any(_inside(weight, slot) for slot in streamer._slot_buffers)
            assert torch.equal(out, expected)
            # Out-of-order arrival (resync): every other unit goes back to the host.
            model.blocks[1](x)
            assert streamer.resyncs == 1
            for index in (0, 2, 3):
                assert _inside(model.blocks[index].weight, streamer.packed[index].host)
            assert any(_inside(model.blocks[1].weight, slot) for slot in streamer._slot_buffers)
        finally:
            streamer.exit()


# ---------------------------------------------------------------------------
# DiT checkpoint loading straight into the model (LOW g)
# ---------------------------------------------------------------------------
def _streamed_loader():
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "hymm/models/checkpointing/load.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_load_streamed_into_model")
    namespace = {"torch": torch, "is_camera_param": lambda key: key.startswith("action_modules.")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "load.py", "exec"), namespace)
    return namespace["_load_streamed_into_model"]


def _bf16_model():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(4, 3), nn.Linear(3, 2)])
    model.action_modules = nn.ModuleList([nn.Linear(2, 2)])
    model.register_buffer("table", torch.zeros(3))
    return model.to(torch.bfloat16)


def test_streamed_checkpoint_load_casts_into_model_and_keeps_key_semantics(tmp_path):
    from safetensors.torch import save_file

    load = _streamed_loader()
    source = _bf16_model().float()
    for param in source.parameters():
        nn.init.normal_(param)
    weights = {key: value.float().contiguous() for key, value in source.state_dict().items()
               if not key.startswith("action_modules.")}
    weights["unused.extra"] = torch.ones(2)
    path = tmp_path / "dit.safetensors"
    save_file(weights, str(path))

    model = _bf16_model()
    expected = model.state_dict()
    assert load(model, path, expected, lambda key: key, None) is model
    for key, value in weights.items():
        if key in expected:
            assert model.state_dict()[key].dtype == torch.bfloat16
            assert torch.equal(model.state_dict()[key], value.to(torch.bfloat16))

    missing = dict(weights)
    del missing["blocks.0.weight"]
    save_file(missing, str(path))
    with pytest.raises(ValueError, match="Missing 1 non-camera"):
        load(_bf16_model(), path, _bf16_model().state_dict(), lambda key: key, None)

    wrong = dict(weights)
    wrong["blocks.1.bias"] = torch.zeros(5)
    save_file(wrong, str(path))
    with pytest.raises(ValueError, match="Incompatible DiT tensor shapes"):
        load(_bf16_model(), path, _bf16_model().state_dict(), lambda key: key, None)


# ---------------------------------------------------------------------------
# Resident-VAE warning (A-L7)
# ---------------------------------------------------------------------------
def _warnings_during(callable_):
    from loguru import logger

    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        callable_()
    finally:
        logger.remove(sink)
    return messages


def test_block_streaming_warns_when_the_vae_stays_resident():
    from hymm.utils.streaming_config import validate_streaming_config

    resident_vae = _compose(
        "+experiment=single_gpu_bf16_16gb_inference",
        "model.vae.name=dcae-v-f32t4c32", "model.vae.layer_streaming=false",
    )
    messages = _warnings_during(lambda: validate_streaming_config(resident_vae))
    assert any("stays resident" in m and "2.4 GB" in m for m in messages)
    preset = _compose("+experiment=single_gpu_bf16_16gb_inference")
    assert not any("stays resident" in m for m in _warnings_during(lambda: validate_streaming_config(preset)))


# ---------------------------------------------------------------------------
# Host-staged KV pre-flight arithmetic (B-H1)
# ---------------------------------------------------------------------------
TOKENS_720P = 880  # 22 x 40 latent tokens per frame at 704x1280


def test_kv_estimates_match_the_documented_720p_numbers():
    from hymm.utils.streaming_config import (
        estimate_host_staged_kv_gpu_bytes,
        estimate_host_staged_kv_host_bytes,
        kv_layer_token_bytes,
    )

    assert kv_layer_token_bytes(16, 128) == 8192
    per_frame = estimate_host_staged_kv_gpu_bytes(
        history_tokens=TOKENS_720P, cache_heads=16, head_dim=128, slots=2, guidance_batch=1,
    )
    assert per_frame == 3 * 8192 * TOKENS_720P  # ~22 MB per latent frame at B=1
    with_cfg = estimate_host_staged_kv_gpu_bytes(
        history_tokens=TOKENS_720P, cache_heads=16, head_dim=128, slots=2, guidance_batch=2,
    )
    assert 28e6 < with_cfg < 30e6
    host = estimate_host_staged_kv_host_bytes(
        capacity_tokens=TOKENS_720P, num_layers=54, cache_heads=16, head_dim=128,
    )
    assert 388e6 < host < 390e6  # ~389 MB per latent frame


def test_max_history_tokens_is_the_inverse_of_the_estimate():
    from hymm.utils.streaming_config import estimate_host_staged_kv_gpu_bytes, max_history_tokens_for

    common = dict(cache_heads=16, head_dim=128, slots=2, guidance_batch=3, extra_tokens=17_856)
    budget = 9 * 2**30
    tokens = max_history_tokens_for(budget_bytes=budget, **common)
    assert estimate_host_staged_kv_gpu_bytes(history_tokens=tokens, **common) <= budget
    assert estimate_host_staged_kv_gpu_bytes(history_tokens=tokens + 1, **common) > budget
    assert max_history_tokens_for(budget_bytes=-1, **common) == 0


def _preflight_args(**overrides):
    args = dict(
        history_tokens=600 * TOKENS_720P, extra_tokens=20 * TOKENS_720P + 256,
        activation_tokens=20 * TOKENS_720P + 256, history_latent_frames=600,
        cache_heads=16, head_dim=128, slots=2, guidance_batch=1, num_layers=54,
        capacity_tokens=620 * TOKENS_720P + 256, resident_bytes=int(1.6 * 2**30), fps=24.0,
    )
    args.update(overrides)
    return args


def test_preflight_rejects_a_history_that_cannot_fit():
    from hymm.utils.streaming_config import KV_PREFLIGHT_ACTIVATION_BYTES_PER_TOKEN, host_staged_kv_preflight

    estimate = host_staged_kv_preflight(**_preflight_args())
    assert estimate["fits"] is None and estimate["free_bytes"] is None
    assert estimate["activation_bytes"] == KV_PREFLIGHT_ACTIVATION_BYTES_PER_TOKEN * (20 * TOKENS_720P + 256)
    assert estimate["predicted_bytes"] == (
        estimate["resident_bytes"] + estimate["kv_gpu_bytes"] + estimate["activation_bytes"]
    )
    free = 8 * 2**30
    with pytest.raises(ValueError) as error:
        host_staged_kv_preflight(**_preflight_args(free_bytes=free))
    message = str(error.value)
    assert "8.00 GiB is free" in message and "latent frames" in message and "fps" in message
    assert "kv_cache_host_staging_preflight=false" in message
    report = host_staged_kv_preflight(**_preflight_args(free_bytes=free), enforce=False)
    assert report["fits"] is False and 0 < report["max_latent_frames"] < 600
    assert report["max_seconds"] == pytest.approx(report["max_latent_frames"] * 4 / 24.0)
    assert f"At most {report['max_latent_frames']} latent frames" in message
    # The reported maximum really fits.
    frames = report["max_latent_frames"]
    shorter = host_staged_kv_preflight(**_preflight_args(
        free_bytes=free, history_tokens=frames * TOKENS_720P, history_latent_frames=frames,
    ))
    assert shorter["fits"]


def test_preflight_passes_with_enough_free_memory():
    from hymm.utils.streaming_config import host_staged_kv_preflight

    report = host_staged_kv_preflight(**_preflight_args(free_bytes=64 * 2**30))
    assert report["fits"] and report["max_latent_frames"] >= 600

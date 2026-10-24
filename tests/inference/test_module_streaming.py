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


class _FakeCudart:
    def __init__(self, error=0):
        self.error = error
        self.registered = {}
        self.unregistered = []

    def cudaHostRegister(self, ptr, size, flags):
        if self.error == 0:
            self.registered[ptr] = size
        return self.error

    def cudaHostUnregister(self, ptr):
        self.unregistered.append(ptr)
        return 0


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


def _cpu_staged_cache(num_blocks=5, slots=2, capacity=16):
    from hymm.models.dit.modules.host_staged_kv_cache import HostStagedKVCache

    class CPUStagedCache(HostStagedKVCache):
        def _device_empty(self, shape):
            return torch.zeros(shape, dtype=self.dtype)

    cache = CPUStagedCache(
        num_blocks=num_blocks, batch_size=1, max_cache_tokens=capacity, cache_heads=2,
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
    assert cache._staging_capacity == 12
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

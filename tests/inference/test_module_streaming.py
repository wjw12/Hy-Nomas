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

"""Layer-granular weight streaming for frozen BF16 inference on small GPUs.

A :class:`ModuleStreamer` keeps the master copy of a sequential stack of
"units" (one DiT double block plus its camera ActionModule, one PixelUMM
decoder layer, or one text-encoder decoder layer) in pinned host memory and
streams each unit into a small ring of fixed GPU slot buffers just before that
unit's forward runs. While unit ``i`` computes, unit ``i+1`` is already being
copied on a side CUDA stream, so with enough work per unit the PCIe transfer
hides behind compute and the GPU holds only ``slots`` units of weights at any
time instead of the whole model.

Nothing about the arithmetic changes: the same BF16 tensors reach the same
kernels, only their storage moves. The scheme relies on three properties that
the three Hy-Nomas stacks share:

* units are invoked sequentially through ``nn.Module.__call__`` so forward
  pre-hooks fire;
* no weight is shared between units;
* the compiled helpers (``nomas_compile_prepost``) take weights as explicit
  arguments and run without CUDA graphs, so rebinding ``param.data`` to a
  different storage does not invalidate anything.

The object duck-types :class:`hymm.utils.model_cpu_offload.ModelCPUOffload`
(``on_device()``, ``stage_prefix``, ``device``, transfer counters) so the
existing ``@cpu_offloaded`` decorator and the DiT residency context can drive
it without further changes.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from hymm.utils.inference_metrics import stage

_ALIGN = 256  # byte alignment of every tensor inside a flat unit buffer
_PAGE = 4096  # host page size; registered ranges never share a page


# ---------------------------------------------------------------------------
# Exact-size pinned host memory
# ---------------------------------------------------------------------------
_CUDA_ERROR_NOT_SUPPORTED = 801  # cudaErrorNotSupported (some WSL / vGPU / container setups)

# cudaHostRegister state shared by every allocation of the process. Once
# registration is known not to work (unsupported, or the very first attempt
# failed) the remaining few hundred allocations go straight to the fallback
# instead of each failing (and logging) again.
_HOST_REGISTER_DISABLED = False
_HOST_REGISTER_SUCCEEDED = False
_FALLBACK_WARNED = False


def _cuda_error_code(result) -> int:
    value = getattr(result, "value", result)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0 if result is None else 1


def _clear_pending_cuda_error(cudart=None) -> None:
    """Consume the CUDA error a failed runtime call left on this thread.

    PyTorch's ``cudart()`` binding returns the error code but does not reset
    the thread's last-error state, so without this the next unrelated kernel
    launch raises ``RuntimeError: CUDA error``. ``cudaGetLastError`` is used
    when the binding exposes it (current PyTorch builds do not); otherwise a
    trivial kernel is launched inside a ``try``: PyTorch's launch check reads,
    and thereby clears, the pending error and raises it, which is swallowed.
    """
    get_last_error = getattr(cudart, "cudaGetLastError", None) if cudart is not None else None
    if get_last_error is not None:
        try:
            get_last_error()
            return
        except Exception:  # noqa: BLE001 - fall through to the kernel probe
            pass
    try:
        torch.empty(1, device="cuda").zero_()
    except Exception:  # noqa: BLE001 - the probe exists to absorb this error
        pass


def _unregister_host(ptr: int, storage_keepalive) -> None:
    """Finalizer: unpin a registered range once its owning tensor is dropped.

    ``storage_keepalive`` (the untyped storage, not the tensor) keeps the
    memory allocated until the range has been unregistered.
    """
    cudart = None
    try:
        if torch.cuda.is_initialized():
            # A non_blocking copy may still be reading the range.
            torch.cuda.synchronize()
        cudart = torch.cuda.cudart()
        error = _cuda_error_code(cudart.cudaHostUnregister(ptr))
        if error != 0:
            _clear_pending_cuda_error(cudart)
            logger.debug(f"cudaHostUnregister failed with error {error}")
    except Exception:  # noqa: BLE001 - interpreter shutdown / driver teardown
        if cudart is not None:
            try:
                _clear_pending_cuda_error(cudart)
            except Exception:  # noqa: BLE001
                pass
    del storage_keepalive


def allocate_pinned_exact(
    shape: Union[int, Sequence[int]],
    dtype: torch.dtype = torch.uint8,
    *,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, bool]:
    """Allocate an exact-size host tensor and page-lock it in place.

    ``torch.empty(..., pin_memory=True)`` goes through PyTorch's caching host
    allocator, which rounds every block up to the next power of two (a 311 MB
    DiT unit costs 512 MiB, a 567 MB KV layer 1 GiB). Here the memory is a
    pageable allocation of the requested size (plus at most two pages for
    alignment) registered with ``cudaHostRegister``; a finalizer unregisters
    it when the returned tensor is dropped. Registered memory reports
    ``is_pinned() == True`` and makes ``non_blocking`` copies asynchronous.

    Falls back to ``pin_memory=True`` and then to pageable memory (with a
    warning) if registration fails. The pending CUDA error of a failed
    register / unregister call is consumed so later kernels are unaffected.
    After an "unsupported" result, or when the first registration of the
    process fails, registration is skipped for every later allocation.
    Without CUDA the result is pageable and nothing is logged. Returns
    ``(tensor, pinned)``.
    """
    global _HOST_REGISTER_DISABLED, _HOST_REGISTER_SUCCEEDED, _FALLBACK_WARNED

    shape = [int(shape)] if isinstance(shape, int) else [int(s) for s in shape]
    numel = 1
    for size in shape:
        numel *= size
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    if not pin_memory or nbytes == 0 or not torch.cuda.is_available():
        return torch.empty(shape, dtype=dtype, device="cpu"), False

    padded = (nbytes + _PAGE - 1) // _PAGE * _PAGE
    if not _HOST_REGISTER_DISABLED:
        base = None
        try:
            base = torch.empty(padded + _PAGE, dtype=torch.uint8, device="cpu")
        except RuntimeError as exc:
            # Not a registration failure: the host is out of pageable memory.
            logger.warning(f"Host allocation of {padded} bytes failed: {exc}")
        if base is not None:
            offset = (-base.data_ptr()) % _PAGE
            region = base[offset : offset + padded]
            ptr = region.data_ptr()
            cudart = None
            try:
                cudart = torch.cuda.cudart()
                error = _cuda_error_code(cudart.cudaHostRegister(ptr, padded, 0))
                failure = f"error {error}"
            except Exception as exc:  # noqa: BLE001
                error, failure = -1, str(exc)
            if error == 0:
                tensor = region[:nbytes].view(dtype).view(shape)
                finalizer = weakref.finalize(tensor, _unregister_host, ptr, region.untyped_storage())
                # Process exit releases the pages anyway; do not call into a CUDA
                # context that may already be torn down.
                finalizer.atexit = False
                _HOST_REGISTER_SUCCEEDED = True
                return tensor, True
            _clear_pending_cuda_error(cudart)
            # Drop the unregistered buffer before the fallback allocates, so
            # the host footprint never holds two copies of the range.
            base = region = None
            logger.warning(f"cudaHostRegister({padded} bytes) failed: {failure}")
            if error == _CUDA_ERROR_NOT_SUPPORTED or not _HOST_REGISTER_SUCCEEDED:
                _HOST_REGISTER_DISABLED = True
                logger.warning(
                    "cudaHostRegister is disabled for the remaining host allocations "
                    "of this process; using the fallback allocator"
                )
    try:
        tensor = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        message = (
            "Falling back to torch pin_memory=True; the caching host allocator "
            f"rounds {nbytes} bytes up to the next power of two"
        )
        if _FALLBACK_WARNED:
            logger.debug(message)
        else:
            logger.warning(message)
            _FALLBACK_WARNED = True
        return tensor, True
    except RuntimeError as exc:
        logger.warning(f"Pinned host allocation failed ({exc}); using pageable memory")
    return torch.empty(shape, dtype=dtype, device="cpu"), False


# ---------------------------------------------------------------------------
# Pure-Python ring scheduler (unit-testable without CUDA)
# ---------------------------------------------------------------------------
class RingScheduler:
    """Decide which slot serves which unit and what to prefetch next.

    ``sequence`` is the ordered list of streamed unit indices for one forward
    pass (resident units are excluded). Arrivals are counted with a monotonic
    counter ``c``; the ``c``-th arrival is served from slot ``c % slots``.
    When an arrival starts, the slot used by the previous arrival is free (the
    previous unit's work is already enqueued ahead on the compute stream), so
    the unit ``slots - 1`` positions ahead in the sequence is prefetched into
    slot ``(c + slots - 1) % slots``. Positions wrap around the sequence so
    the next forward pass finds its first units already in flight, and the
    counter keeps slot assignment consistent whatever ``len(sequence)`` is.
    """

    def __init__(self, sequence: Sequence[int], slots: int):
        if slots < 2:
            raise ValueError("streaming needs at least 2 slots (one computing, one in flight)")
        if not sequence:
            raise ValueError("streaming needs at least one streamed unit")
        self.sequence = list(sequence)
        self.slots = int(slots)
        self.position_of = {unit: pos for pos, unit in enumerate(self.sequence)}
        self.next_position = 0
        self.counter = 0

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def prefetch_depth(self) -> int:
        return min(self.slots - 1, len(self.sequence))

    @property
    def all_fit(self) -> bool:
        """Every streamed unit owns a slot; nothing is released or prefetched."""
        return len(self.sequence) <= self.slots - 1

    def initial_prefetch(self) -> List[Tuple[int, int]]:
        """``(unit, slot)`` pairs to load before the first forward."""
        self.counter = 0
        return self.resync_prefetch(self.sequence[0])

    def resync_prefetch(self, unit: int) -> List[Tuple[int, int]]:
        """Loads needed to (re)start the ring at ``unit`` (``unit`` itself first).

        The next arrival of ``unit`` is served from slot ``counter % slots``.
        """
        position = self.position_of[unit]
        n = len(self.sequence)
        self.next_position = position
        if n <= self.slots - 1:
            # Everything fits in the ring: each unit owns the slot of its position.
            return [(self.sequence[k], k) for k in range(n)]
        return [
            (self.sequence[(position + k) % n], (self.counter + k) % self.slots)
            for k in range(self.prefetch_depth)
        ]

    def arrive(self, unit: int) -> Tuple[int, Optional[int], Optional[Tuple[int, int]], bool]:
        """Register that ``unit`` is about to run.

        Returns ``(slot, released_unit, prefetch, in_order)``:

        * ``slot`` must hold ``unit``;
        * ``released_unit`` is the unit whose slot just became free (the
          previous position; ``None`` when the ring has a single unit);
        * ``prefetch`` is the ``(unit, slot)`` to start copying now, or
          ``None`` when every streamed unit already fits in the ring;
        * ``in_order`` is ``False`` when the arrival did not match the
          expected position. The caller must then resynchronize: drain
          outstanding copies, call :meth:`resync_prefetch`, and arrive again.
        """
        if unit not in self.position_of:
            raise KeyError(f"unit {unit} is not streamed")
        position = self.position_of[unit]
        in_order = position == self.next_position
        n = len(self.sequence)
        if n <= self.slots - 1:
            # Every streamed unit is resident in its own slot: nothing is ever
            # released or prefetched after the initial load.
            self.next_position = (position + 1) % n
            return position, None, None, in_order
        if not in_order:
            return self.counter % self.slots, None, None, False
        slot = self.counter % self.slots
        released = self.sequence[(position - 1) % n] if n > 1 else None
        ahead = self.slots - 1
        prefetch = (self.sequence[(position + ahead) % n], (self.counter + ahead) % self.slots)
        self.counter += 1
        self.next_position = (position + 1) % n
        return slot, released, prefetch, True


# ---------------------------------------------------------------------------
# Flat packing of a group's parameters and buffers
# ---------------------------------------------------------------------------
def _round_up(value: int, alignment: int = _ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def direct_tensors(module: nn.Module) -> List[torch.Tensor]:
    """A module's own parameters and buffers (children excluded)."""
    own = [p for p in module._parameters.values() if p is not None]
    own += [b for b in module._buffers.values() if b is not None]
    return own


def all_tensors(modules: Iterable[nn.Module]) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    seen = set()
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                result.append(tensor)
    return result


class _Packed:
    """A set of tensors laid out contiguously in one flat uint8 buffer."""

    def __init__(self, tensors: Sequence[torch.Tensor]):
        self.entries: List[Tuple[torch.Tensor, int, int]] = []  # (tensor, offset, nbytes)
        offset = 0
        for tensor in tensors:
            nbytes = tensor.numel() * tensor.element_size()
            self.entries.append((tensor, offset, nbytes))
            offset += _round_up(nbytes)
        self.nbytes = offset
        self.host: Optional[torch.Tensor] = None
        self.host_views: List[torch.Tensor] = []
        self.pinned = False

    def pack_to_host(self, pin_memory: bool) -> None:
        """Copy every tensor into one flat host buffer and rebind ``.data`` to it.

        The original per-tensor CPU storage is released once no reference to
        it remains, so the transient cost is one flat buffer, not two models.
        """
        host, self.pinned = allocate_pinned_exact(
            self.nbytes, torch.uint8, pin_memory=pin_memory,
        )
        for tensor, offset, nbytes in self.entries:
            view = host[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape)
            view.copy_(tensor.detach().to("cpu"))
            tensor.data = view
        self.host = host
        # Precomputed so a unit released mid-phase can be rebound cheaply.
        self.host_views = self.views_into(host)

    def views_into(self, flat: torch.Tensor) -> List[torch.Tensor]:
        return [
            flat[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape)
            for tensor, offset, nbytes in self.entries
        ]

    def bind(self, views: Sequence[torch.Tensor]) -> None:
        for (tensor, _, _), view in zip(self.entries, views):
            tensor.data = view

    def bind_host(self) -> None:
        assert self.host is not None
        self.bind(self.host_views or self.views_into(self.host))


def _check_frozen(modules: Iterable[nn.Module], tensors: Iterable[torch.Tensor]) -> None:
    for module in modules:
        if module.training:
            raise ValueError("Module streaming requires eval() modules")
        for child in module.modules():
            if getattr(child, "_orig_mod", None) is not None:
                raise ValueError("Module streaming does not support torch.compile'd modules")
    for tensor in tensors:
        if tensor.requires_grad:
            raise ValueError("Module streaming requires frozen weights (requires_grad=False)")
        if tensor.is_meta or hasattr(tensor, "device_mesh"):
            raise ValueError("Module streaming requires materialized, unsharded tensors")
        if tensor.device.type != "cpu":
            raise ValueError(
                "Module streaming expects the model on CPU before installation; "
                f"found a tensor on {tensor.device}"
            )


# ---------------------------------------------------------------------------
# The streamer
# ---------------------------------------------------------------------------
class ModuleStreamer:
    """Stream ``units`` through ``slots`` GPU buffers; keep ``resident`` on GPU per phase.

    Parameters
    ----------
    units:
        Ordered list of module groups. Each group is loaded and bound as one
        transfer. The first module of a group receives the forward pre-hook,
        so it must be the module that is called first in the forward pass.
    resident:
        Modules whose *own* tensors must be on the GPU for the whole phase but
        are not part of the streamed sequence (embedders, final layers,
        norms). They are uploaded on ``enter()`` and returned to pinned host
        on ``exit()``. Use :func:`modules_outside` to collect them.
    device:
        CUDA execution device.
    slots:
        Number of GPU slot buffers; ``slots - 1`` units are prefetched ahead.
    resident_units:
        Keep the first ``resident_units`` units on the GPU for the whole phase
        (spends spare VRAM to cut PCIe traffic).
    stage_prefix:
        Metrics prefix: ``"dit"``, ``"te"`` or ``"pixelumm"``.
    """

    def __init__(
        self,
        units: Sequence[Sequence[nn.Module]],
        *,
        resident: Sequence[nn.Module] = (),
        device: torch.device,
        slots: int = 2,
        resident_units: int = 0,
        stage_prefix: str = "dit",
        pin_memory: bool = True,
        name: str = "",
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Module streaming requires a CUDA execution device")
        if stage_prefix not in {"te", "pixelumm", "dit"}:
            raise ValueError("Unknown streaming metric prefix")
        if not units:
            raise ValueError("Module streaming needs at least one unit")
        self.stage_prefix = stage_prefix
        self.name = name or stage_prefix
        self.slots = int(slots)
        self.resident_units = max(0, min(int(resident_units), len(units)))
        if self.slots < 2 and self.resident_units < len(units):
            raise ValueError("streaming needs at least 2 slots")

        unit_modules = [m for group in units for m in group]
        unit_tensors = [all_tensors(group) for group in units]
        # Rebinding .data of a tensor shared by two units (or by a unit and the
        # resident set) would silently point one of them at the other's slot,
        # so sharing is rejected instead of deduplicated.
        owner: Dict[int, int] = {}
        for unit_index, group in enumerate(unit_tensors):
            for tensor in group:
                previous = owner.setdefault(id(tensor), unit_index)
                if previous != unit_index:
                    raise ValueError(
                        f"Module streaming: units {previous} and {unit_index} share a tensor "
                        f"of shape {tuple(tensor.shape)}; streamed units must not share weights"
                    )
        resident_tensors: List[torch.Tensor] = []
        seen = set()
        for module in resident:
            for tensor in direct_tensors(module):
                if id(tensor) in owner:
                    raise ValueError(
                        f"Module streaming: resident module {type(module).__name__} shares a "
                        f"tensor with streamed unit {owner[id(tensor)]}"
                    )
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    resident_tensors.append(tensor)
        _check_frozen(unit_modules + list(resident), [t for g in unit_tensors for t in g] + resident_tensors)

        self.unit_modules = [list(group) for group in units]
        self.packed = [_Packed(tensors) for tensors in unit_tensors]
        self.resident_packed = _Packed(resident_tensors) if resident_tensors else None
        self.unit_bytes = max(p.nbytes for p in self.packed)
        self.total_bytes = sum(p.nbytes for p in self.packed) + (
            self.resident_packed.nbytes if self.resident_packed else 0
        )
        started = time.perf_counter()
        for packed in self.packed:
            packed.pack_to_host(pin_memory)
        if self.resident_packed is not None:
            self.resident_packed.pack_to_host(pin_memory)
        self.pack_seconds = time.perf_counter() - started
        self.pinned = all(p.pinned for p in self.packed) and (
            self.resident_packed is None or self.resident_packed.pinned
        )

        streamed = list(range(self.resident_units, len(self.packed)))
        self.scheduler = RingScheduler(streamed, self.slots) if streamed else None

        # Device-side state, allocated per phase.
        self._slot_buffers: List[torch.Tensor] = []
        self._slot_views: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._slot_occupant: List[Optional[int]] = []
        self._resident_buffers: Dict[int, torch.Tensor] = {}
        self._resident_module_buffer: Optional[torch.Tensor] = None
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._ready_events: Dict[int, torch.cuda.Event] = {}
        self._free_events: Dict[int, torch.cuda.Event] = {}
        self._bound_unit_slot: Dict[int, int] = {}
        self.depth = 0

        # Counters compatible with cpu_offload_snapshot.
        self.completed_calls = 0
        self.gpu_transfers = 0
        self.cpu_transfers = 0
        self.transfer_seconds = 0.0
        self.unit_loads = 0
        self.bytes_streamed = 0
        self.resyncs = 0
        self.phases = 0

        self._hooks = []
        for unit_index, group in enumerate(self.unit_modules):
            self._hooks.append(
                group[0].register_forward_pre_hook(self._make_pre_hook(unit_index))
            )
        for module in unit_modules + list(resident):
            module._module_streamer = self

    # ------------------------------------------------------------------
    # phase management
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.depth > 0

    def enter(self) -> None:
        if self.depth > 0:
            self.depth += 1
            return
        try:
            self._enter_phase()
        except BaseException:
            # An OOM (or any error) part-way through must not leave a
            # half-bound model or a depth that never returns to zero: rebind
            # everything to the host copies and drop the device buffers.
            self.depth = 1
            try:
                self.exit()
            except BaseException:  # noqa: BLE001 - keep the original error
                pass
            raise
        self.depth = 1

    def _enter_phase(self) -> None:
        started = time.perf_counter()
        with stage(self.stage_prefix + "_H2D"), torch.inference_mode(False), torch.no_grad():
            self._copy_stream = torch.cuda.Stream(device=self.device)
            n_slots = self._phase_slot_count()
            self._slot_buffers = [
                torch.empty(self.unit_bytes, dtype=torch.uint8, device=self.device)
                for _ in range(n_slots)
            ]
            self._slot_occupant = [None] * n_slots
            self._slot_views = {}
            self._free_events = {}
            self._ready_events = {}
            self._bound_unit_slot = {}
            if self.resident_packed is not None and self.resident_packed.nbytes:
                buffer = torch.empty(
                    self.resident_packed.nbytes, dtype=torch.uint8, device=self.device
                )
                buffer.copy_(self.resident_packed.host, non_blocking=False)
                self.resident_packed.bind(self.resident_packed.views_into(buffer))
                self._resident_module_buffer = buffer
                self.bytes_streamed += self.resident_packed.nbytes
            for unit in range(self.resident_units):
                packed = self.packed[unit]
                buffer = torch.empty(packed.nbytes, dtype=torch.uint8, device=self.device)
                buffer.copy_(packed.host, non_blocking=False)
                packed.bind(packed.views_into(buffer))
                self._resident_buffers[unit] = buffer
                self.bytes_streamed += packed.nbytes
            torch.cuda.synchronize(self.device)
            if self.scheduler is not None:
                for unit, slot in self.scheduler.initial_prefetch():
                    self._issue_load(unit, slot)
        self.transfer_seconds += time.perf_counter() - started
        self.gpu_transfers += 1
        self.phases += 1

    def _phase_slot_count(self) -> int:
        """GPU slot buffers a phase allocates.

        When every streamed unit owns a slot (``n <= slots - 1``) slots are
        indexed by position, so only ``n`` of them are ever used.
        """
        if self.scheduler is None:
            return 0
        return len(self.scheduler) if self.scheduler.all_fit else self.slots

    def phase_device_bytes(self) -> int:
        """Weight bytes a phase holds on the GPU (slot ring + resident units/modules).

        Valid before ``enter()``, so callers can budget GPU memory ahead of the phase.
        """
        total = self._phase_slot_count() * self.unit_bytes
        total += sum(self.packed[unit].nbytes for unit in range(self.resident_units))
        if self.resident_packed is not None:
            total += self.resident_packed.nbytes
        return total

    def _bind_host(self, unit: int) -> None:
        """Point a unit that left its slot back at its host copy.

        A later access outside the hooked ``__call__`` then fails with a device
        mismatch instead of silently computing with whatever unit the slot is
        being refilled with.
        """
        with torch.inference_mode(False), torch.no_grad():
            packed = self.packed[unit]
            packed.bind(packed.host_views)

    def exit(self) -> None:
        if self.depth == 0:
            return
        if self.depth > 1:
            self.depth -= 1
            return
        started = time.perf_counter()
        try:
            torch.cuda.synchronize(self.device)
        finally:
            with stage(self.stage_prefix + "_D2H"), torch.inference_mode(False), torch.no_grad():
                for packed in self.packed:
                    packed.bind_host()
                if self.resident_packed is not None:
                    self.resident_packed.bind_host()
                self._slot_buffers = []
                self._slot_views = {}
                self._slot_occupant = []
                self._resident_buffers = {}
                self._resident_module_buffer = None
                self._ready_events = {}
                self._free_events = {}
                self._bound_unit_slot = {}
                self._copy_stream = None
            self.depth = 0
            self.completed_calls += 1
            self.cpu_transfers += 1
            self.transfer_seconds += time.perf_counter() - started

    @contextmanager
    def on_device(self):
        """Phase scope; nests like ``ModelCPUOffload.on_device``."""
        if torch.is_grad_enabled():
            raise RuntimeError("Module streaming supports frozen no-grad inference only")
        self.enter()
        try:
            yield
        finally:
            self.exit()

    # ------------------------------------------------------------------
    # transfers
    # ------------------------------------------------------------------
    def _views(self, unit: int, slot: int) -> List[torch.Tensor]:
        key = (unit, slot)
        views = self._slot_views.get(key)
        if views is None:
            with torch.inference_mode(False), torch.no_grad():
                views = self.packed[unit].views_into(self._slot_buffers[slot])
            self._slot_views[key] = views
        return views

    def _issue_load(self, unit: int, slot: int) -> None:
        """Copy ``unit`` into ``slot`` on the copy stream once the slot is free."""
        packed = self.packed[unit]
        stream = self._copy_stream
        free_event = self._free_events.pop(slot, None)
        with torch.cuda.stream(stream):
            if free_event is not None:
                stream.wait_event(free_event)
            self._slot_buffers[slot][: packed.nbytes].copy_(
                packed.host, non_blocking=packed.pinned
            )
            ready = torch.cuda.Event()
            ready.record(stream)
        self._ready_events[unit] = ready
        self._slot_occupant[slot] = unit
        self.unit_loads += 1
        self.bytes_streamed += packed.nbytes

    def _make_pre_hook(self, unit: int):
        def hook(module, args):
            self._before_unit(unit)
            return None

        return hook

    def _before_unit(self, unit: int) -> None:
        if self.depth == 0:
            raise RuntimeError(
                f"{self.name}: streamed module ran outside an on_device() phase; "
                "wrap the call (text encode / DiT residency / decode) in the phase scope"
            )
        if unit < self.resident_units or self.scheduler is None:
            return  # resident for this phase
        compute = torch.cuda.current_stream(self.device)
        slot, released, prefetch, in_order = self.scheduler.arrive(unit)
        # When every unit owns its slot, order does not matter: only the
        # occupant decides whether a reload is needed.
        if (not in_order and not self.scheduler.all_fit) or self._slot_occupant[slot] != unit:
            # A unit was skipped or repeated (exception mid-forward, partial
            # forward, a model path that visits blocks differently). Drain the
            # copy stream and restart the ring from here.
            self.resyncs += 1
            torch.cuda.synchronize(self.device)
            for bound_unit in list(self._bound_unit_slot):
                self._bind_host(bound_unit)
            self._free_events = {}
            self._ready_events = {}
            self._bound_unit_slot = {}
            self._slot_occupant = [None] * len(self._slot_buffers)
            for load_unit, load_slot in self.scheduler.resync_prefetch(unit):
                self._issue_load(load_unit, load_slot)
            slot, released, prefetch, _ = self.scheduler.arrive(unit)
        ready = self._ready_events.get(unit)
        if ready is not None:
            compute.wait_event(ready)
        self.packed[unit].bind(self._views(unit, slot))
        self._bound_unit_slot[unit] = slot
        if released is not None and released != unit:
            released_slot = self._bound_unit_slot.pop(released, None)
            if released_slot is not None:
                event = torch.cuda.Event()
                event.record(compute)
                self._free_events[released_slot] = event
                self._bind_host(released)
        if prefetch is not None:
            prefetch_unit, prefetch_slot = prefetch
            if self._slot_occupant[prefetch_slot] != prefetch_unit:
                self._issue_load(prefetch_unit, prefetch_slot)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "policy": "layer_streaming",
            "units": len(self.packed),
            "resident_units": self.resident_units,
            "slots": self.slots,
            "unit_bytes_max": self.unit_bytes,
            "total_weight_bytes": self.total_bytes,
            "pinned": self.pinned,
            "pack_seconds": self.pack_seconds,
            "phases": self.phases,
            "unit_loads": self.unit_loads,
            "bytes_streamed": self.bytes_streamed,
            "resyncs": self.resyncs,
            "completed_calls": self.completed_calls,
            "gpu_transfers": self.gpu_transfers,
            "cpu_transfers": self.cpu_transfers,
            "transfer_seconds": self.transfer_seconds,
            "depth": self.depth,
        }

    def device_bytes(self) -> int:
        """Bytes of weight storage currently held on the GPU."""
        total = sum(b.numel() for b in self._slot_buffers)
        total += sum(b.numel() for b in self._resident_buffers.values())
        if self._resident_module_buffer is not None:
            total += self._resident_module_buffer.numel()
        return total


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def modules_outside(root: nn.Module, groups: Sequence[Sequence[nn.Module]]) -> List[nn.Module]:
    """Modules of ``root`` that own tensors not covered by any streamed group.

    The result lists every module (including ``root`` itself) that has at
    least one *direct* parameter or buffer and is not a streamed module or
    one of its descendants. Pass it as ``resident`` so the streamer keeps
    those tensors on the GPU during a phase.
    """
    excluded = set()
    for group in groups:
        for module in group:
            for child in module.modules():
                excluded.add(id(child))
    result = []
    for module in root.modules():
        if id(module) in excluded:
            continue
        if direct_tensors(module):
            result.append(module)
    return result


def install_module_streaming(
    root: nn.Module,
    units: Sequence[Sequence[nn.Module]],
    *,
    device: torch.device,
    slots: int,
    resident_units: int = 0,
    stage_prefix: str,
    name: str = "",
) -> ModuleStreamer:
    """Build a streamer for ``units`` and keep everything else in ``root`` resident per phase."""
    if getattr(root, "_model_cpu_offload", None) is not None:
        raise ValueError("Module streaming cannot be combined with whole-model CPU offload")
    resident = modules_outside(root, units)
    streamer = ModuleStreamer(
        units,
        resident=resident,
        device=device,
        slots=slots,
        resident_units=resident_units,
        stage_prefix=stage_prefix,
        name=name,
    )
    # Reuse the controller slot so @cpu_offloaded, residency scopes and the
    # offload snapshot drive the streamer transparently.
    root._model_cpu_offload = streamer
    root._module_streamer = streamer
    return streamer

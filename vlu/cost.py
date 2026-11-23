"""The efficiency of an op: latency, compute, peak GPU memory and GPU energy of its phases (PLAN.md: the
cost of every unlearning op). vlu/run_op.py measures every op; cost.json keeps the result, the reports show it.

    meter = Meter()                          # picks this GPU's energy source (below)
    with meter.active(), meter.phase("train"):
        ...
        with phase("probe"):                 # anywhere: a phase of the active meter (no meter: nothing)
            ...
        with flops("train"):                 # a repeated unit of work: its FLOPs are counted once
            ...                              # (its first call), times the number of calls
        step_done(start, "train", frames, waited)   # a training step: began at time.monotonic() == start

A phase records its wall time (monotonic clock; CUDA synchronized at both ends, so queued GPU work is
counted where it runs), the peak GPU memory PyTorch allocated and reserved within it (an outer phase's
peak includes its inner phases'; the CUDA context, ~0.5 GiB, comes on top), and its GPU energy. Phases
of an op (vlu/run_op.py): load (not part of the op's cost), prepare (SSD: all of it; SalUn: its mask; NPO:
the reference copy), train (VLA-Forget: also its boundary set, layer selection and merge), and inside
these probe, checkpoint and save: our instrumentation, not the method. The method's cost = prepare +
train - probe - checkpoint - save.

Steps (per group: "train", or VLA-Forget's "train:<stage>"): each step's time once its batch has arrived
(CUDA synchronized at its end), the time it waited for its batch (the DataLoader: a CPU bottleneck shows
here, not in the step), and its frames. The first WARMUP_STEPS of each group and session are left out
(first kernel launches, a fresh DataLoader), and so is the step whose FLOPs were counted (the counter's
own overhead).

Compute: torch's FLOP counter (torch.utils.flop_counter) on the first call of each repeated unit (a
training step; an SSD / SalUn importance batch; a VLA-Forget selection batch or boundary chunk), times the
number of calls. It counts matrix multiplications, convolutions and PyTorch's own attention, forward and
backward; not elementwise work (optimizer, norms, activations) and not flash-attn's kernels (GR00T's LLM
attention). So it is a lower bound of the true FLOPs, the same model's for every method: comparable across
methods and GPUs. Achieved TFLOP/s = a step's FLOPs / its median time.

Idle: after loading, before the op's work, IDLE_SECONDS with nothing running give the GPU's idle power
(a board just used for loading may not have reached its lowest power state: the idle power is then too
high, and the energy above idle a lower bound). Energy above idle = energy - idle power x time: the
part the method's work adds.
Other processes on the GPU (NVML's list, when it can be read) are counted: more than this one means the
energy includes their work (a shared GPU), and the report flags it.

Energy, by what the GPU allows. Every number carries its source:
  zeus         ML.energy's Zeus (ZeusMonitor), which reads the GPU's NVML total-energy counter: a full GPU
  nvml         the same counter read directly (Zeus not installed, or failing)
  board        a MIG slice. Zeus refuses MIG devices (NVML has no per-slice power or energy), so: the board's
               counter x the slice's share of the board. A slice with more than 7/8 of the board's memory
               is the board's only slice (the smallest MIG slice has 1/8 of it): share 1, exact up to the
               board's idle power, which it includes (the cluster's 4g.96gb is the whole board). Otherwise
               share = the slice's memory share: an estimate, as the counter includes the other slices' work.
  power_limit  no counter: the board's power limit (NVML's enforced limit, else TDP_WATTS) x time x share,
               a pseudo-estimate and an upper bound (the board at its limit the whole time)
  None         neither NVML nor a known GPU: no energy
The CPU's and DRAM's energy are not measured.
"""

import contextlib
import os
import statistics
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# Board power by GPU name (vendor specifications), for the power_limit estimate when NVML cannot tell; the
# first entry whose text is in the name wins.
TDP_WATTS = (("Max-Q", 300.0), ("RTX PRO 6000 Blackwell", 600.0), ("RTX A6000", 300.0), ("RTX A5000", 230.0))
# Training steps of each session left out of the step latency: CUDA kernels' first launches, the allocator's
# growth, the DataLoader's first batches.
WARMUP_STEPS = 5
# Phases inside prepare / train that are not the method's work.
INSTRUMENTATION = ("probe", "checkpoint", "save")
IDLE_SECONDS = 5.0


# --- Energy sources ------------------------------------------------------------------------------------


class NoEnergy:
    name, share, watts = None, 1.0, None
    board = None  # (NVML module, board handle) when NVML knows this GPU: other processes, power limit
    limit_watts = None  # the board's power limit (NVML's enforced limit, else TDP_WATTS), for the report's check

    def __init__(self, note: str = ""):
        self.note = note

    def begin(self, key: str):
        return None

    def end(self, key: str, start, seconds: float) -> float | None:
        return None


class ZeusEnergy(NoEnergy):
    """ML.energy's Zeus on one GPU (its index as PyTorch sees it, after CUDA_VISIBLE_DEVICES)."""

    name = "zeus"

    def __init__(self, index: int, note: str = ""):
        from zeus.monitor import ZeusMonitor

        super().__init__(note)
        self.index = index
        # No CPU (RAPL) measurement: it needs root, and the GPU is what the methods differ in.
        self.monitor = ZeusMonitor(gpu_indices=[index], cpu_indices=[], sync_execution_with="torch")

    def begin(self, key):
        self.monitor.begin_window(key, sync_execution=False)  # the meter synchronizes CUDA itself

    def end(self, key, start, seconds):
        return float(self.monitor.end_window(key, sync_execution=False).gpu_energy[self.index])


class CounterEnergy(NoEnergy):
    """NVML's total-energy counter (mJ since the driver loaded) of a board, times the slice's share."""

    def __init__(self, nvml, handle, name: str, share: float, note: str = ""):
        super().__init__(note)
        self.nvml, self.handle, self.name, self.share = nvml, handle, name, share

    def begin(self, key):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def end(self, key, start, seconds):
        return (self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - start) / 1000 * self.share


class PowerLimitEnergy(NoEnergy):
    name = "power_limit"

    def __init__(self, watts: float, share: float, note: str = ""):
        super().__init__(note)
        self.watts, self.share = watts, share

    def end(self, key, start, seconds):
        return self.watts * seconds * self.share


def tdp_watts(gpu_name: str) -> float | None:
    return next((watts for text, watts in TDP_WATTS if text in gpu_name), None)


def _find_board(nvml, uuid: str):
    """(board handle, is a MIG slice) of the CUDA device with this UUID, or (None, False)."""
    candidates = [f"GPU-{uuid}", f"MIG-{uuid}"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.startswith(("GPU-", "MIG-")):  # Slurm may name the device by its UUID
        candidates.append(visible)
    for candidate in candidates:
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(candidate)
        except nvml.NVMLError:
            continue
        if candidate.startswith("MIG-"):
            return nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle), True
        return handle, False
    return None, False


def energy_source(index: int) -> NoEnergy:
    """The best energy source for CUDA device `index` (module docstring), with the board and its power limit."""
    props = torch.cuda.get_device_properties(index)
    try:
        import pynvml as nvml

        nvml.nvmlInit()
    except Exception as e:  # no bindings, or no driver library
        return _without_counter(props.name, 1.0, [f"no NVML ({type(e).__name__}: {e})"])
    board, mig = _find_board(nvml, str(props.uuid))
    if board is None:
        return _without_counter(props.name, 1.0, [f"GPU {props.uuid} not found in NVML"])
    try:
        limit, limit_from = nvml.nvmlDeviceGetEnforcedPowerLimit(board) / 1000, "NVML's enforced power limit"
    except nvml.NVMLError:
        limit, limit_from = tdp_watts(props.name), "the TDP of vlu/cost.py's TDP_WATTS"
    notes, share = [], 1.0
    if mig:
        try:
            ratio = props.total_memory / nvml.nvmlDeviceGetMemoryInfo(board).total
            share = 1.0 if ratio > 7 / 8 else ratio
            notes.append(f"MIG slice with {ratio:.3f} of the board's memory: share {share:.3f}")
        except nvml.NVMLError as e:
            notes.append(f"MIG slice, board memory unknown ({e}): share 1 assumed (an upper bound)")
    source = None
    if not mig:
        try:
            source = ZeusEnergy(index)
        except Exception as e:
            notes.append(f"Zeus not used ({type(e).__name__}: {e})")
    if source is None:
        try:
            nvml.nvmlDeviceGetTotalEnergyConsumption(board)
            source = CounterEnergy(nvml, board, "board" if mig else "nvml", share)
        except nvml.NVMLError as e:
            notes.append(f"no energy counter ({e})")
    if source is None:
        if limit:
            source = PowerLimitEnergy(limit, share)
            notes.append(f"{limit:.0f} W = {limit_from}")
        else:
            source = NoEnergy()
            notes.append("no power limit known")
    source.note = "; ".join(notes)
    source.board, source.limit_watts = (nvml, board), limit
    return source


def _without_counter(gpu_name: str, share: float, notes: list[str]) -> NoEnergy:
    """No NVML board: the TDP pseudo-estimate if the GPU is known, else no energy."""
    watts = tdp_watts(gpu_name)
    source = PowerLimitEnergy(watts, share) if watts else NoEnergy()
    source.note = "; ".join([*notes, f"{watts:.0f} W = the TDP of vlu/cost.py's TDP_WATTS" if watts else "no power limit known"])
    source.limit_watts = watts
    return source


class FlopCount(TorchDispatchMode):
    """FLOPs of the ops run inside it, forward and backward, by torch's own FLOP formulas
    (torch.utils.flop_counter.flop_registry: matrix multiplications, convolutions, PyTorch attention). Not
    torch's FlopCounterMode: its per-module tracking puts gradient hooks on every module's inputs and
    outputs, and a hook that a backward never reaches (VLA-Forget's torch.autograd.grad on its adapters
    only) keeps that forward's whole graph alive."""

    def __init__(self):
        super().__init__()
        from torch.utils.flop_counter import flop_registry

        self.registry, self.total = flop_registry, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        formula = self.registry.get(func._overloadpacket)
        if formula is not None:
            self.total += formula(*args, **kwargs, out_val=out)
        return out


# --- The meter -------------------------------------------------------------------------------------------


_ACTIVE: list["Meter"] = []


class Meter:
    def __init__(self, source: NoEnergy | None = None, device: int | None = None):
        self.device = torch.cuda.current_device() if device is None else device
        self.source = energy_source(self.device) if source is None else source
        self.phases: dict[str, dict] = {}  # name -> seconds, energy_j, peaks, count (summed over its occurrences)
        self.steps: dict[str, dict] = {}  # group -> seconds, waits (lists), frames per step
        self.flops: dict[str, dict] = {}  # unit -> per_call (FLOPs of its first call), calls
        self.idle_watts: float | None = None
        self.max_processes: int | None = None  # the most processes seen on the GPU (this one included)
        self._open: list[dict] = []
        self._windows = 0
        self._counted: str | None = None  # the unit whose FLOPs were just counted (its step is not timed)

    @contextlib.contextmanager
    def active(self):
        """Make this the meter that phase() and step_done() report to."""
        _ACTIVE.append(self)
        try:
            yield self
        finally:
            _ACTIVE.remove(self)

    def _fold_peaks(self) -> None:
        """Credit the peak memory since the last reset to every open phase (before the next reset)."""
        allocated = torch.cuda.max_memory_allocated(self.device) / 2**30
        reserved = torch.cuda.max_memory_reserved(self.device) / 2**30
        for record in self._open:
            record["allocated"] = max(record["allocated"], allocated)
            record["reserved"] = max(record["reserved"], reserved)

    def peak_allocated_gb(self) -> float:
        """The peak allocated memory of the outermost open phase so far."""
        self._fold_peaks()
        return self._open[0]["allocated"] if self._open else torch.cuda.max_memory_allocated(self.device) / 2**30

    @contextlib.contextmanager
    def phase(self, name: str):
        torch.cuda.synchronize(self.device)
        self._fold_peaks()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._windows += 1
        key = f"{name}#{self._windows}"  # Zeus: window names must be unique
        try:
            begun, energy_start = True, self.source.begin(key)
        except Exception as e:  # a failing reading must not fail the op
            begun, energy_start = False, None
            self.source.note += f"; a reading failed ({type(e).__name__}: {e})"
        record = {"allocated": 0.0, "reserved": 0.0, "start": time.monotonic(), "energy": energy_start}
        self._open.append(record)
        try:
            yield
        finally:
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:  # the phase failed with a CUDA error: record what can be recorded
                pass
            seconds = time.monotonic() - record["start"]
            try:
                energy = self.source.end(key, record["energy"], seconds) if begun else None
            except Exception as e:  # a failing read must not fail the op
                energy = None
                self.source.note += f"; a reading failed ({type(e).__name__}: {e})"
            self._fold_peaks()
            self._count_processes()
            self._open.remove(record)
            total = self.phases.setdefault(name, {"seconds": 0.0, "energy_j": 0.0, "peak_allocated_gb": 0.0,
                                                  "peak_reserved_gb": 0.0, "count": 0})
            total["seconds"] += seconds
            total["energy_j"] = None if energy is None or total["energy_j"] is None else total["energy_j"] + energy
            total["peak_allocated_gb"] = max(total["peak_allocated_gb"], record["allocated"])
            total["peak_reserved_gb"] = max(total["peak_reserved_gb"], record["reserved"])
            total["count"] += 1

    def _count_processes(self) -> None:
        if self.source.board is None:
            return
        nvml, handle = self.source.board
        try:
            n = len(nvml.nvmlDeviceGetComputeRunningProcesses(handle))
        except Exception:  # e.g. not supported on this board: unknown
            return
        self.max_processes = n if self.max_processes is None else max(self.max_processes, n)

    @contextlib.contextmanager
    def count_flops(self, unit: str):
        entry = self.flops.setdefault(unit, {"per_call": None, "calls": 0})
        entry["calls"] += 1
        if entry["calls"] > 1:
            yield
            return
        self._counted = unit
        counter = FlopCount()
        with counter:
            yield
        entry["per_call"] = float(counter.total)

    def step_done(self, start: float, group: str = "train", frames: int | None = None, waited: float | None = None) -> None:
        torch.cuda.synchronize(self.device)
        seconds = time.monotonic() - start
        entry = self.steps.setdefault(group, {"seconds": [], "waits": [], "frames": frames})
        if self._counted == group:  # its time includes the FLOP counter's overhead
            self._counted = None
            return
        entry["seconds"].append(seconds)
        entry["waits"].append(waited)

    def measure_idle(self, seconds: float = IDLE_SECONDS) -> None:
        """The GPU's idle power, over `seconds` with nothing running (a phase "idle", not the op's work)."""
        with self.phase("idle"):
            time.sleep(seconds)
        idle = self.phases["idle"]
        if idle["energy_j"] is not None and self.source.name in ("zeus", "nvml", "board"):  # measured, not estimated
            self.idle_watts = idle["energy_j"] / idle["seconds"]

    def summary(self) -> dict:
        """One session's record (vlu/run_op.py appends it to cost_sessions.jsonl)."""
        steps = {g: {"seconds": [round(x, 5) for x in e["seconds"]],
                     "waits": [None if w is None else round(w, 5) for w in e["waits"]], "frames": e["frames"]}
                 for g, e in self.steps.items()}
        return {"phases": self.phases, "steps": steps, "flops": self.flops, "idle_watts": self.idle_watts,
                "max_processes": self.max_processes, "limit_watts": self.source.limit_watts,
                "energy_source": self.source.name, "energy_share": self.source.share, "energy_note": self.source.note,
                "gpu": torch.cuda.get_device_name(self.device)}


def phase(name: str):
    """A phase of the active meter; nothing without one (tests, tools that train without vlu/run_op.py)."""
    return _ACTIVE[-1].phase(name) if _ACTIVE else contextlib.nullcontext()


def flops(unit: str):
    """Count the FLOPs of a repeated unit of work of the active meter (module docstring); nothing without one."""
    return _ACTIVE[-1].count_flops(unit) if _ACTIVE else contextlib.nullcontext()


def step_done(start: float, group: str = "train", frames: int | None = None, waited: float | None = None) -> None:
    if _ACTIVE:
        _ACTIVE[-1].step_done(start, group, frames, waited)


def frames(batch: dict) -> int | None:
    """The frames of a training batch ({part: inputs}), or None if its parts do not say."""
    try:
        return sum(len(inputs["action"]) for inputs in batch.values())
    except (KeyError, TypeError):
        return None


def peak_allocated_gb() -> float:
    """For the training logs: the peak of the active meter's outermost phase (a phase resets PyTorch's peak
    counter), else PyTorch's peak since the process started."""
    return _ACTIVE[-1].peak_allocated_gb() if _ACTIVE else torch.cuda.max_memory_allocated() / 2**30


# --- cost.json -------------------------------------------------------------------------------------------


def _stats_ms(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "median": 1000 * statistics.median(values), "mean": 1000 * statistics.fmean(values),
            "p90": 1000 * ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]}


def aggregate(sessions: list[dict]) -> dict:
    """The efficiency of an op over its sessions' records (a rerun after a crash adds its own): phases
    summed (seconds, energy; peaks: the maximum), the method's own cost (time, FLOPs, energy, energy above
    idle, mean power, peaks), the steps of every group over every session after its first WARMUP_STEPS,
    and flags for the report. An energy (FLOP count) is None if any session could not measure it.
    Sessions without measurements (killed before recording) are counted in "unmeasured_sessions": the sums
    are then lower bounds."""
    measured = [s for s in sessions if "phases" in s]
    unmeasured = sum("start" in s for s in sessions) - len(measured)
    if not measured:
        return {"method": None, "unmeasured_sessions": unmeasured}
    phases: dict[str, dict] = {}
    for s in measured:
        for name, p in s["phases"].items():
            total = phases.setdefault(name, {"seconds": 0.0, "energy_j": 0.0, "peak_allocated_gb": 0.0,
                                             "peak_reserved_gb": 0.0, "count": 0})
            total["seconds"] += p["seconds"]
            total["energy_j"] = None if p["energy_j"] is None or total["energy_j"] is None else total["energy_j"] + p["energy_j"]
            total["peak_allocated_gb"] = max(total["peak_allocated_gb"], p["peak_allocated_gb"])
            total["peak_reserved_gb"] = max(total["peak_reserved_gb"], p["peak_reserved_gb"])
            total["count"] += p["count"]
    zero = {"seconds": 0.0, "energy_j": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    work = [phases.get(n, zero) for n in ("prepare", "train")]
    inside = [phases.get(n, zero) for n in INSTRUMENTATION]
    energies = [p["energy_j"] for p in work + inside]
    seconds = sum(p["seconds"] for p in work) - sum(p["seconds"] for p in inside)
    energy = None if any(e is None for e in energies) else sum(energies[:2]) - sum(energies[2:])
    # FLOPs: every counted unit of every session (a resumed session counts what it computed itself)
    units: dict[str, dict] = {}
    for s in measured:
        for unit, f in s.get("flops", {}).items():
            u = units.setdefault(unit, {"per_call": f["per_call"], "calls": 0, "flops": 0.0})
            u["calls"] += f["calls"]
            u["flops"] = None if f["per_call"] is None or u["flops"] is None else u["flops"] + f["per_call"] * f["calls"]
    total_flops = None if any(u["flops"] is None for u in units.values()) else sum(u["flops"] for u in units.values())
    idle = [s["idle_watts"] for s in measured if s.get("idle_watts") is not None]
    idle_watts = statistics.median(idle) if idle and len(idle) == len(measured) else None
    method = {"seconds": seconds, "energy_j": energy, "flops": total_flops,
              "energy_above_idle_j": None if energy is None or idle_watts is None else energy - idle_watts * seconds,
              "mean_power_w": None if energy is None or seconds <= 0 else energy / seconds,
              "peak_allocated_gb": max(p["peak_allocated_gb"] for p in work),
              "peak_reserved_gb": max(p["peak_reserved_gb"] for p in work)}
    # Steps per group
    groups: dict[str, dict] = {}
    for s in measured:
        for group, e in s.get("steps", {}).items():
            g = groups.setdefault(group, {"seconds": [], "waits": [], "frames": e["frames"]})
            g["seconds"] += e["seconds"][WARMUP_STEPS:]
            g["waits"] += [w for w in e["waits"][WARMUP_STEPS:] if w is not None]
    steps = {}
    for group, g in groups.items():
        step_ms = _stats_ms(g["seconds"])
        per_step = units.get(group, {}).get("per_call")
        steps[group] = {"step_ms": step_ms, "wait_ms": _stats_ms(g["waits"]), "frames": g["frames"], "flops_per_step": per_step,
                        "achieved_tflops": per_step / (step_ms["median"] / 1000) / 1e12 if per_step and step_ms else None}
    all_steps = [x for g in groups.values() for x in g["seconds"]]
    all_waits = [x for g in groups.values() for x in g["waits"]]
    sources = sorted({s["energy_source"] or "none" for s in measured})
    shares = {s.get("energy_share", 1.0) for s in measured}
    processes = [s["max_processes"] for s in measured if s.get("max_processes") is not None]
    limits = [s["limit_watts"] for s in measured if s.get("limit_watts")]
    flags = []
    if processes and max(processes) > 1:
        flags.append(f"shared GPU: {max(processes)} processes on it, so the energy includes other work")
    if "power_limit" in sources:
        flags.append("energy is a pseudo-estimate (power limit x time), an upper bound")
    if any(share < 1 for share in shares):
        flags.append("MIG slice smaller than the board: energy = board energy x memory share, an estimate")
    if method["mean_power_w"] and limits and sources != ["power_limit"] and method["mean_power_w"] > 1.05 * max(limits):
        flags.append(f"mean power {method['mean_power_w']:.0f} W above the power limit {max(limits):.0f} W: check the readings")
    if idle_watts is not None and method["mean_power_w"] is not None and method["mean_power_w"] < idle_watts:
        flags.append("mean power below the idle power: the idle reading was taken in a higher power state")
    return {"phases": phases, "method": method, "units": units, "steps": steps,
            "step_ms": _stats_ms(all_steps), "wait_ms": _stats_ms(all_waits), "idle_watts": idle_watts,
            "limit_watts": max(limits) if limits else None, "max_processes": max(processes) if processes else None,
            "energy_source": "+".join(sources), "energy_notes": sorted({s["energy_note"] for s in measured if s.get("energy_note")}),
            "gpus": sorted({s["gpu"] for s in measured}), "flags": flags, "unmeasured_sessions": unmeasured}

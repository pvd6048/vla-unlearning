"""Checks of the op cost meter (vlu/cost.py): phases' time and peak memory (nested phases), the energy source
chosen for a full GPU / a MIG slice / no counter / no NVML (fake NVML and Zeus modules: no power is read on
the test machine), and cost.json's aggregate over sessions.   python tests/test_cost.py
"""
import os
from pathlib import Path
import sys
import time
import types

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu import cost  # noqa: E402

dev = torch.cuda.current_device()
props = torch.cuda.get_device_properties(dev)
UUID, MiB = str(props.uuid), 2**20


# --- 1. Phases: time, nested peak memory, the active meter ------------------------------------------------
class Ticks(cost.NoEnergy):
    """A counter that advances 10 J per reading: energy = 10 J x readings in between."""
    name = "ticks"

    def __init__(self):
        super().__init__()
        self.count = 0

    def begin(self, key):
        self.count += 1
        return self.count

    def end(self, key, start, seconds):
        self.count += 1
        return 10.0 * (self.count - start)


meter = cost.Meter(source=Ticks())
assert cost.phase("x").__class__.__name__ == "nullcontext", "no active meter: phase() does nothing"
cost.step_done(time.monotonic())  # no meter: nothing recorded, no error
torch.cuda.empty_cache()
base = torch.cuda.memory_allocated() / 2**30
with meter.active():
    with cost.phase("outer"):
        a = torch.empty(64 * MiB, dtype=torch.uint8, device="cuda")
        big = torch.empty(512 * MiB, dtype=torch.uint8, device="cuda")  # the outer phase's peak, before any inner phase
        del big
        with cost.phase("inner"):
            b = torch.empty(256 * MiB, dtype=torch.uint8, device="cuda")
            time.sleep(0.05)
            del b
        with cost.phase("small"):  # a later phase does not inherit the inner phase's peak
            c = torch.empty(16 * MiB, dtype=torch.uint8, device="cuda")
        running = cost.peak_allocated_gb()
        for _ in range(8):
            start = time.monotonic()
            torch.cuda._sleep(1_000_000)  # GPU work queued but not waited for: step_done synchronizes
            cost.step_done(start)
    with cost.phase("inner"):  # a phase name twice: summed
        pass
    del a, c
ph = meter.phases
gb = lambda mib: mib / 1024  # noqa: E731
assert ph["inner"]["count"] == 2 and ph["inner"]["seconds"] >= 0.05 and ph["outer"]["seconds"] >= ph["inner"]["seconds"] - 0.01
assert abs(ph["inner"]["peak_allocated_gb"] - (base + gb(64 + 256))) < gb(2), (ph["inner"], base)
assert ph["outer"]["peak_allocated_gb"] >= ph["inner"]["peak_allocated_gb"], "an outer phase's peak includes its inner phases'"
assert abs(ph["outer"]["peak_allocated_gb"] - (base + gb(64 + 512))) < gb(2), "kept across the inner phases' resets"
assert abs(ph["small"]["peak_allocated_gb"] - (base + gb(64 + 16))) < gb(2), ph["small"]
assert abs(running - ph["outer"]["peak_allocated_gb"]) < gb(2), "the logs' peak is the outer phase's, across the resets"
assert ph["inner"]["peak_reserved_gb"] >= ph["inner"]["peak_allocated_gb"]
steps = meter.steps["train"]["seconds"]
assert len(steps) == 8 and min(steps) > 0.0003, steps  # the queued GPU sleep was waited for
# energy: inner 10 J each (begin, end); outer spans its readings and those of the phases inside it
assert ph["inner"]["energy_j"] == 20.0 and ph["small"]["energy_j"] == 10.0 and ph["outer"]["energy_j"] == 10.0 * 5, ph
s = meter.summary()
assert s["energy_source"] == "ticks" and len(s["steps"]["train"]["seconds"]) == 8 and s["gpu"] == torch.cuda.get_device_name()
print("1. phases: time, peak memory per phase (outer includes inner, later phases start afresh), energy, steps "
      "timed after their GPU work")

# FLOPs: a repeated unit is counted on its first call (exactly: forward 2BIO + weight gradient 2BIO for a Linear
# layer whose input needs no gradient), times its calls; that call's step is not timed (the counter's overhead).
lin, x = torch.nn.Linear(64, 32).cuda(), torch.randn(8, 64, device="cuda")
m = cost.Meter(source=Ticks())
with m.active():
    for i in range(3):
        start = time.monotonic()
        with cost.flops("lin"):
            lin(x).sum().backward()
        cost.step_done(start, "lin", frames=8, waited=0.01 * i)
    with cost.flops("other"):  # a unit that is not a step (an importance batch)
        lin(x).sum().backward()
    cost.step_done(time.monotonic(), "train")
assert m.flops["lin"] == {"per_call": 4 * 8 * 64 * 32, "calls": 3} and m.flops["other"]["calls"] == 1, m.flops
assert len(m.steps["lin"]["seconds"]) == 2 and m.steps["lin"]["waits"] == [0.01, 0.02] and m.steps["lin"]["frames"] == 8, m.steps
assert len(m.steps["train"]["seconds"]) == 1, "a counted unit that is not a step does not hide the next step"
with cost.flops("x"):  # no active meter: nothing
    pass
assert cost.frames({"a": {"action": torch.zeros(3, 2)}, "b": {"action": torch.zeros(5, 2)}}) == 8 and cost.frames({"a": {}}) is None
# Our counter = torch's FlopCounterMode (same formulas), without its module tracking, whose gradient hooks kept a
# graph alive when a backward did not reach them (tests/test_vlaforget.py counts the saved tensors of a real op)
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Flatten(), torch.nn.Linear(8 * 30 * 30, 16)).cuda()
img, q = torch.randn(4, 3, 32, 32, device="cuda", requires_grad=True), torch.randn(2, 4, 16, 8, device="cuda", requires_grad=True)


def work():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = net(img).sum() + torch.nn.functional.scaled_dot_product_attention(q, q, q).float().sum()
    y.backward()


with FlopCounterMode(display=False) as reference:
    work()
with cost.FlopCount() as ours:
    work()
assert ours.total == reference.get_total_flops() > 0, (ours.total, reference.get_total_flops())
# A FLOP formula that fails leaves the count unknown, never the op; the meter then counts nothing more
class BadFormula(dict):
    def get(self, key, default=None):
        return lambda *a, **k: 1 / 0  # every op's formula fails


m = cost.Meter(source=Ticks())
assert m.flops_note is None, m.flops_note  # counting is safe here (torch's attention; flash-attn if installed)
real_init = cost.FlopCount.__init__
cost.FlopCount.__init__ = lambda self: (real_init(self), setattr(self, "registry", BadFormula()))[0]
try:
    with m.active():
        with cost.flops("u"):
            out = lin(x)
        with cost.flops("v"):
            pass
finally:
    cost.FlopCount.__init__ = real_init
assert out.shape == (8, 32) and m.flops["u"]["per_call"] is None and "ZeroDivisionError" in m.flops_note, (m.flops, m.flops_note)
assert m.flops["v"] == {"per_call": None, "calls": 1}, "not counted once counting failed"
print("   FLOPs counted exactly on a unit's first call, times its calls (= torch's FLOP counter, conv / linear / attention, "
      "forward and backward); that step untimed; waits and frames kept")


# --- 2. Energy source by what the GPU allows (fake NVML, fake Zeus) ----------------------------------------
class FakeNVML(types.ModuleType):
    class NVMLError(Exception):
        pass

    def __init__(self, uuids, board_memory=None, counter=True, limit_mw=600_000, processes=1, by_index=None, mig_parent=True):
        super().__init__("pynvml")
        self.uuids, self.board_memory, self.counter, self.limit_mw, self.mj = uuids, board_memory, counter, limit_mw, 0
        self.processes, self.mig_parent = processes, mig_parent
        self.by_index = {dev: uuids[0]} if by_index is None and uuids else (by_index or {})  # NVML index -> GPU

    def nvmlDeviceGetHandleByIndex(self, i):
        if i not in self.by_index:
            raise self.NVMLError("Invalid Argument")
        return self.by_index[i]

    def nvmlDeviceGetUUID(self, handle):
        return handle.encode()  # older bindings return bytes

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        if self.processes is None:
            raise self.NVMLError("Not Supported")
        return [object()] * self.processes

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByUUID(self, uuid):
        if uuid not in self.uuids:
            raise self.NVMLError("Not Found")
        return uuid

    def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(self, handle):
        if not self.mig_parent:
            raise self.NVMLError("Not Supported")
        return "board-of-" + handle

    def nvmlDeviceGetMemoryInfo(self, handle):
        return types.SimpleNamespace(total=self.board_memory)

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if not self.counter:
            raise self.NVMLError("Not Supported")
        self.mj += 7_000  # 7 J between readings
        return self.mj

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        if self.limit_mw is None:
            raise self.NVMLError("Not Supported")
        return self.limit_mw


class FakeZeusMonitor:
    instances = []

    def __init__(self, gpu_indices=None, cpu_indices=None, approx_instant_energy=False, log_file=None, sync_execution_with="torch"):
        self.args, self.windows = (gpu_indices, cpu_indices, sync_execution_with), set()
        FakeZeusMonitor.instances.append(self)

    def begin_window(self, key, sync_execution=True, restart=False):
        if key in self.windows:
            raise ValueError(f"Measurement window '{key}' already exists")
        self.windows.add(key)

    def end_window(self, key, sync_execution=True, cancel=False):
        self.windows.remove(key)
        return types.SimpleNamespace(time=0.0, gpu_energy={dev: 42.0})


zeus_pkg, zeus_monitor = types.ModuleType("zeus"), types.ModuleType("zeus.monitor")
zeus_monitor.ZeusMonitor = FakeZeusMonitor


def source_with(nvml, zeus=False, visible=None, tdp=cost.TDP_WATTS):
    saved = {k: sys.modules.get(k) for k in ("pynvml", "zeus", "zeus.monitor")}
    saved_env, saved_tdp = os.environ.get("CUDA_VISIBLE_DEVICES"), cost.TDP_WATTS
    try:
        sys.modules["pynvml"] = nvml  # None: `import pynvml` raises ImportError
        sys.modules["zeus"], sys.modules["zeus.monitor"] = (zeus_pkg, zeus_monitor) if zeus else (None, None)
        if visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        cost.TDP_WATTS = tdp
        src = cost.energy_source(dev)
        with cost.Meter(source=src).active() as m:
            for _ in range(2):  # the same phase name twice: Zeus windows must still be unique
                with cost.phase("p"):
                    time.sleep(0.02)
        return src, m.phases["p"], m
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if visible is not None:
            if saved_env is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_env
        cost.TDP_WATTS = saved_tdp


full = [f"GPU-{UUID}"]
src, p, mtr = source_with(FakeNVML(full), zeus=True)
assert src.name == "zeus" and p["energy_j"] == 84.0 and FakeZeusMonitor.instances[-1].args == ([dev], [], "torch"), (src.name, p)
src, p, mtr = source_with(FakeNVML(full), zeus=False)
assert src.name == "nvml" and p["energy_j"] == 14.0 and "Zeus not used" in src.note, (src.name, p, src.note)
assert src.limit_watts == 600.0 and mtr.max_processes == 1, (src.limit_watts, mtr.max_processes)
idle = cost.Meter(source=src)  # idle power: the energy of a window with nothing running / its length
idle.measure_idle(0.1)
assert abs(idle.idle_watts - 7.0 / idle.phases["idle"]["seconds"]) < 1e-9 and idle.phases["idle"]["seconds"] >= 0.1
src, p, mtr = source_with(FakeNVML(full, processes=3))
assert mtr.max_processes == 3, "another program on the GPU is seen"
src, p, mtr = source_with(FakeNVML(full, processes=None))
assert mtr.max_processes is None, "a board that cannot list its processes: unknown"
# MIG: Zeus is not used even if installed (it refuses MIG; NVML has no per-slice energy)
mig = [f"MIG-{UUID}"]
src, p, mtr = source_with(FakeNVML(mig, board_memory=props.total_memory / 0.98), zeus=True)
assert src.name == "board" and src.share == 1.0 and p["energy_j"] == 14.0, (src.name, src.share, p)  # 98%: the board's only slice
src, p, mtr = source_with(FakeNVML(mig, board_memory=4 * props.total_memory), zeus=True)
assert src.name == "board" and src.share == 0.25 and p["energy_j"] == 14.0 * 0.25, (src.share, p)
src, p, mtr = source_with(FakeNVML(mig, board_memory=4 * props.total_memory, counter=False))
assert src.name == "power_limit" and src.watts == 600.0 and abs(p["energy_j"] - 600 * 0.25 * p["seconds"]) < 1e-9, (src.name, p)
src, p, mtr = source_with(FakeNVML(mig, board_memory=props.total_memory, counter=False, limit_mw=None), tdp=(("NVIDIA", 50.0),))
assert src.name == "power_limit" and src.watts == 50.0 and "TDP" in src.note
src, p, mtr = source_with(FakeNVML(mig, board_memory=props.total_memory, counter=False, limit_mw=None), tdp=())
assert src.name is None and p["energy_j"] is None
src, p, mtr = source_with(FakeNVML(["MIG-from-slurm"], board_memory=props.total_memory), visible="MIG-from-slurm")
assert src.name == "board", "a MIG UUID in CUDA_VISIBLE_DEVICES is found too"
src, p, mtr = source_with(None, tdp=(("NVIDIA", 50.0),))
assert src.name == "power_limit" and "no NVML" in src.note and abs(p["energy_j"] - 50 * p["seconds"]) < 1e-9
assert src.limit_watts == 50.0 and mtr.max_processes is None
estimate = cost.Meter(source=src)
estimate.measure_idle(0.01)
assert estimate.idle_watts is None, "no idle power from an estimate"
src, p, mtr = source_with(None, tdp=())
assert src.name is None and p["energy_j"] is None
src, p, mtr = source_with(FakeNVML([]), tdp=())
assert src.name is None and "not found" in src.note
# Zeus finds its GPU by index: if that is another board (CUDA's order is not NVML's), the counter of ours is read
src, p, mtr = source_with(FakeNVML(full, by_index={dev: "GPU-someone-else"}), zeus=True)
assert src.name == "nvml" and "Zeus would measure GPU-someone-else" in src.note, src.note
# Never a failed op: NVML refusing the MIG parent, or failing unexpectedly
src, p, mtr = source_with(FakeNVML(mig, board_memory=props.total_memory, mig_parent=False), tdp=(("NVIDIA", 50.0),))
assert src.name == "power_limit" and "not found" in src.note


class Exploding(FakeNVML):
    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        raise ValueError("driver says no")


src, p, mtr = source_with(Exploding(full))
assert src.name is None and "driver says no" in src.note and p["energy_j"] is None
print("2. energy: Zeus on a full GPU, else NVML's counter; MIG: the board's counter x the slice's share (1 when it "
      "is the whole board), else power limit / TDP x time x share; none without NVML or a known GPU")


# --- 3. A failing reading does not fail the op ----------------------------------------------------------------
class Broken(cost.NoEnergy):
    name = "broken"

    def end(self, key, start, seconds):
        raise RuntimeError("NVML went away")


with cost.Meter(source=Broken()).active() as m, cost.phase("p"):
    pass
assert m.phases["p"]["energy_j"] is None and "NVML went away" in m.source.note


class BrokenStart(cost.NoEnergy):
    name = "broken"

    def begin(self, key):
        raise RuntimeError("no counter today")

    def end(self, key, start, seconds):
        raise AssertionError("not read after a failed start")


with cost.Meter(source=BrokenStart()).active() as m, cost.phase("p"):
    pass
assert m.phases["p"]["energy_j"] is None and "no counter today" in m.source.note
try:
    with cost.Meter(source=Ticks()).active() as m, cost.phase("p"):
        raise ValueError("the op failed")
except ValueError:
    pass
assert m.phases["p"]["count"] == 1, "a failing phase is still recorded"
print("3. a failing energy reading gives no energy, not a failed op; a failing phase is still recorded")


# --- 4. cost.json's aggregate over sessions -------------------------------------------------------------------
def phase_record(seconds, energy, peak, count=1):
    return {"seconds": seconds, "energy_j": energy, "peak_allocated_gb": peak, "peak_reserved_gb": peak + 1, "count": count}


s1 = {"start": 0.0}
extra = {"energy_source": "board", "energy_note": "MIG", "gpu": "G", "limit_watts": 600.0, "max_processes": 1, "energy_share": 1.0}
s1_end = {"seconds": 100.0, "phases": {"load": phase_record(30, 3000, 9), "prepare": phase_record(20, 2000, 30),
                                       "train": phase_record(70, 7000, 20), "probe": phase_record(5, 500, 15, 3),
                                       "checkpoint": phase_record(4, 100, 12)},
          "steps": {"train": {"seconds": [9.0] * cost.WARMUP_STEPS + [0.5, 0.7], "waits": [1.0] * cost.WARMUP_STEPS + [0.1, 0.3],
                              "frames": 16}},
          "flops": {"train": {"per_call": 1e12, "calls": 7}, "importance": {"per_call": 5e11, "calls": 4}}, "idle_watts": 50.0, **extra}
s2_end = {"seconds": 50.0, "phases": {"train": phase_record(50, 5000, 25), "probe": phase_record(2, 200, 15),
                                      "save": phase_record(3, 300, 10)},
          "steps": {"train": {"seconds": [9.0] * cost.WARMUP_STEPS + [0.6], "waits": [1.0] * cost.WARMUP_STEPS + [0.2], "frames": 16}},
          "flops": {"train": {"per_call": 1e12, "calls": 6}}, "idle_watts": 60.0, **extra}
agg = cost.aggregate([s1, s1_end, {"start": 1.0}, s2_end])
m = agg["method"]
assert m["seconds"] == 20 + 120 - 7 - 4 - 3 and m["energy_j"] == 2000 + 12000 - 700 - 100 - 300, m  # load not counted
assert m["flops"] == 13 * 1e12 + 4 * 5e11 and agg["units"]["train"]["calls"] == 13, (m["flops"], agg["units"])
assert abs(m["data_wait_s"] - (10 * 1.0 + 0.1 + 0.3 + 0.2)) < 1e-9, m["data_wait_s"]  # every wait, warm-up included
assert agg["idle_watts"] == 55.0 and abs(m["energy_above_idle_j"] - (12900 - 55.0 * 126)) < 1e-9 and abs(m["mean_power_w"] - 12900 / 126) < 1e-9
assert (m["peak_allocated_gb"], m["peak_reserved_gb"]) == (30, 31) and agg["phases"]["probe"]["count"] == 4
train = agg["steps"]["train"]
assert train["step_ms"]["n"] == 3 and abs(train["step_ms"]["median"] - 600) < 1e-9 and abs(train["step_ms"]["p90"] - 700) < 1e-9, train
assert abs(train["wait_ms"]["median"] - 200) < 1e-9 and train["frames"] == 16 and train["flops_per_step"] == 1e12
assert abs(train["achieved_tflops"] - 1 / 0.6) < 1e-9 and agg["step_ms"] == train["step_ms"]
assert agg["energy_source"] == "board" and agg["gpus"] == ["G"] and agg["unmeasured_sessions"] == 0 and agg["flags"] == [], agg["flags"]
s3 = {"start": 2.0}  # killed: no record
s2_none = {**s2_end, "phases": {**s2_end["phases"], "train": phase_record(50, None, 25)}, "energy_source": None,
           "flops": {"train": {"per_call": None, "calls": 6}}}
agg = cost.aggregate([s1, s1_end, {"start": 1.0}, s2_none, s3])
assert agg["method"]["energy_j"] is None and agg["method"]["energy_above_idle_j"] is None
assert agg["method"]["flops"] == 13 * 1e12 + 4 * 5e11, "a session that could not count takes another session's count"
agg = cost.aggregate([s1, s2_none, {"start": 1.0}, s1_end])  # also when the session that could not count came first
assert agg["method"]["flops"] == 13 * 1e12 + 4 * 5e11, agg["units"]
only_unknown = {**s1_end, "flops": {"train": {"per_call": None, "calls": 7}}, "flops_note": "unit train: boom"}
agg = cost.aggregate([{"start": 0.0}, only_unknown])
assert agg["method"]["flops"] is None and "FLOPs not counted (see the FLOP notes)" in agg["flags"] and agg["flops_notes"] == ["unit train: boom"]
agg = cost.aggregate([s1, s1_end, {"start": 1.0}, s2_none, s3])
assert agg["energy_source"] == "board+none" and agg["unmeasured_sessions"] == 1, agg
assert cost.aggregate([{"start": 0.0}, {"seconds": 5.0}])["method"] is None, "sessions killed before recording"
# Flags: a shared GPU, a pseudo-estimate, a slice smaller than the board, a mean power above the limit or below idle
flagged = lambda **kw: cost.aggregate([{"start": 0.0}, {**s1_end, **kw}])["flags"]  # noqa: E731
assert any("shared GPU: 2 processes" in f for f in flagged(max_processes=2))
assert any("pseudo-estimate" in f for f in flagged(energy_source="power_limit"))
assert any("MIG slice smaller" in f for f in flagged(energy_share=0.25))
assert any("above the power limit 50 W" in f for f in flagged(limit_watts=50.0))
assert any("below the idle power" in f for f in flagged(idle_watts=500.0))
print("4. aggregate: method = prepare + train - probes - checkpoints - saves, summed over sessions (time, energy, FLOPs); "
      "energy above idle, mean power; peaks the max; steps, waits and achieved TFLOP/s after each session's warm-up; "
      "flags for a shared GPU, estimates and implausible readings; an unmeasurable value or a session without record is flagged")
print("ALL COST CHECKS PASSED")

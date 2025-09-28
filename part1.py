#!/usr/bin/env python3
# coherence.py
#
# Usage:
#   python coherence.py MESI bodytrack 1024 1 16
#   python coherence.py Dragon blackscholes 4096 2 32
#
# Implements Part 1: single-core trace-driven cache/DRAM timing with stats.
# Leaves hooks for multi-core + MESI/Dragon.

import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# --------- Timing & architectural constants (from spec) ----------
WORD_BYTES = 4  # 32-bit word
HIT_CYCLES = 1
MEM_FETCH_BLOCK_CYCLES = 100       # miss service from memory
DIRTY_EVICT_TO_MEM_CYCLES = 100    # writeback dirty block to memory
C2C_WORD_CYCLES = 2                # (unused in Part 1; used in Dragon/MESI cache-to-cache)

# --------- Labels in the trace ----------
LOAD = 0
STORE = 1
OTHER = 2

# --------- Cache line states (subset for Part 1; expand for MESI/Dragon) ----------
class LineState:
    INVALID = "I"
    EXCLUSIVE = "E"  # single-core: first read miss can be E
    MODIFIED = "M"   # single-core: after a write

@dataclass
class CacheLine:
    tag: int = -1
    state: str = LineState.INVALID
    dirty: bool = False
    lru_tick: int = 0  # larger

@dataclass
class CacheSet:
    lines: List[CacheLine]

@dataclass
class CacheConfig:
    size_bytes: int
    assoc: int
    block_bytes: int

    def __post_init__(self):
        assert self.size_bytes % (self.assoc * self.block_bytes) == 0, "Invalid cache geometry"

    @property
    def num_sets(self) -> int:
        return self.size_bytes // (self.assoc * self.block_bytes)

    @property
    def idx_bits(self) -> int:
        return int(math.log2(self.num_sets)) if self.num_sets > 1 else 0

    @property
    def off_bits(self) -> int:
        return int(math.log2(self.block_bytes))

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    stores: int = 0
    compute_cycles: int = 0
    idle_cycles: int = 0
    bus_data_bytes: int = 0            # data traffic only (no addresses)
    invalidations_or_updates: int = 0  # 0 for single-core
    private_accesses: int = 0          # accesses to E/M (private)
    shared_accesses: int = 0           # accesses to S state (for Part 1, 0)

class LRU:
    """Monotonic counter used to implement LRU across lines."""
    def __init__(self): self.t = 0
    def tick(self) -> int: self.t += 1; return self.t

class L1Cache:
    """
    Blocking, write-back, write-allocate, LRU.
    Single-core behavior for Part 1 (no snoops).
    """
    def __init__(self, cfg: CacheConfig, stats: CacheStats):
        self.cfg = cfg
        self.stats = stats
        self.lru = LRU()
        self.sets = [CacheSet([CacheLine() for _ in range(cfg.assoc)]) for _ in range(cfg.num_sets or 1)]

    def _addr_fields(self, addr: int) -> Tuple[int, int, int]:
        off_mask = (1 << self.cfg.off_bits) - 1
        idx_mask = (1 << self.cfg.idx_bits) - 1
        off = addr & off_mask
        idx = (addr >> self.cfg.off_bits) & idx_mask if self.cfg.idx_bits > 0 else 0
        tag = addr >> (self.cfg.off_bits + self.cfg.idx_bits)
        return tag, idx, off

    # if line is alreayd inside the cache, return it; else None
    def _find_line(self, idx: int, tag: int) -> Optional[CacheLine]:
        for line in self.sets[idx].lines:
            if line.state != LineState.INVALID and line.tag == tag:
                return line
        return None

    def _select_victim(self, idx: int) -> CacheLine:
        # Prefer INVALID if available, else LRU 
        for line in self.sets[idx].lines:
            if line.state == LineState.INVALID:
                return line
        # Evict line with largest lru_tick smallest
        return min(self.sets[idx].lines, key=lambda ln: ln.lru_tick)

    def _touch_lru(self, idx: int, line: CacheLine) -> None:
        line.lru_tick = self.lru.tick()

    def read(self, addr: int) -> int:
        """
        Returns service latency (cycles) for this access, and mutates stats.
        """
        self.stats.loads += 1
        tag, idx, _ = self._addr_fields(addr)
        line = self._find_line(idx, tag)

        if line:
            # Hit
            self.stats.hits += 1
            self._touch_lru(idx, line)
            # Private vs shared: single-core -> E or M count as private
            if line.state in (LineState.EXCLUSIVE, LineState.MODIFIED):
                self.stats.private_accesses += 1
            # Hit cost
            return HIT_CYCLES

        # Miss
        self.stats.misses += 1
        victim = self._select_victim(idx)

        # If victim is dirty, write it back (data traffic + latency)
        latency = 0
        if victim.state != LineState.INVALID and victim.dirty:
            self.stats.bus_data_bytes += self.cfg.block_bytes
            latency += DIRTY_EVICT_TO_MEM_CYCLES

        # Fetch block from memory (data traffic + latency)
        self.stats.bus_data_bytes += self.cfg.block_bytes
        latency += MEM_FETCH_BLOCK_CYCLES

        # Install as E on read miss (single-core)
        victim.tag = tag
        victim.state = LineState.EXCLUSIVE
        victim.dirty = False
        self._touch_lru(idx, victim)

        # Access itself (1 cycle)
        return latency + HIT_CYCLES

    def write(self, addr: int) -> int:
        """
        Returns service latency (cycles) for this access, and mutates stats.
        """
        self.stats.stores += 1
        tag, idx, _ = self._addr_fields(addr)
        line = self._find_line(idx, tag)

        if line:
            # Hit
            self.stats.hits += 1
            # Upgrade E->M if needed
            if line.state == LineState.EXCLUSIVE:
                line.state = LineState.MODIFIED
            elif line.state == LineState.MODIFIED:
                pass
            else:
                # (No S in single-core. Multi-core coherence would handle S->M via BusUpgr.)
                line.state = LineState.MODIFIED
            line.dirty = True
            self._touch_lru(idx, line)
            self.stats.private_accesses += 1  # single-core writes are private
            return HIT_CYCLES

        # Miss (write-allocate)
        self.stats.misses += 1
        victim = self._select_victim(idx)

        latency = 0
        if victim.state != LineState.INVALID and victim.dirty:
            self.stats.bus_data_bytes += self.cfg.block_bytes
            latency += DIRTY_EVICT_TO_MEM_CYCLES

        # Fetch block (write-allocate)
        self.stats.bus_data_bytes += self.cfg.block_bytes
        latency += MEM_FETCH_BLOCK_CYCLES

        victim.tag = tag
        victim.state = LineState.MODIFIED
        victim.dirty = True
        self._touch_lru(idx, victim)

        return latency + HIT_CYCLES

class TraceReader:
    """
    Reads a single core's trace file: lines 'Label Value'.
    OTHER lines provide compute cycles (Value is hex).
    """
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with self.path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                a, b = line.split()
                label = int(a)
                val = int(b, 16 if b.lower().startswith("0x") else 10)
                yield label, val

@dataclass
class SingleCoreCPU:
    """Blocking core+L1 model; progresses its own time counter."""
    l1: L1Cache
    stats: CacheStats
    time: int = 0  # core-local cycle counter

    def exec_trace(self, tr: TraceReader) -> None:
        for label, value in tr:
            if label == OTHER:
                self.stats.compute_cycles += value
                self.time += value
                continue

            # Memory access: blocking cache; add service latency
            if label == LOAD:
                latency = self.l1.read(value)
            elif label == STORE:
                latency = self.l1.write(value)
            else:
                raise ValueError(f"Unknown label {label}")

            # All miss service time is idle (core stalled); even hits cost 1 cycle "busy".
            idle = max(0, latency - 1)  # treat the extra beyond the 1-cycle access as idle
            self.stats.idle_cycles += idle
            self.time += latency

# --------------- CLI & Orchestration -----------------------------

def resolve_trace_path(benchmark_base: str, traces_root: Optional[Path]) -> Path:
    """
    For Part 1 we use core 0 trace: <benchmark>_0.data.
    You can pass --traces-root to point to the folder with *.data files.
    """
    # Default search locations: current dir, provided root, or SoC path env.
    candidates = []
    if traces_root: candidates.append(traces_root)
    candidates += [Path.cwd(), Path("/home/course/cs4223/assignments/assignment2")]

    fname = f"{benchmark_base}_0.data"
    for root in candidates:
        p = root / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {fname} under {candidates}")

def run_single_core(protocol: str, benchmark_base: str, cfg: CacheConfig, traces_root: Optional[Path]) -> Tuple[SingleCoreCPU, CacheStats]:
    stats = CacheStats()
    l1 = L1Cache(cfg, stats)
    core = SingleCoreCPU(l1, stats)
    trace_path = resolve_trace_path(benchmark_base, traces_root)
    tr = TraceReader(trace_path)
    core.exec_trace(tr)
    return core, stats

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"

def main():
    ap = argparse.ArgumentParser(description="Trace-driven cache coherence simulator")
    ap.add_argument("protocol", choices=["MESI", "Dragon"], help="Coherence protocol (for Part 2). Part 1 ignores.")
    ap.add_argument("input_file", help="Benchmark base name (e.g., bodytrack, blackscholes, fluidanimate)")
    ap.add_argument("cache_size", type=int, help="L1 size in bytes")
    ap.add_argument("associativity", type=int, help="L1 associativity")
    ap.add_argument("block_size", type=int, help="L1 block size in bytes")
    ap.add_argument("--traces-root", type=Path, default=None, help="Directory with *_N.data files")
    args = ap.parse_args()

    cfg = CacheConfig(size_bytes=args.cache_size, assoc=args.associativity, block_bytes=args.block_size)

    # Part 1: single-core
    core, stats = run_single_core(args.protocol, args.input_file, cfg, args.traces_root)

    # ----- Output (machine-readable-ish then human summary) -----
    # Core-level (only one core for Part 1)
    print("==== RESULTS ====")
    print(f"protocol={args.protocol}")
    print(f"benchmark={args.input_file}")
    print(f"cache_size_bytes={cfg.size_bytes} assoc={cfg.assoc} block_bytes={cfg.block_bytes}")
    print(f"overall_exec_cycles={core.time}")           # For multi-core later: max across cores
    print(f"core0_exec_cycles={core.time}")
    print(f"core0_compute_cycles={stats.compute_cycles}")
    print(f"core0_loads={stats.loads} core0_stores={stats.stores}")
    print(f"core0_hits={stats.hits} core0_misses={stats.misses}")
    print(f"core0_idle_cycles={stats.idle_cycles}")
    print(f"bus_data_traffic_bytes={stats.bus_data_bytes} ({human_bytes(stats.bus_data_bytes)})")
    print(f"bus_invals_or_updates={stats.invalidations_or_updates}")  # 0 for single-core
    print(f"private_accesses={stats.private_accesses} shared_accesses={stats.shared_accesses}")  # shared=0 in Part 1

if __name__ == "__main__":
    main()

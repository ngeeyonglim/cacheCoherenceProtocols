#!/usr/bin/env python3
# coherence.py
#
# Usage:
#   python coherence.py MESI bodytrack 1024 1 16
#   python coherence.py Dragon blackscholes 4096 2 32
#
# Implements Part 2: multi-core MESI trace-driven cache/DRAM timing with stats.

import argparse
import math
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# --------- Timing & architectural constants (from spec) ----------
WORD_BYTES = 4  # 32-bit word
HIT_CYCLES = 1
MEM_FETCH_BLOCK_CYCLES = 100       # miss service from memory
DIRTY_EVICT_TO_MEM_CYCLES = 100    # writeback dirty block to memory
# (unused in Part 1; used in Dragon/MESI cache-to-cache)
C2C_WORD_CYCLES = 2

# --------- Labels in the trace ----------
LOAD = 0    # [cite: 25]
STORE = 1   # [cite: 25]
OTHER = 2   # [cite: 25]

# --------- Cache line states (MESI) ----------


class LineState:
    MODIFIED = "M"
    EXCLUSIVE = "E"
    SHARED = "S"
    INVALID = "I"

# --------- Bus Transaction Types ----------


class BusXact:
    BusRd = "BusRd"      # Read request from a cache
    BusRdX = "BusRdX"    # Read-for-ownership (write miss or S->M upgrade)
    Flush = "Flush"      # A cache is flushing a block to another cache/memory


@dataclass
class CacheLine:
    tag: int = -1
    state: str = LineState.INVALID
    lru_tick: int = 0


@dataclass
class CacheSet:
    lines: List[CacheLine]


@dataclass
class CacheConfig:
    size_bytes: int
    assoc: int
    block_bytes: int

    def __post_init__(self):
        assert self.size_bytes % (
            self.assoc * self.block_bytes) == 0, "Invalid cache geometry"

    @property
    def num_sets(self) -> int:
        return self.size_bytes // (self.assoc * self.block_bytes)

    @property
    def idx_bits(self) -> int:
        return int(math.log2(self.num_sets)) if self.num_sets > 1 else 0

    @property
    def off_bits(self) -> int:
        return int(math.log2(self.block_bytes))

    @property
    def words_per_block(self) -> int:
        return self.block_bytes // WORD_BYTES


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    stores: int = 0
    compute_cycles: int = 0
    idle_cycles: int = 0
    private_accesses: int = 0
    shared_accesses: int = 0

# --- Global Stats (shared across all cores/caches) ---


@dataclass
class GlobalStats:
    bus_data_bytes: int = 0
    invalidations_or_updates: int = 0


class LRU:
    def __init__(self): self.t = 0
    def tick(self) -> int: self.t += 1; return self.t


class Bus:
    """
    FCFS shared bus. Manages transactions and snooping.
    A transaction takes multiple cycles to complete.
    """

    def __init__(self, global_stats: GlobalStats):
        self.g_stats = global_stats
        self.queue = deque()
        self.snoopers: List['L1Cache'] = []
        self.current_xact: Optional[Dict[str, Any]] = None
        self.cycles_rem: int = 0

    def add_snooper(self, cache: 'L1Cache'):
        self.snoopers.append(cache)

    def push_request(self, core_id: int, xact_type: str, addr: int):
        self.queue.append(
            {'core_id': core_id, 'type': xact_type, 'addr': addr})

    def tick(self):
        if not self.current_xact:
            if not self.queue:
                return  # Bus is idle
            # Start a new transaction
            self.current_xact = self.queue.popleft()
            self.process_new_xact()

        self.cycles_rem -= 1

        if self.cycles_rem <= 0:
            self.finish_current_xact()
            self.current_xact = None

    def process_new_xact(self):
        """Snoop other caches and determine latency for the new transaction."""
        xact = self.current_xact
        req_core_id, xact_type, addr = xact['core_id'], xact['type'], xact['addr']

        # Snoop all OTHER caches
        snoop_responses = []
        for snooper in self.snoopers:
            if snooper.core_id != req_core_id:
                snoop_responses.append(snooper.snoop(xact_type, addr))

        was_dirty = LineState.MODIFIED in snoop_responses
        # checks if any other cache has a valid copy (M, E, or S).
        is_shared = any(r is not None for r in snoop_responses)

        if xact_type == BusXact.BusRd:
            if was_dirty:
                # Cache-to-cache transfer
                self.cycles_rem = C2C_WORD_CYCLES * \
                    self.snoopers[0].cfg.words_per_block
                self.g_stats.bus_data_bytes += self.snoopers[0].cfg.block_bytes
            else:
                # Memory fetch
                self.cycles_rem = MEM_FETCH_BLOCK_CYCLES
                self.g_stats.bus_data_bytes += self.snoopers[0].cfg.block_bytes
            # Determine final state for requesting cache
            xact['final_state'] = LineState.SHARED if is_shared else LineState.EXCLUSIVE

        elif xact_type == BusXact.BusRdX:
            # Invalidation count is handled by snooping caches
            if was_dirty:
                # C2C transfer
                self.cycles_rem = C2C_WORD_CYCLES * \
                    self.snoopers[0].cfg.words_per_block
                self.g_stats.bus_data_bytes += self.snoopers[0].cfg.block_bytes
            else:
                # Memory fetch
                self.cycles_rem = MEM_FETCH_BLOCK_CYCLES
                self.g_stats.bus_data_bytes += self.snoopers[0].cfg.block_bytes
            xact['final_state'] = LineState.MODIFIED

        elif xact_type == BusXact.Flush:
            # This is a writeback due to eviction
            self.cycles_rem = DIRTY_EVICT_TO_MEM_CYCLES
            self.g_stats.bus_data_bytes += self.snoopers[0].cfg.block_bytes
            xact['final_state'] = None  # No state change for others

    def finish_current_xact(self):
        """Notify the originating cache that its transaction is complete."""
        xact = self.current_xact
        if xact['type'] != BusXact.Flush:  # Flushes don't need a response
            origin_cache = self.snoopers[xact['core_id']]
            origin_cache.complete_request(xact['addr'], xact['final_state'])


class L1Cache:
    def __init__(self, core_id: int, cfg: CacheConfig, stats: CacheStats, bus: Bus, g_stats: GlobalStats):
        self.core_id = core_id
        self.cfg = cfg
        self.stats = stats
        self.bus = bus
        self.g_stats = g_stats
        self.lru = LRU()
        self.sets = [CacheSet([CacheLine() for _ in range(cfg.assoc)])
                     for _ in range(cfg.num_sets)]

    def _addr_fields(self, addr: int) -> Tuple[int, int, int]:
        off_mask = (1 << self.cfg.off_bits) - 1
        idx_mask = (1 << self.cfg.idx_bits) - 1
        off = addr & off_mask
        idx = (addr >> self.cfg.off_bits) & idx_mask if self.cfg.idx_bits > 0 else 0
        tag = addr >> (self.cfg.off_bits + self.cfg.idx_bits)
        return tag, idx, off

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
        return min(self.sets[idx].lines, key=lambda ln: ln.lru_tick)

    def _touch_lru(self, line: CacheLine):
        line.lru_tick = self.lru.tick()

    def access(self, addr: int, access_type: int) -> bool:
        """
        Returns True if hit, False if miss (and queues bus request).
        Mutates stats.
        """
        is_load = (access_type == LOAD)
        if is_load:
            self.stats.loads += 1
        else:
            self.stats.stores += 1

        tag, idx, off = self._addr_fields(addr)
        line = self._find_line(idx, tag)

        if not line:  # Miss
            self.stats.misses += 1
            # Issue BusRd on read miss, BusRdX on write miss
            xact_type = BusXact.BusRd if is_load else BusXact.BusRdX
            self.bus.push_request(self.core_id, xact_type, addr)
            return False  # Stall

        # Hit
        self.stats.hits += 1

        if line.state in (LineState.EXCLUSIVE, LineState.MODIFIED):
            self.stats.private_accesses += 1
        else:  # Shared
            self.stats.shared_accesses += 1

        if is_load:  # Load
            self._touch_lru(line)
            return True
        else:  # Store
            if line.state == LineState.MODIFIED:
                self._touch_lru(line)
                return True
            elif line.state == LineState.EXCLUSIVE:
                line.state = LineState.MODIFIED
                self._touch_lru(line)
                return True
            elif line.state == LineState.SHARED:
                # Upgrade S -> M. Need to invalidate others.
                self.bus.push_request(self.core_id, BusXact.BusRdX, addr)
                return False  # Stall

    def complete_request(self, addr: int, final_state: str):
        """Callback from the bus when our memory request is done."""
        tag, idx, off = self._addr_fields(addr)

        # Check if we are upgrading a shared line
        line = self._find_line(idx, tag)
        if line and line.state == LineState.SHARED:  # S->M upgrade
            line.state = LineState.MODIFIED
            self._touch_lru(line)
            return

        # Otherwise, it's a miss fill
        victim = self._select_victim(idx)
        if victim.state == LineState.MODIFIED:
            victim_addr = self._reconstruct_addr(victim.tag, idx)
            self.bus.push_request(self.core_id, BusXact.Flush, victim_addr)

        victim.tag = tag
        victim.state = final_state
        self._touch_lru(victim)

    def snoop(self, xact_type: str, addr: int) -> Optional[str]:
        """Process a snooped transaction. Return line state if we have it."""
        tag, idx, off = self._addr_fields(addr)
        line = self._find_line(idx, tag)
        if not line:
            return None

        original_state = line.state
        if xact_type == BusXact.BusRd:
            if line.state == LineState.EXCLUSIVE or line.state == LineState.MODIFIED:
                line.state = LineState.SHARED

        elif xact_type == BusXact.BusRdX:
            if line.state in (LineState.SHARED, LineState.EXCLUSIVE, LineState.MODIFIED):
                line.state = LineState.INVALID
                self.g_stats.invalidations_or_updates += 1

        return original_state

    def _reconstruct_addr(self, tag: int, idx: int) -> int:
        """Reconstructs the base address of a block from its tag and index."""
        return (tag << (self.cfg.idx_bits + self.cfg.off_bits)) | (idx << self.cfg.off_bits)


class TraceReader:
    def __init__(self, path: Path):
        self.trace_iter = iter(self._read_trace(path))
        self.current_instr = None
        self.finished = False
        self.advance()

    def _read_trace(self, path):
        with path.open('r') as f:
            for line in f:
                if line.strip():
                    a, b = line.split()
                    yield int(a), int(b, 16)

    def advance(self):
        try:
            self.current_instr = next(self.trace_iter)
        except StopIteration:
            self.current_instr = None
            self.finished = True


@dataclass
class CPU:
    core_id: int
    l1: L1Cache
    stats: CacheStats
    tr: TraceReader

    # Core state machine
    stalled: bool = False
    compute_cycles_rem: int = 0

    def is_finished(self) -> bool:
        return self.tr.finished

    def tick(self):
        if self.stalled or self.is_finished():
            if not self.is_finished():
                self.stats.idle_cycles += 1
            return

        # Currently computing between memory ops
        if self.compute_cycles_rem > 0:
            self.compute_cycles_rem -= 1
            return

        label, value = self.tr.current_instr
        if label == OTHER:
            self.compute_cycles_rem = value
            self.stats.compute_cycles += value
            self.tr.advance()

            if self.compute_cycles_rem > 0:
                self.compute_cycles_rem -= 1
            return

        is_hit = self.l1.access(value, label)
        if is_hit:
            self.tr.advance()
        else:
            self.stalled = True

    def un_stall(self):
        self.stalled = False
        self.tr.advance()


class Simulator:
    def __init__(self, protocol: str, benchmark: str, cfg: CacheConfig, traces_root: Optional[Path]):
        self.global_time = 0
        self.g_stats = GlobalStats()
        self.bus = Bus(self.g_stats)
        self.cpus: List[CPU] = []

        trace_paths = self.resolve_trace_paths(benchmark, traces_root)
        print(f"Found {len(trace_paths)} trace file(s) for '{benchmark}'.")

        for i, path in enumerate(trace_paths):
            stats = CacheStats()
            cache = L1Cache(i, cfg, stats, self.bus, self.g_stats)
            cpu = CPU(i, cache, stats, TraceReader(path))
            cache.complete_request = lambda addr, state, c=cpu, self_cache=cache: (
                c.un_stall(),
                L1Cache.complete_request(self_cache, addr, state)
            )
            self.bus.add_snooper(cache)
            self.cpus.append(cpu)

    def resolve_trace_paths(self, benchmark_base: str, traces_root: Optional[Path]) -> List[Path]:
        candidates = [traces_root] if traces_root else []
        candidates += [Path.cwd(),
                       Path("/home/course/cs4223/assignments/assignment2")]

        found_paths = []
        for core_id in range(4):  # Max 4 cores
            fname = f"{benchmark_base}_{core_id}.data"
            for root in candidates:
                if root and (p := root / fname).exists():
                    found_paths.append(p)
                    break
        if not found_paths:
            raise FileNotFoundError(
                f"Could not find any traces for {benchmark_base} under {candidates}")
        return found_paths

    def run(self):
        while any(not cpu.is_finished() for cpu in self.cpus):
            self.global_time += 1
            self.bus.tick()
            for cpu in self.cpus:
                if not cpu.is_finished():
                    cpu.tick()

    def print_results(self, protocol, benchmark, cfg):
        print("==== RESULTS ====")
        print(f"protocol={protocol}")
        print(f"benchmark={benchmark}")
        print(
            f"cache_size_bytes={cfg.size_bytes} assoc={cfg.assoc} block_bytes={cfg.block_bytes}")
        print(f"overall_exec_cycles={self.global_time}")

        # Loop through each CPU/core and print its specific results
        for i, cpu in enumerate(self.cpus):
            stats = cpu.stats

            # Final exec time for a core is the global time if it finished
            print(f"--- Core {i} Stats ---")
            print(f"core{i}_compute_cycles={stats.compute_cycles}")
            print(f"core{i}_loads={stats.loads} core{i}_stores={stats.stores}")
            print(f"core{i}_hits={stats.hits} core{i}_misses={stats.misses}")
            print(f"core{i}_idle_cycles={stats.idle_cycles}")

            # ADDED: Per-core private and shared access counts
            print(f"core{i}_private_accesses={stats.private_accesses}")
            print(f"core{i}_shared_accesses={stats.shared_accesses}")

        # Print the shared bus statistics once at the end
        print("--- Global Bus Stats ---")
        print(
            f"bus_data_traffic_bytes={self.g_stats.bus_data_bytes} ({human_bytes(self.g_stats.bus_data_bytes)})")
        print(f"bus_invals_or_updates={self.g_stats.invalidations_or_updates}")


def human_bytes(n: int) -> str:
    if n is None:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"


def main():
    ap = argparse.ArgumentParser(
        description="Trace-driven cache coherence simulator")
    ap.add_argument("protocol", choices=[
                    "MESI", "Dragon"], help="Coherence protocol.")
    ap.add_argument("input_file", help="Benchmark base name.")
    ap.add_argument("cache_size", type=int, help="L1 size in bytes")
    ap.add_argument("associativity", type=int, help="L1 associativity")
    ap.add_argument("block_size", type=int, help="L1 block size in bytes")
    ap.add_argument("--traces-root", type=Path, default=None,
                    help="Directory with *.data files")
    args = ap.parse_args()

    if args.protocol == "Dragon":
        print("Dragon protocol not implemented in this version.")
        return

    cfg = CacheConfig(size_bytes=args.cache_size,
                      assoc=args.associativity, block_bytes=args.block_size)

    sim = Simulator(args.protocol, args.input_file, cfg, args.traces_root)
    sim.run()
    sim.print_results(args.protocol, args.input_file, cfg)


if __name__ == "__main__":
    main()

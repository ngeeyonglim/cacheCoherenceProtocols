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
import heapq
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# --------- Timing & architectural constants (from spec) ----------
WORD_BYTES = 4  # 32-bit word
HIT_CYCLES = 1
MEM_FETCH_BLOCK_CYCLES = 100       # miss service from memory
DIRTY_EVICT_TO_MEM_CYCLES = 100    # writeback dirty block to memory
C2C_WORD_CYCLES = 2

# --------- Labels in the trace ----------
LOAD = 0
STORE = 1
OTHER = 2

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
    A transaction is now an event, not a tick-based process.
    """

    def __init__(self, global_stats: GlobalStats, sim: 'Simulator'):
        self.g_stats = global_stats
        self.sim = sim
        self.queue = deque()
        self.snoopers: List['L1Cache'] = []
        self.is_busy: bool = False

    def add_snooper(self, cache: 'L1Cache'):
        self.snoopers.append(cache)

    def push_request(self, core_id: int, xact_type: str, addr: int):
        self.queue.append(
            {'core_id': core_id, 'type': xact_type, 'addr': addr})
        self.try_start_next_xact()

    def try_start_next_xact(self):
        """If the bus is free and has pending requests, start one."""
        if self.is_busy or not self.queue:
            return  # Bus is busy or no work to do

        self.is_busy = True
        xact = self.queue.popleft()
        self.process_new_xact(xact)

    def process_new_xact(self, xact: Dict[str, Any]):
        """Snoop other caches, determine latency, and schedule completion event."""
        req_core_id, xact_type, addr = xact['core_id'], xact['type'], xact['addr']

        # Snoop all OTHER caches
        snoop_responses = []
        for snooper in self.snoopers:
            if snooper.core_id != req_core_id:
                snoop_responses.append(snooper.snoop(xact_type, addr))

        was_dirty = LineState.MODIFIED in snoop_responses
        is_shared = any(r is not None for r in snoop_responses)

        latency = 0
        block_bytes = self.snoopers[0].cfg.block_bytes
        words_per_block = self.snoopers[0].cfg.words_per_block

        if xact_type == BusXact.BusRd:
            if was_dirty or is_shared:
                latency = C2C_WORD_CYCLES * words_per_block
                self.g_stats.bus_data_bytes += block_bytes
            else:
                latency = MEM_FETCH_BLOCK_CYCLES
                self.g_stats.bus_data_bytes += block_bytes
            xact['final_state'] = LineState.SHARED if is_shared else LineState.EXCLUSIVE

        elif xact_type == BusXact.BusRdX:
            if was_dirty:
                latency = C2C_WORD_CYCLES * words_per_block
                self.g_stats.bus_data_bytes += block_bytes
            else:
                latency = MEM_FETCH_BLOCK_CYCLES
                self.g_stats.bus_data_bytes += block_bytes
            xact['final_state'] = LineState.MODIFIED

        elif xact_type == BusXact.Flush:
            latency = DIRTY_EVICT_TO_MEM_CYCLES
            self.g_stats.bus_data_bytes += block_bytes
            xact['final_state'] = None

        self.sim.schedule(latency, self.finish_current_xact, xact)

    def finish_current_xact(self, xact: Dict[str, Any]):
        """Event callback when a bus transaction completes."""
        self.is_busy = False  # Free the bus

        # Notify the originating cache
        if xact['type'] != BusXact.Flush:
            origin_cache = self.snoopers[xact['core_id']]
            origin_cache.complete_request(xact['addr'], xact['final_state'])

        # Try to start the next transaction in the queue
        self.try_start_next_xact()


class L1Cache:
    def __init__(self, core_id: int, cfg: CacheConfig, stats: CacheStats, bus: Bus, g_stats: GlobalStats, cpu: 'CPU'):
        self.core_id = core_id
        self.cfg = cfg
        self.stats = stats
        self.bus = bus
        self.g_stats = g_stats
        self.cpu = cpu
        self.lru = LRU()
        self.sets = [CacheSet([CacheLine() for _ in range(cfg.assoc)])
                     for _ in range(cfg.num_sets)]

        # Maps block address -> victim line to be filled
        self.pending_misses: Dict[int, CacheLine] = {}

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
        for line in self.sets[idx].lines:
            if line.state == LineState.INVALID:
                return line
        return min(self.sets[idx].lines, key=lambda ln: ln.lru_tick)

    def _touch_lru(self, line: CacheLine):
        line.lru_tick = self.lru.tick()

    def access(self, addr: int, access_type: int) -> Tuple[bool, int]:
        """
        Returns (is_hit, latency). On miss, returns (False, 0)
        and queues a bus request.
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

            victim = self._select_victim(idx)

            # If victim is dirty, queue the blocking Flush
            if victim.state == LineState.MODIFIED:
                victim_addr = self._reconstruct_addr(victim.tag, idx)
                self.bus.push_request(self.core_id, BusXact.Flush, victim_addr)

            # Store this victim to be filled when the fetch completes
            # We use the block-aligned address as the key
            block_addr = addr & ~((1 << self.cfg.off_bits) - 1)
            self.pending_misses[block_addr] = victim

            # Mark victim as Invalid so it can't be snooped
            victim.state = LineState.INVALID

            # Queue the data fetch (BusRd or BusRdX)
            xact_type = BusXact.BusRd if is_load else BusXact.BusRdX
            self.bus.push_request(self.core_id, xact_type, addr)

            return (False, 0)  # Stall

        # Hit
        self.stats.hits += 1

        if line.state in (LineState.EXCLUSIVE, LineState.MODIFIED):
            self.stats.private_accesses += 1
        else:  # Shared
            self.stats.shared_accesses += 1

        if is_load:  # Load Hit
            self._touch_lru(line)
            return (True, HIT_CYCLES)
        else:  # Store Hit
            if line.state in (LineState.MODIFIED, LineState.EXCLUSIVE):
                line.state = LineState.MODIFIED
                self._touch_lru(line)
                return (True, HIT_CYCLES)
            elif line.state == LineState.SHARED:
                # Upgrade S -> M. We must stall and wait for the bus.
                block_addr = addr & ~((1 << self.cfg.off_bits) - 1)
                self.pending_misses[block_addr] = line
                line.state = LineState.INVALID
                self.bus.push_request(self.core_id, BusXact.BusRdX, addr)
                return (False, 0)

    def complete_request(self, addr: int, final_state: str):
        """Callback from the bus when our memory request is done."""
        tag, idx, off = self._addr_fields(addr)
        block_addr = addr & ~((1 << self.cfg.off_bits) - 1)

        if block_addr not in self.pending_misses:
            # This should *really* not happen now
            raise Exception(
                f"Completed request for {addr:x} but no pending miss found.")

        # 1. Retrieve the reserved line (either a new victim or the old S line)
        line_to_fill = self.pending_misses.pop(block_addr)

        # 2. Fill it with the new tag and state
        line_to_fill.tag = tag
        line_to_fill.state = final_state
        self._touch_lru(line_to_fill)

        # 3. Un-stall the CPU
        self.cpu.un_stall()

    def snoop(self, xact_type: str, addr: int) -> Optional[str]:
        """Process a snooped transaction. Return line state if we have it."""
        tag, idx, off = self._addr_fields(addr)
        line = self._find_line(idx, tag)
        if not line:
            return None

        original_state = line.state
        if xact_type == BusXact.BusRd:
            if line.state in (LineState.EXCLUSIVE, LineState.MODIFIED):
                line.state = LineState.SHARED
        elif xact_type == BusXact.BusRdX:
            if line.state in (LineState.SHARED, LineState.EXCLUSIVE, LineState.MODIFIED):
                line.state = LineState.INVALID
                self.g_stats.invalidations_or_updates += 1
        return original_state

    def _reconstruct_addr(self, tag: int, idx: int) -> int:
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
    l1: L1Cache  # Will be set by Simulator after init
    stats: CacheStats
    tr: TraceReader
    sim: 'Simulator'

    # Core state
    stalled: bool = False
    stall_start_time: int = 0
    finish_time: int = 0

    def is_finished(self) -> bool:
        # Core is finished if its trace is done AND it's not stalled
        # waiting for a final memory operation.
        return self.tr.finished and not self.stalled

    def execute(self):
        """
        This is the main event handler for the CPU.
        It processes one instruction and schedules its next 'execute' event.
        """
        if self.stalled:
            return  # Waiting for un_stall()
        if self.tr.finished:
            return  # This core is done

        label, value = self.tr.current_instr

        if label == OTHER:
            self.stats.compute_cycles += value
            self.tr.advance()
            delay = value
            if self.tr.finished:
                # This was the last op. Record finish time.
                self.finish_time = self.sim.current_time + delay
            else:
                # Schedule next execution after compute delay
                self.sim.schedule(delay, self.execute)

        else:  # LOAD or STORE
            is_hit, latency = self.l1.access(value, label)

            if is_hit:
                self.tr.advance()
                if self.tr.finished:
                    # This was the last op. Record finish time.
                    self.finish_time = self.sim.current_time + latency
                else:
                    # Schedule next execution after hit latency
                    self.sim.schedule(latency, self.execute)
            else:
                # Miss! Stall the CPU and record when the stall began
                self.stalled = True
                self.stall_start_time = self.sim.current_time

    def un_stall(self):
        """Callback from L1Cache when a request is complete."""
        if not self.stalled:
            return  # Should not happen

        self.stalled = False

        # Calculate and add idle time
        idle_duration = self.sim.current_time - self.stall_start_time
        self.stats.idle_cycles += idle_duration

        # We missed on this instruction, so now we advance past it
        self.tr.advance()

        if self.tr.finished:
            # This was the last op. Record finish time.
            self.finish_time = self.sim.current_time
        else:
            # Schedule the next instruction to execute *now*
            self.sim.schedule(0, self.execute)


class Simulator:
    def __init__(self, protocol: str, benchmark: str, cfg: CacheConfig, traces_root: Optional[Path]):
        self.current_time = 0
        self.event_queue = []
        self.event_id_counter = 0  # Tie-breaker for events at same time

        self.g_stats = GlobalStats()
        self.bus = Bus(self.g_stats, self)
        self.cpus: List[CPU] = []

        trace_paths = self.resolve_trace_paths(benchmark, traces_root)
        print(f"Found {len(trace_paths)} trace file(s) for '{benchmark}'.")

        for i, path in enumerate(trace_paths):
            stats = CacheStats()
            # Create CPU and Cache, then link them
            cpu = CPU(i, None, stats, TraceReader(path), self)
            cache = L1Cache(i, cfg, stats, self.bus, self.g_stats, cpu)
            cpu.l1 = cache  # Set the back-link

            self.bus.add_snooper(cache)
            self.cpus.append(cpu)

            self.schedule(0, cpu.execute)

    def schedule(self, delay: int, callback: callable, *args):
        """Add an event to the priority queue."""
        assert delay >= 0
        event_time = self.current_time + delay

        # Use a counter for stable sorting if times are equal
        event = (event_time, self.event_id_counter, callback, args)
        self.event_id_counter += 1

        heapq.heappush(self.event_queue, event)

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
        """Main discrete event simulation loop."""
        while self.event_queue:
            # Get the next event
            (time, _id, callback, args) = heapq.heappop(self.event_queue)

            # Advance global time to the event's time
            self.current_time = time

            # Execute the event
            callback(*args)

        # Simulation is over when the event queue is empty
        print(f"Simulation finished at time {self.current_time}")

    def print_results(self, protocol, benchmark, cfg):
        print("==== RESULTS ====")
        print(f"protocol={protocol}")
        print(f"benchmark={benchmark}")
        print(
            f"cache_size_bytes={cfg.size_bytes} assoc={cfg.assoc} block_bytes={cfg.block_bytes}")

        overall_max_cycles = 0
        if self.cpus:
            overall_max_cycles = max(c.finish_time for c in self.cpus)
        print(f"overall_exec_cycles={overall_max_cycles}")

        for i, cpu in enumerate(self.cpus):
            stats = cpu.stats
            print(f"--- Core {i} Stats ---")
            print(f"core{i}_exec_cycles={cpu.finish_time}")
            print(f"core{i}_compute_cycles={stats.compute_cycles}")
            print(f"core{i}_loads={stats.loads} core{i}_stores={stats.stores}")
            print(f"core{i}_hits={stats.hits} core{i}_misses={stats.misses}")
            print(f"core{i}_idle_cycles={stats.idle_cycles}")
            print(f"core{i}_private_accesses={stats.private_accesses}")
            print(f"core{i}_shared_accesses={stats.shared_accesses}")

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

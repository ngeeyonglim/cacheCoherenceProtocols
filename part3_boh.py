#!/usr/bin/env python

# Usage (examples):
#   python coherence.py MESI bodytrack 1024 1 16 --num-cores 4
#   python coherence.py Dragon blackscholes 4096 2 32 --num-cores 2

import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Literal

WORD_BYTES = 4  # 32-bit word
HIT_CYCLES = 1
MEM_FETCH_BLOCK_CYCLES = 100       # miss service from memory
DIRTY_EVICT_TO_MEM_CYCLES = 100    # writeback dirty block to memory
C2C_WORD_CYCLES = 2

LOAD = 0
STORE = 1
OTHER = 2


class MESI:
    INVALID = "I"
    SHARED = "S"
    EXCLUSIVE = "E"
    MODIFIED = "M"


class Dragon:
    INVALID = "I"
    EXCLUSIVE = "E"       # Clean, sole copy
    SHARED_CLEAN = "Sc"   # Clean, shared
    SHARED_MODIFIED = "Sm"  # Dirty, shared (owner)
    MODIFIED = "M"        # Dirty, sole copy


class BusTxnType:
    BusRd = "BusRd"   # read miss (shared if others have it)
    BusRdX = "BusRdX"  # read for ownership (write miss)
    BusUpg = "BusUpgr"  # no need to fetch data from other caches
    BusWB = "BusWB"   # FLUSH
    BusUpd = "BusUpd"  # Dragon write broadcast


@dataclass
class CacheLine:
    tag: int = -1
    state: str = MESI.INVALID
    lru_tick: int = 0  # larger is more recent


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


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    loads: int = 0
    stores: int = 0
    compute_cycles: int = 0
    idle_cycles: int = 0
    # data traffic only (no addresses)
    bus_data_bytes: int = 0
    # BusUpgr/BusRdX-driven peer invalidations or BusUpd
    invalidations_or_updates: int = 0
    private_accesses: int = 0          # accesses to E/M, and installs of E/M
    shared_accesses: int = 0           # accesses to S/Sc/Sm, and installs of S/Sc/Sm
    mem_data_bytes: int = 0            # memory-sourced bytes
    c2c_data_bytes: int = 0            # c2c-sourced bytes


class LRU:
    def __init__(self): self.t = 0
    def tick(self) -> int: self.t += 1; return self.t


class TraceReader:
    """
    Reads a single core's trace file: lines 'Label Value'.
    OTHER lines provide compute cycles (Value can be hex or dec).
    """

    def __init__(self, path: Path):
        self.path = path
        self._lines = None
        self._i = 0

    def __iter__(self):
        with self.path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split()
                label = int(a)
                # treat value as 32-bit even if top bits zero
                val = int(b, 16 if b.lower().startswith(
                    "0x") else 10) & 0xffffffff
                yield label, val


@dataclass
class BusTxn:
    ttype: str
    addr: int
    src_core: int


@dataclass
class BusResp:
    shared: bool
    owner_supplies: bool
    inval_count: int
    data_source: Literal['mem', 'c2c']


class Snooper:
    """Interface implemented by each L1 for snoop callbacks."""

    def on_snoop(self, txn: BusTxn) -> Tuple[bool, bool, bool, str]:
        """
        Return (had_line, supplied_data, invalidated_or_updated)
        - had_line: cache had the block in {S,E,M,Sc,Sm}
        - supplied_data: true if we flushed the block on this txn
        - invalidated_or_updated: true if we transitioned to I (MESI)
                                  or updated block (Dragon) due to this txn
        - prev_state: state BEFORE applying the snoop
        """
        raise NotImplementedError


class Bus:
    """
    Single shared snooping bus, FCFS. One txn at a time.
    simple cycle timing: requester starts at max(core.time, bus.free_at).
    """

    def __init__(self, cfg: CacheConfig, snoopers: List[Snooper],
                 stats_per_core: List[CacheStats], protocol: str):
        self.cfg = cfg
        self.snoopers = snoopers
        self.stats_per_core = stats_per_core
        self.free_at: int = 0  # global time when bus becomes free
        self.protocol = protocol

    def _block_words(self) -> int:
        return self.cfg.block_bytes // WORD_BYTES

    def request(self, txn: BusTxn, start_time: int) -> Tuple[BusResp, int, int]:
        """
        Execute txn starting no earlier than start_time & bus.free_at.
        Broadcast to all other caches; compute shared/owner_supplies/invals.
        Return (resp, latency, bus_start_time).
        """
        t0 = max(start_time, self.free_at)
        shared = False
        owner_supplies = False
        inval_or_update_count = 0

        dirty_owner_core = None

        # checks all caches for whether they have the relevant data
        for cid, snp in enumerate(self.snoopers):
            if cid == txn.src_core:
                continue

            had, supplied, peer_changed, prev = snp.on_snoop(
                txn)  # check behaviour of the other caches
            if had:
                shared = True
            if supplied:
                owner_supplies = True
            if peer_changed:
                inval_or_update_count += 1

            if (self.protocol == "MESI" and prev == MESI.MODIFIED) or \
               (self.protocol == "Dragon" and prev in (Dragon.MODIFIED, Dragon.SHARED_MODIFIED)):
                dirty_owner_core = cid

        # Decide data source + latency and account data traffic
        if txn.ttype == BusTxnType.BusUpg:  # busupg for S -> M (MESI)
            latency = 1  # control only
            data_source = 'mem'

        elif txn.ttype == BusTxnType.BusUpd:
            latency = C2C_WORD_CYCLES
            data_source = 'c2c'
            # Attribute word write bytes to the updating core
            self.stats_per_core[txn.src_core].bus_data_bytes += WORD_BYTES
            self.stats_per_core[txn.src_core].c2c_data_bytes += WORD_BYTES

        elif txn.ttype == BusTxnType.BusWB:
            latency = DIRTY_EVICT_TO_MEM_CYCLES
            data_source = 'mem'
            # Attribute writeback bytes to the evicting core
            self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
            self.stats_per_core[txn.src_core].mem_data_bytes += self.cfg.block_bytes
        else:
            # Data-carrying
            if txn.ttype == BusTxnType.BusRd:
                if self.protocol == "MESI":
                    if owner_supplies:
                        words = self._block_words()
                        if dirty_owner_core is not None:
                            # Explicit write-back to memory (100), attributed to dirty owner
                            latency = DIRTY_EVICT_TO_MEM_CYCLES
                            self.stats_per_core[dirty_owner_core].bus_data_bytes += self.cfg.block_bytes
                            self.stats_per_core[dirty_owner_core].mem_data_bytes += self.cfg.block_bytes
                            # Then C2C transfer to requester (2N), attributed to requester
                            latency += C2C_WORD_CYCLES * words
                            self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                            self.stats_per_core[txn.src_core].c2c_data_bytes += self.cfg.block_bytes
                            data_source = 'c2c'
                        else:
                            # Supplier was E, C2C only
                            latency = C2C_WORD_CYCLES * words
                            data_source = 'c2c'
                            self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                            self.stats_per_core[txn.src_core].c2c_data_bytes += self.cfg.block_bytes
                    else:
                        # No supplier, fetch from memory (100)
                        latency = MEM_FETCH_BLOCK_CYCLES
                        data_source = 'mem'
                        self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                        self.stats_per_core[txn.src_core].mem_data_bytes += self.cfg.block_bytes

                elif self.protocol == "Dragon":
                    words = self._block_words()
                    if owner_supplies:  # Supplied by E, M, or Sm
                        latency = C2C_WORD_CYCLES * words
                        data_source = 'c2c'
                        self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                        self.stats_per_core[txn.src_core].c2c_data_bytes += self.cfg.block_bytes
                    else:  # Supplied by memory (no owner, or only Sc)
                        latency = MEM_FETCH_BLOCK_CYCLES
                        data_source = 'mem'
                        self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                        self.stats_per_core[txn.src_core].mem_data_bytes += self.cfg.block_bytes

            elif txn.ttype == BusTxnType.BusRdX:
                words = self._block_words()
                if owner_supplies:
                    latency = C2C_WORD_CYCLES * words
                    data_source = 'c2c'
                    self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                    self.stats_per_core[txn.src_core].c2c_data_bytes += self.cfg.block_bytes
                else:
                    latency = MEM_FETCH_BLOCK_CYCLES
                    data_source = 'mem'
                    self.stats_per_core[txn.src_core].bus_data_bytes += self.cfg.block_bytes
                    self.stats_per_core[txn.src_core].mem_data_bytes += self.cfg.block_bytes

        # Reserve bus and finish
        total_latency = latency  # bus occupancy for payload/control
        self.free_at = t0 + total_latency
        resp = BusResp(shared=shared, owner_supplies=owner_supplies,
                       inval_count=inval_or_update_count, data_source=data_source)
        return resp, total_latency, t0


class L1Cache(Snooper):
    def __init__(self, cfg: CacheConfig, stats: CacheStats, core_id: int,
                 bus: Bus, protocol: str):
        self.cfg = cfg
        self.stats = stats
        self.core_id = core_id
        self.bus = bus
        self.lru = LRU()
        self.sets = [CacheSet([CacheLine() for _ in range(cfg.assoc)])
                     for _ in range(cfg.num_sets or 1)]
        self.protocol = protocol

    def _addr_fields(self, addr: int) -> Tuple[int, int, int]:
        off_mask = (1 << self.cfg.off_bits) - 1
        idx_mask = (1 << self.cfg.idx_bits) - 1
        off = addr & off_mask
        idx = (addr >> self.cfg.off_bits) & idx_mask if self.cfg.idx_bits > 0 else 0
        tag = addr >> (self.cfg.off_bits + self.cfg.idx_bits)
        return tag, idx, off

    def _reconstruct_addr(self, tag: int, idx: int) -> int:
        return (tag << (self.cfg.idx_bits + self.cfg.off_bits)) | (idx << self.cfg.off_bits)

    def _find_line(self, idx: int, tag: int) -> Optional[CacheLine]:
        for line in self.sets[idx].lines:
            if line.state != MESI.INVALID and line.tag == tag:
                return line
        return None

    def _select_victim(self, idx: int) -> CacheLine:
        for line in self.sets[idx].lines:
            if line.state == MESI.INVALID:
                return line
        return min(self.sets[idx].lines, key=lambda ln: ln.lru_tick)

    def _touch_lru(self, line: CacheLine) -> None:
        line.lru_tick = self.lru.tick()

    def on_pr_rd(self, addr: int, core_time: int) -> int:
        self.stats.loads += 1
        tag, idx, _ = self._addr_fields(addr)
        line = self._find_line(idx, tag)

        if self.protocol == "MESI":
            if line:
                # Hit in S/E/M
                self.stats.hits += 1
                if line.state in (MESI.EXCLUSIVE, MESI.MODIFIED):
                    self.stats.private_accesses += 1
                else:
                    self.stats.shared_accesses += 1
                self._touch_lru(line)
                return HIT_CYCLES

            # Miss: may need to evict
            self.stats.misses += 1
            victim = self._select_victim(idx)
            latency = 0

            # Evict M -> BusWB (to memory)
            if victim.state == MESI.MODIFIED:
                victim_addr = self._reconstruct_addr(victim.tag, idx)
                txn = BusTxn(BusTxnType.BusWB, addr=victim_addr,
                             src_core=self.core_id)
                _, wb_lat, _ = self.bus.request(
                    txn, start_time=core_time + latency)
                latency += wb_lat

            # Install incoming block via BusRd
            txn = BusTxn(BusTxnType.BusRd, addr=addr, src_core=self.core_id)
            issued_at = core_time + latency
            resp, bus_lat, t0 = self.bus.request(
                txn, start_time=core_time + latency)
            wait = t0 - issued_at
            latency += bus_lat + wait

            # Fill and set state: E if not shared, else S
            victim.tag = tag
            victim.state = MESI.SHARED if resp.shared else MESI.EXCLUSIVE
            self._touch_lru(victim)

            # Access itself (1 cycle busy)
            if victim.state == MESI.EXCLUSIVE:
                self.stats.private_accesses += 1
            else:
                self.stats.shared_accesses += 1

            return latency + HIT_CYCLES

        elif self.protocol == "Dragon":
            if line:
                # Hit in E, M, Sc, Sm
                self.stats.hits += 1
                if line.state in (Dragon.EXCLUSIVE, Dragon.MODIFIED):
                    self.stats.private_accesses += 1
                else:  # Sc or Sm
                    self.stats.shared_accesses += 1
                self._touch_lru(line)
                return HIT_CYCLES

            # Miss: may need to evict
            self.stats.misses += 1
            victim = self._select_victim(idx)
            latency = 0

            # Evict M or Sm -> BusWB (to memory)
            if victim.state in (Dragon.MODIFIED, Dragon.SHARED_MODIFIED):
                victim_addr = self._reconstruct_addr(victim.tag, idx)
                txn = BusTxn(BusTxnType.BusWB, addr=victim_addr,
                             src_core=self.core_id)
                _, wb_lat, _ = self.bus.request(
                    txn, start_time=core_time + latency)
                latency += wb_lat

            # Install incoming block via BusRd
            txn = BusTxn(BusTxnType.BusRd, addr=addr, src_core=self.core_id)
            issued_at = core_time + latency
            resp, bus_lat, t0 = self.bus.request(
                txn, start_time=core_time + latency)
            wait = t0 - issued_at
            latency += bus_lat + wait

            # Fill and set state: E if not shared, else Sc
            victim.tag = tag
            victim.state = Dragon.SHARED_CLEAN if resp.shared else Dragon.EXCLUSIVE
            self._touch_lru(victim)

            # Access itself (1 cycle busy)
            if victim.state == Dragon.EXCLUSIVE:
                self.stats.private_accesses += 1
            else:
                self.stats.shared_accesses += 1

            return latency + HIT_CYCLES

        return 0  # Should not reach

    def on_pr_wr(self, addr: int, core_time: int) -> int:
        self.stats.stores += 1
        tag, idx, _ = self._addr_fields(addr)
        line = self._find_line(idx, tag)

        if self.protocol == "MESI":
            if line:
                # Hit paths
                self.stats.hits += 1
                if line.state == MESI.MODIFIED:
                    self._touch_lru(line)
                    self.stats.private_accesses += 1
                    return HIT_CYCLES
                if line.state == MESI.EXCLUSIVE:
                    line.state = MESI.MODIFIED  # silent upgrade
                    self._touch_lru(line)
                    self.stats.private_accesses += 1
                    return HIT_CYCLES
                if line.state == MESI.SHARED:
                    # Need BusUpgr to invalidate peer S
                    issued_at = core_time
                    txn = BusTxn(BusTxnType.BusUpg, addr=addr,
                                 src_core=self.core_id)
                    resp, bus_lat, t0 = self.bus.request(
                        txn, start_time=core_time)
                    wait = t0 - issued_at
                    self.stats.invalidations_or_updates += resp.inval_count
                    line.state = MESI.MODIFIED
                    self._touch_lru(line)
                    self.stats.private_accesses += 1
                    return bus_lat + HIT_CYCLES + wait

            # Miss in I: write-allocate via BusRdX
            self.stats.misses += 1
            victim = self._select_victim(idx)
            latency = 0
            # Evict M victim if needed
            if victim.state == MESI.MODIFIED:
                victim_addr = self._reconstruct_addr(victim.tag, idx)
                txn = BusTxn(BusTxnType.BusWB, addr=victim_addr,
                             src_core=self.core_id)
                _, wb_lat, _ = self.bus.request(
                    txn, start_time=core_time + latency)
                latency += wb_lat

            issued_at = core_time + latency
            txn = BusTxn(BusTxnType.BusRdX, addr=addr, src_core=self.core_id)
            resp, bus_lat, t0 = self.bus.request(
                txn, start_time=core_time + latency)
            wait = t0 - issued_at
            latency += bus_lat + wait
            self.stats.invalidations_or_updates += resp.inval_count

            victim.tag = tag
            victim.state = MESI.MODIFIED
            self._touch_lru(victim)
            self.stats.private_accesses += 1
            return latency + HIT_CYCLES

        elif self.protocol == "Dragon":
            if line:
                # Hit paths
                self.stats.hits += 1
                latency = 0

                if line.state == Dragon.MODIFIED:
                    # M -> M (silent)
                    self._touch_lru(line)
                    self.stats.private_accesses += 1
                    return HIT_CYCLES

                if line.state == Dragon.EXCLUSIVE:
                    # E -> M (silent)
                    line.state = Dragon.MODIFIED
                    self._touch_lru(line)
                    self.stats.private_accesses += 1
                    return HIT_CYCLES

                # Shared hit: Need to broadcast BusUpd
                issued_at = core_time
                txn = BusTxn(BusTxnType.BusUpd, addr=addr,
                             src_core=self.core_id)
                resp, bus_lat, t0 = self.bus.request(txn, start_time=core_time)
                wait = t0 - issued_at
                latency = bus_lat + wait
                self.stats.invalidations_or_updates += resp.inval_count

                if resp.shared:
                    # Others still have it, become/stay Sm
                    line.state = Dragon.SHARED_MODIFIED
                else:
                    # No one else has it, become M
                    line.state = Dragon.MODIFIED

                self._touch_lru(line)
                self.stats.shared_accesses += 1
                return latency + HIT_CYCLES

            # Miss in I: write-allocate
            self.stats.misses += 1
            victim = self._select_victim(idx)
            latency = 0

            # Evict M or Sm victim if needed
            if victim.state in (Dragon.MODIFIED, Dragon.SHARED_MODIFIED):
                victim_addr = self._reconstruct_addr(victim.tag, idx)
                txn_wb = BusTxn(BusTxnType.BusWB, addr=victim_addr,
                                src_core=self.core_id)
                _, wb_lat, _ = self.bus.request(
                    txn_wb, start_time=core_time + latency)
                latency += wb_lat

            # 1. Fetch block via BusRd
            issued_at_rd = core_time + latency
            txn_rd = BusTxn(BusTxnType.BusRd, addr=addr, src_core=self.core_id)
            resp_rd, bus_lat_rd, t0_rd = self.bus.request(
                txn_rd, start_time=issued_at_rd)
            wait_rd = t0_rd - issued_at_rd
            latency += bus_lat_rd + wait_rd

            victim.tag = tag
            self._touch_lru(victim)

            if resp_rd.shared:
                # Sc -> Sm and issue BusUpd
                victim.state = Dragon.SHARED_MODIFIED
                self.stats.shared_accesses += 1

                issued_at_upd = core_time + latency
                txn_upd = BusTxn(BusTxnType.BusUpd, addr=addr,
                                 src_core=self.core_id)
                resp_upd, bus_lat_upd, t0_upd = self.bus.request(
                    txn_upd, start_time=issued_at_upd)
                wait_upd = t0_upd - issued_at_upd
                latency += bus_lat_upd + wait_upd
                self.stats.invalidations_or_updates += resp_upd.inval_count
            else:
                # transition to M (silent)
                victim.state = Dragon.MODIFIED
                self.stats.private_accesses += 1

            return latency + HIT_CYCLES

        return 0  # Should not reach

    def on_snoop(self, txn: BusTxn) -> Tuple[bool, bool, bool, str]:
        tag, idx, _ = self._addr_fields(txn.addr)
        line = self._find_line(idx, tag)
        if not line:
            return (False, False, False, MESI.INVALID)

        # Protocol-specific snoop logic
        if self.protocol == "MESI":
            supplied = False
            invalidated = False
            prev = line.state

            if txn.ttype == BusTxnType.BusRd:
                if line.state == MESI.MODIFIED:
                    # Flush data to bus; M->S
                    supplied = True
                    line.state = MESI.SHARED
                elif line.state == MESI.EXCLUSIVE:
                    # Flush data to bus; E->S (C2C enabled per user)
                    supplied = True
                    line.state = MESI.SHARED
                # S stays S
                # I no-op
            elif txn.ttype == BusTxnType.BusRdX:
                if line.state in (MESI.MODIFIED, MESI.EXCLUSIVE):
                    supplied = True
                    line.state = MESI.INVALID
                    invalidated = True
                elif line.state == MESI.SHARED:
                    line.state = MESI.INVALID
                    invalidated = True
            elif txn.ttype == BusTxnType.BusUpg:
                if line.state == MESI.SHARED:
                    line.state = MESI.INVALID
                    invalidated = True
            # BusWB is ignored by peers
            return (True, supplied, invalidated, prev)

        elif self.protocol == "Dragon":
            supplied = False
            updated = False  # Use third return val for "updated"
            prev = line.state

            if txn.ttype == BusTxnType.BusRd:
                if line.state == Dragon.MODIFIED:
                    # M -> Sm, supply data
                    supplied = True
                    line.state = Dragon.SHARED_MODIFIED
                elif line.state == Dragon.EXCLUSIVE:
                    # E -> Sc, supply data
                    supplied = True
                    line.state = Dragon.SHARED_CLEAN
                elif line.state == Dragon.SHARED_MODIFIED:
                    # Sm -> Sm, supply data
                    supplied = True

            elif txn.ttype == BusTxnType.BusUpd:
                if line.state == Dragon.SHARED_CLEAN:
                    # Sc -> Sc, update word
                    updated = True
                elif line.state == Dragon.SHARED_MODIFIED:
                    # A peer (Sc) wrote and is becoming the new Sm.
                    # We update our data and relinquish ownership.
                    line.state = Dragon.SHARED_CLEAN
                    updated = True
                # Sm, M, E should not see BusUpd for their own lines

            # BusRdX/BusUpg are MESI-only, BusWB is ignored
            return (True, supplied, updated, prev)

        return (False, False, False, MESI.INVALID)  # Should not reach


@dataclass
class SingleCoreCPU:
    l1: L1Cache
    stats: CacheStats
    time: int = 0   # core-local cycle counter


class Simulator:
    """
    Manages N blocking cores with a shared bus.
    Cores advance independently; bus enforces FCFS on transactions.
    """

    def __init__(self, protocol: str, benchmark_base: str, cfg: CacheConfig,
                 traces_root: Optional[Path], num_cores: int):
        self.protocol = protocol
        self.benchmark = benchmark_base
        self.cfg = cfg
        self.traces_root = traces_root
        self.num_cores = num_cores

        self.stats: List[CacheStats] = [CacheStats() for _ in range(num_cores)]
        # temporary bus placeholder for L1 constructors; we replace after
        self.bus: Optional[Bus] = None

        # Build L1s with a bus stub first
        self.cores: List[SingleCoreCPU] = []
        l1s: List[L1Cache] = []
        for cid in range(num_cores):
            # stub bus; replaced below
            stub_bus = Bus(cfg, [], self.stats, protocol)
            l1 = L1Cache(cfg, self.stats[cid], cid, stub_bus, protocol)
            l1s.append(l1)

        # Real bus wired with snoopers
        self.bus = Bus(cfg, l1s, self.stats, protocol)

        # Replace bus refs in L1s and create cores
        self.cores = [SingleCoreCPU(
            l1=l1s[cid], stats=self.stats[cid], time=0) for cid in range(num_cores)]
        for l1 in l1s:
            l1.bus = self.bus

        # Traces
        self.traces: List[List[Tuple[int, int]]] = []
        for cid in range(num_cores):
            path = self._resolve_trace_path(benchmark_base, traces_root, cid)
            self.traces.append(list(TraceReader(path)))
        self.ptrs: List[int] = [0]*num_cores  # per-core next event index

    def _resolve_trace_path(self, benchmark_base: str, traces_root: Optional[Path], core_id: int) -> Path:
        """
        Find <benchmark>_<core_id>.data
        """
        candidates = []
        if traces_root:
            candidates.append(traces_root)
        candidates += [Path.cwd(),
                       Path("/home/course/cs4223/assignments/assignment2")]
        fname = f"{benchmark_base}_{core_id}.data"
        for root in candidates:
            p = root / fname
            if p.exists():
                return p
        # Fall back to _0.data for single-core traces
        if core_id == 0:
            fname_0 = f"{benchmark_base}_0.data"
            for root in candidates:
                p = root / fname_0
                if p.exists():
                    return p
        raise FileNotFoundError(
            f"Could not find {fname} (or {fname_0}) under {candidates}")

    def _core_has_work(self, cid: int) -> bool:
        return self.ptrs[cid] < len(self.traces[cid])

    def _op_str(self, label: int) -> str:
        return "LOAD" if label == LOAD else ("STORE" if label == STORE else "OTHER")

    def _print_pre_action(self, next_cid: int, label: int, value: int) -> None:
        times = " ".join(f"c{i}={core.time}" for i,
                         core in enumerate(self.cores))
        # show hex for addresses (LOAD/STORE), decimal for OTHER (compute)
        val_str = f"{value:#010x}" if label != OTHER else str(value)
        print(
            f"[pre] bus.free_at={self.bus.free_at} | {times} | next=core{next_cid} {self._op_str(label)} {val_str}")

    def run(self):
        # Simple event loop: pick the earliest-time core that still has an event
        while True:
            candidates = [(cid, self.cores[cid].time)
                          for cid in range(self.num_cores)
                          if self._core_has_work(cid)]
            if not candidates:
                break
            # will always pick the core with the lowest local time, so that process order makes sense
            candidates.sort(key=lambda x: (x[1], x[0]))
            cid, _ = candidates[0]
            core = self.cores[cid]
            label, value = self.traces[cid][self.ptrs[cid]]

            # self._print_pre_action(cid, label, value)
            self.ptrs[cid] += 1

            if label == OTHER:
                core.stats.compute_cycles += value
                core.time += value
                continue

            # Memory access; cache is blocking: compute full latency
            if label == LOAD:
                latency = core.l1.on_pr_rd(value, core.time)
            elif label == STORE:
                latency = core.l1.on_pr_wr(value, core.time)
            else:
                raise ValueError(f"Unknown label {label}")

            # Idle cycles: (latency - 1) beyond the 1-cycle access itself
            # A 1-cycle hit has latency=1, idle=0.
            # A >1-cycle miss has latency=L, idle=(L-1).
            # The core time advances by L.
            idle = max(0, latency - 1)
            core.stats.idle_cycles += idle + 1
            core.time += latency


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} PB"


def main():
    ap = argparse.ArgumentParser(
        description="Trace-driven cache coherence simulator (MESI/Dragon)")
    ap.add_argument("protocol", choices=[
                    "MESI", "Dragon"], help="Coherence protocol (MESI or Dragon).")
    ap.add_argument(
        "input_file", help="Benchmark base name (e.g., bodytrack, blackscholes, fluidanimate)")
    ap.add_argument("cache_size", default=4096,
                    type=int, help="L1 size in bytes")
    ap.add_argument("associativity", default=2,
                    type=int, help="L1 associativity")
    ap.add_argument("block_size", default=32, type=int,
                    help="L1 block size in bytes")
    ap.add_argument("--traces-root", type=Path, default=None,
                    help="Directory with *_N.data files")
    ap.add_argument("--num-cores", type=int, default=4,
                    help="Number of cores to simulate")
    args = ap.parse_args()

    cfg = CacheConfig(size_bytes=args.cache_size,
                      assoc=args.associativity, block_bytes=args.block_size)

    # Build & run
    sim = Simulator(protocol=args.protocol, benchmark_base=args.input_file,
                    cfg=cfg, traces_root=args.traces_root, num_cores=args.num_cores)

    print(f"Found {len(sim.traces)} trace file(s) for '{args.input_file}'.")

    sim.run()

    print("==== RESULTS ====")
    print(f"protocol={args.protocol}")
    print(f"benchmark={args.input_file}")
    print(
        f"cache_size_bytes={cfg.size_bytes} assoc={cfg.assoc} block_bytes={cfg.block_bytes}")

    overall_exec_cycles = max(
        core.time for core in sim.cores) if sim.cores else 0
    print(f"overall_exec_cycles={overall_exec_cycles}")

    # Per-core stats blocks
    for i, core in enumerate(sim.cores):
        st = core.stats
        print(f"--- Core {i} Stats ---")
        print(f"core{i}_exec_cycles={core.time}")
        print(f"core{i}_compute_cycles={st.compute_cycles}")
        print(f"core{i}_loads={st.loads} core{i}_stores={st.stores}")
        print(f"core{i}_hits={st.hits} core{i}_misses={st.misses}")
        print(f"core{i}_idle_cycles={st.idle_cycles}")
        print(f"core{i}_private_accesses={st.private_accesses}")
        print(f"core{i}_shared_accesses={st.shared_accesses}")

    # Global bus stats = sum over cores (your model attributes bytes to the requester/evictor core)
    total_bus_bytes = sum(c.stats.bus_data_bytes for c in sim.cores)
    total_invals_or_updates = sum(
        c.stats.invalidations_or_updates for c in sim.cores)

    print("--- Global Bus Stats ---")
    print(
        f"bus_data_traffic_bytes={total_bus_bytes} ({human_bytes(total_bus_bytes)})")
    print(f"bus_invals_or_updates={total_invals_or_updates}")


if __name__ == "__main__":
    main()

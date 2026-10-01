"""Frozen-equipment maintenance window planning.

A maintenance window is more than a calendar slot.  Taking a freezer out of
service is only feasible when all of the following hold simultaneously:

* N+1 redundancy: while the unit is down, every affected power circuit still
  keeps its minimum number of healthy units running;
* split power feeds: units sharing a circuit are counted together, units on
  separate circuits are independent;
* relocation time: every cargo lot on the unit can be moved into surviving
  destination capacity *before* work starts, and the move fits the buffer;
* post-maintenance reconfirmation: a requalification task finishes before the
  window deadline, so the unit only returns to service once re-qualified.

Scheduling a window atomically locks the equipment, the inventory transfers
and the crew/review tasks.  When a window is squeezed out the planner keeps
every feasible part (already committed windows, relocatable lots) and reports
exactly which cargo lots must be postponed and why.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable, Sequence

ROLE_RELOCATE = "relocate"
ROLE_MAINTAIN = "maintain"
ROLE_REQUALIFY = "requalify"

PHASE_RELOCATE = "relocate"
PHASE_MAINTAIN = "maintain"
PHASE_REQUALIFY = "requalify"

# Reasons a window or lot cannot be committed.
R_TIME = "no_feasible_time"
R_EQUIPMENT = "equipment_busy"
R_REDUNDANCY = "redundancy"
R_RESOURCE = "missing_resource"
R_RELOCATION = "relocation"
R_NO_DESTINATION = "no_allowed_destination"
R_CAPACITY = "destination_capacity"
R_MOVE_TIME = "relocation_time"


def utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    from datetime import timezone
    return v.astimezone(timezone.utc)


@dataclass(frozen=True)
class FreezerUnit:
    key: str
    zone: str
    circuit: str
    capacity: Decimal
    available: bool = True

    def validate(self) -> None:
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if not self.zone or not self.circuit:
            raise ValueError("zone and circuit required")


@dataclass(frozen=True)
class CargoLot:
    key: str
    freezer_key: str
    quantity: Decimal
    allowed_freezers: frozenset[str]
    relocation_minutes: int = 0
    deferrable: bool = True

    def validate(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.relocation_minutes < 0:
            raise ValueError("relocation_minutes must be non-negative")


@dataclass(frozen=True)
class MaintenanceWindow:
    key: str
    freezer_key: str
    earliest: datetime
    latest: datetime
    duration: timedelta
    relocate_buffer: timedelta
    requalify_duration: timedelta
    priority: int = 0
    partial_relocation_allowed: bool = False

    def validate(self) -> None:
        if self.duration <= timedelta():
            raise ValueError("duration must be positive")
        if self.relocate_buffer < timedelta():
            raise ValueError("relocate_buffer must be non-negative")
        if self.requalify_duration < timedelta():
            raise ValueError("requalify_duration must be non-negative")
        if utc(self.latest) <= utc(self.earliest):
            raise ValueError("invalid scheduling window")


@dataclass(frozen=True)
class CrewResource:
    """A crew member / technician qualified for one or more phase roles."""

    key: str
    zone: str
    available_from: datetime
    available_until: datetime
    roles: frozenset[str]

    def validate(self) -> None:
        if not self.roles:
            raise ValueError("resource needs at least one role")
        if utc(self.available_until) <= utc(self.available_from):
            raise ValueError("invalid resource availability")


@dataclass(frozen=True)
class ScheduledTask:
    window_key: str
    phase: str
    resource_key: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class RelocationMove:
    window_key: str
    lot_key: str
    source: str
    destination: str
    quantity: Decimal


@dataclass(frozen=True)
class Lock:
    """An atomic reservation created when a window is committed."""

    window_key: str
    kind: str  # freezer | lot | resource
    target: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class DeferredLot:
    lot_key: str
    window_key: str
    quantity: Decimal
    reason: str


@dataclass(frozen=True)
class WindowResult:
    window_key: str
    scheduled: bool
    tasks: tuple[ScheduledTask, ...] = ()
    moves: tuple[RelocationMove, ...] = ()
    locks: tuple[Lock, ...] = ()
    deferred_lots: tuple[DeferredLot, ...] = ()
    reason: str = ""

    @property
    def start(self) -> datetime | None:
        ordered = [PHASE_RELOCATE, PHASE_MAINTAIN, PHASE_REQUALIFY]
        for phase in ordered:
            for t in self.tasks:
                if t.phase == phase:
                    return t.start
        return None

    @property
    def end(self) -> datetime | None:
        ends = [t.end for t in self.tasks]
        return max(ends) if ends else None


@dataclass(frozen=True)
class PlanResult:
    windows: tuple[WindowResult, ...]

    @property
    def scheduled(self) -> tuple[WindowResult, ...]:
        return tuple(w for w in self.windows if w.scheduled)

    @property
    def deferred(self) -> tuple[WindowResult, ...]:
        return tuple(w for w in self.windows if not w.scheduled)

    def all_tasks(self) -> tuple[ScheduledTask, ...]:
        return tuple(t for w in self.windows for t in w.tasks)

    def all_locks(self) -> tuple[Lock, ...]:
        return tuple(lk for w in self.windows for lk in w.locks)

    def all_moves(self) -> tuple[RelocationMove, ...]:
        return tuple(m for w in self.windows for m in w.moves)

    def all_deferred_lots(self) -> tuple[DeferredLot, ...]:
        return tuple(d for w in self.windows for d in w.deferred_lots)


def _overlaps(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a < end_b and start_b < end_a


def _redundancy_violated(
    window: MaintenanceWindow,
    start: datetime,
    end: datetime,
    freezers: Sequence[FreezerUnit],
    committed: Sequence[tuple[str, datetime, datetime]],
    min_running: dict[str, int],
    default_min: int,
) -> str | None:
    """Return a reason if N+1 / split-feed rules would be broken.

    Only circuits that lose a unit are inspected; untouched circuits are
    independent and never constrain the window.
    """
    by_key = {f.key: f for f in freezers}
    target = by_key[window.freezer_key]
    down: dict[str, set[str]] = {target.circuit: {target.key}}
    for fk, s, e in committed:
        if fk == window.freezer_key and _overlaps(start, end, s, e):
            return R_EQUIPMENT
        if not _overlaps(start, end, s, e):
            continue
        unit = by_key.get(fk)
        if unit is None:
            continue
        down.setdefault(unit.circuit, set()).add(fk)
    for circuit, down_units in down.items():
        healthy = [
            f for f in freezers
            if f.circuit == circuit and f.available and f.key not in down_units
        ]
        required = min_running.get(circuit, default_min)
        if len(healthy) < required:
            return R_REDUNDANCY
    return None


def _allocate_relocations(
    window: MaintenanceWindow,
    start: datetime,
    end: datetime,
    freezers: Sequence[FreezerUnit],
    lots_on_unit: Sequence[CargoLot],
    occupancy: dict[str, Decimal],
    position: dict[str, str],
    committed: Sequence[tuple[str, datetime, datetime]],
) -> tuple[list[RelocationMove], list[tuple[CargoLot, str, Decimal]]]:
    """Greedily reserve destination capacity.

    Non-deferrable lots are served first so scarce destination capacity is
    never spent on a lot that could wait.
    """
    by_key = {f.key: f for f in freezers}
    down_during_work = {window.freezer_key}
    for fk, s, e in committed:
        if _overlaps(start, end, s, e):
            down_during_work.add(fk)

    moves: list[RelocationMove] = []
    blockers: list[tuple[CargoLot, str]] = []
    buffer_minutes = int(window.relocate_buffer.total_seconds() // 60)

    for lot in sorted(lots_on_unit, key=lambda x: (x.deferrable, x.key)):
        if lot.relocation_minutes > buffer_minutes:
            blockers.append((lot, R_MOVE_TIME, lot.quantity))
            continue
        source = position[lot.key]
        outstanding = lot.quantity
        for dest_key in sorted(lot.allowed_freezers):
            if outstanding <= 0:
                break
            dest = by_key.get(dest_key)
            if (
                dest is None
                or not dest.available
                or dest_key in down_during_work
                or dest_key == source
            ):
                continue
            room = dest.capacity - occupancy.get(dest_key, Decimal("0"))
            take = min(room, outstanding)
            if take <= 0:
                continue
            moves.append(RelocationMove(window.key, lot.key, source, dest_key, take))
            occupancy[dest_key] = occupancy.get(dest_key, Decimal("0")) + take
            outstanding -= take
        if outstanding > 0:
            has_eligible = any(
                dest_key in by_key
                and by_key[dest_key].available
                and dest_key not in down_during_work
                and dest_key != source
                for dest_key in lot.allowed_freezers
            )
            blockers.append((lot, R_CAPACITY if has_eligible else R_NO_DESTINATION, outstanding))
    return moves, blockers


def plan(
    windows: Iterable[MaintenanceWindow],
    freezers: Iterable[FreezerUnit],
    lots: Iterable[CargoLot],
    resources: Iterable[CrewResource],
    min_running: dict[str, int] | None = None,
    default_min_running: int = 1,
    strict: bool = False,
) -> PlanResult:
    """Build a maintenance plan, preserving every feasible partial result.

    ``min_running`` gives the minimum healthy units each circuit must keep
    while work is in progress; the default of 1 encodes plain N+1 (at least
    one surviving unit while the serviced one is down).  With ``strict`` the
    first un-schedulable window raises; otherwise it is reported and later
    windows are still attempted.
    """
    ws = list(windows)
    fs = list(freezers)
    all_lots = list(lots)
    rs = list(resources)
    for w in ws:
        w.validate()
    for f in fs:
        f.validate()
    for r in rs:
        r.validate()

    freezer_by_key = {f.key: f for f in fs}
    for w in ws:
        if w.freezer_key not in freezer_by_key:
            raise ValueError(f"window {w.key} references unknown freezer {w.freezer_key}")
    lot_records: dict[str, CargoLot] = {}
    for lot in all_lots:
        lot.validate()
        if lot.freezer_key not in freezer_by_key:
            raise ValueError(f"lot {lot.key} references unknown freezer")
        lot_records[lot.key] = lot

    min_running = min_running or {}
    free_at = {r.key: utc(r.available_from) for r in rs}
    committed: list[tuple[str, datetime, datetime]] = []
    occupancy: dict[str, Decimal] = {f.key: Decimal("0") for f in fs}
    position: dict[str, str] = {}
    for lot in all_lots:
        occupancy[lot.freezer_key] += lot.quantity
        position[lot.key] = lot.freezer_key

    results: list[WindowResult] = []
    for window in sorted(
        ws,
        key=lambda w: (-w.priority, utc(w.latest), utc(w.earliest), w.key),
    ):
        result = _plan_one(
            window, fs, lot_records, position, rs, free_at, committed,
            occupancy, min_running, default_min_running,
        )
        if result.scheduled:
            for task in result.tasks:
                free_at[task.resource_key] = max(free_at[task.resource_key], utc(task.end))
            maintain = next(t for t in result.tasks if t.phase == PHASE_MAINTAIN)
            committed.append((window.freezer_key, utc(maintain.start), utc(maintain.end)))
            for move in result.moves:
                occupancy[move.source] -= move.quantity
                occupancy[move.destination] = occupancy.get(move.destination, Decimal("0")) + move.quantity
                position[move.lot_key] = move.destination
        elif strict:
            raise ValueError(f"cannot schedule {result.window_key}: {result.reason}")
        results.append(result)

    order = {w.key: i for i, w in enumerate(ws)}
    results.sort(key=lambda r: order[r.window_key])
    return PlanResult(tuple(results))


def _plan_one(
    window: MaintenanceWindow,
    freezers: Sequence[FreezerUnit],
    lot_records: dict[str, CargoLot],
    position: dict[str, str],
    resources: Sequence[CrewResource],
    free_at: dict[str, datetime],
    committed: list[tuple[str, datetime, datetime]],
    occupancy: dict[str, Decimal],
    min_running: dict[str, int],
    default_min_running: int,
) -> WindowResult:
    zone = next(f for f in freezers if f.key == window.freezer_key).zone
    lots_on_unit = [lot for lot in lot_records.values() if position[lot.key] == window.freezer_key]

    def deferred_window(reason: str) -> WindowResult:
        rows = [
            DeferredLot(lot.key, window.key, lot.quantity, reason)
            for lot in lots_on_unit
        ]
        return WindowResult(window.key, False, deferred_lots=tuple(rows), reason=reason)

    def crew(role: str) -> list[CrewResource]:
        return [r for r in resources if r.zone == zone and role in r.roles]

    relocate_crew = crew(ROLE_RELOCATE)
    maintain_crew = crew(ROLE_MAINTAIN)
    requalify_crew = crew(ROLE_REQUALIFY) if window.requalify_duration > timedelta() else []
    if not relocate_crew or not maintain_crew or not requalify_crew:
        missing = []
        if not relocate_crew:
            missing.append(ROLE_RELOCATE)
        if not maintain_crew:
            missing.append(ROLE_MAINTAIN)
        if not requalify_crew:
            missing.append(ROLE_REQUALIFY)
        return deferred_window(f"{R_RESOURCE}:{','.join(missing)}")

    earliest = utc(window.earliest)
    latest = utc(window.latest)
    duration = window.duration
    requalify = window.requalify_duration
    buffer = window.relocate_buffer

    candidate = None
    for rr in relocate_crew:
        rel_start = max(earliest, free_at[rr.key], utc(rr.available_from))
        rel_ready = rel_start + buffer
        if rel_ready > utc(rr.available_until):
            continue
        for rm in maintain_crew:
            start = max(rel_ready, free_at[rm.key], utc(rm.available_from))
            end = start + duration
            if end > utc(rm.available_until) or end + requalify > latest:
                continue
            for rq in requalify_crew or [None]:
                q_start = end
                if rq is not None:
                    q_start = max(q_start, free_at[rq.key], utc(rq.available_from))
                    if q_start + requalify > utc(rq.available_until) or q_start + requalify > latest:
                        continue
                violation = _redundancy_violated(
                    window, start, end, freezers, committed, min_running, default_min_running,
                )
                if violation is not None:
                    continue
                choice = (start, rr.key, rm.key, rq.key if rq else "")
                if candidate is None or choice < candidate[0]:
                    candidate = (choice, rr, rm, rq, rel_start, start, end, q_start)

    if candidate is None:
        probe_start = earliest + buffer
        reason = R_TIME
        if _redundancy_violated(
            window, probe_start, probe_start + duration,
            freezers, committed, min_running, default_min_running,
        ) == R_REDUNDANCY:
            reason = R_REDUNDANCY
        return deferred_window(reason)

    _, rr, rm, rq, rel_start, start, end, q_start = candidate

    trial_occupancy = dict(occupancy)
    moves, blockers = _allocate_relocations(
        window, start, end, freezers, lots_on_unit, trial_occupancy, position, committed,
    )
    hard = [b for b in blockers if not b[0].deferrable]
    if hard or (blockers and not window.partial_relocation_allowed):
        why = (hard[0] if hard else blockers[0])[1]
        return deferred_window(f"{R_RELOCATION}:{why}")

    tasks = [
        ScheduledTask(window.key, PHASE_RELOCATE, rr.key, rel_start, start),
        ScheduledTask(window.key, PHASE_MAINTAIN, rm.key, start, end),
    ]
    if rq is not None:
        tasks.append(ScheduledTask(window.key, PHASE_REQUALIFY, rq.key, q_start, q_start + requalify))
    window_end = max(t.end for t in tasks)

    locks = [Lock(window.key, "freezer", window.freezer_key, rel_start, window_end)]
    for move in moves:
        locks.append(Lock(window.key, "lot", move.lot_key, rel_start, start))
    for task in tasks:
        locks.append(Lock(window.key, "resource", f"{task.resource_key}:{task.phase}", task.start, task.end))

    deferred_lots = tuple(
        DeferredLot(lot.key, window.key, remaining, why)
        for lot, why, remaining in blockers
    )
    return WindowResult(
        window_key=window.key,
        scheduled=True,
        tasks=tuple(tasks),
        moves=tuple(moves),
        locks=tuple(locks),
        deferred_lots=deferred_lots,
    )


def summarize(result: PlanResult) -> dict[str, object]:
    return {
        "windows_total": len(result.windows),
        "windows_scheduled": len(result.scheduled),
        "windows_deferred": len(result.deferred),
        "locks": len(result.all_locks()),
        "moves": len(result.all_moves()),
        "deferred_lots": len(result.all_deferred_lots()),
    }


def resource_conflicts(result: PlanResult) -> list[tuple[str, str]]:
    """Diagnostic: two committed tasks of the same resource overlapping."""
    by_resource: dict[str, list[ScheduledTask]] = {}
    for task in result.all_tasks():
        by_resource.setdefault(task.resource_key, []).append(task)
    out = []
    for rows in by_resource.values():
        rows.sort(key=lambda t: utc(t.start))
        for a, b in zip(rows, rows[1:]):
            if utc(b.start) < utc(a.end):
                out.append((a.window_key, b.window_key))
    return out

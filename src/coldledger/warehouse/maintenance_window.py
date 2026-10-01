from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

STATUSES = ("locked", "partial", "deferred")
TASK_ORDER = ("move_out", "service", "qualification", "restock")


@dataclass(frozen=True)
class CoolingUnit:
    key: str
    feed: str
    capacity: Decimal
    zone: str = "default"

    def validate(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not self.feed:
            raise ValueError("feed required")


@dataclass(frozen=True)
class StoredLoad:
    key: str
    unit: str
    quantity: Decimal
    priority: int = 0
    deferrable: bool = True

    def validate(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class WindowRequest:
    key: str
    unit: str
    earliest: datetime
    latest: datetime
    service: timedelta
    qualification: timedelta
    move_minutes_per_unit: Decimal = Decimal("15")
    restock: bool = True
    allow_partial: bool = True

    def validate(self) -> None:
        if self.service <= timedelta():
            raise ValueError("service must be positive")
        if self.qualification <= timedelta():
            raise ValueError("qualification must be positive")
        if self.move_minutes_per_unit <= 0:
            raise ValueError("move rate must be positive")
        if _utc(self.latest) <= _utc(self.earliest):
            raise ValueError("invalid scheduling window")


@dataclass(frozen=True)
class LockedTask:
    kind: str
    unit: str
    start: datetime
    end: datetime

    def validate(self) -> None:
        if self.kind not in TASK_ORDER:
            raise ValueError("unknown task kind")
        if _utc(self.end) <= _utc(self.start):
            raise ValueError("non-positive task")


@dataclass(frozen=True)
class TransferLock:
    load: str
    source: str
    target: str
    quantity: Decimal
    start: datetime
    end: datetime

    def validate(self) -> None:
        if self.source == self.target:
            raise ValueError("self transfer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if _utc(self.end) <= _utc(self.start):
            raise ValueError("non-positive transfer")


@dataclass(frozen=True)
class WindowPlan:
    request: str
    status: str
    tasks: tuple[LockedTask, ...] = ()
    transfers: tuple[TransferLock, ...] = ()
    deferred_loads: tuple[tuple[str, Decimal], ...] = ()
    reasons: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("unknown status")
        for task in self.tasks:
            task.validate()
        for transfer in self.transfers:
            transfer.validate()
        for key, quantity in self.deferred_loads:
            if not key or quantity <= 0:
                raise ValueError("invalid deferred load")
        order = [TASK_ORDER.index(t.kind) for t in self.tasks]
        if order != sorted(order):
            raise ValueError("task chain out of order")
        for a, b in zip(self.tasks, self.tasks[1:]):
            if _utc(b.start) < _utc(a.end):
                raise ValueError("overlapping tasks")
        if self.status == "locked" and (self.deferred_loads or self.reasons):
            raise ValueError("locked plan carries no deferrals")
        if self.status == "partial" and not (self.deferred_loads or self.reasons):
            raise ValueError("partial plan must explain itself")
        if self.status == "deferred" and self.tasks:
            raise ValueError("deferred plan locks no tasks")


def _utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return v.astimezone(timezone.utc)


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def n_plus_one_ok(units: Iterable[CoolingUnit], loads: Iterable[StoredLoad], down: Iterable[str] = ()) -> bool:
    offline = set(down)
    remaining = [u for u in units if u.key not in offline]
    if not remaining:
        return False
    total_load = sum((l.quantity for l in loads), Decimal("0"))
    capacity = sum((u.capacity for u in remaining), Decimal("0"))
    largest = max(u.capacity for u in remaining)
    return capacity - largest >= total_load


def feed_ok(units: Iterable[CoolingUnit], down: Iterable[str] = ()) -> bool:
    offline = set(down)
    feeds: dict[str, list[str]] = {}
    for u in units:
        feeds.setdefault(u.feed, []).append(u.key)
    return all(any(key not in offline for key in keys) for keys in feeds.values())


def move_duration(quantity: Decimal, rate: Decimal) -> timedelta:
    if rate <= 0:
        raise ValueError("move rate must be positive")
    return timedelta(minutes=float(quantity * rate))


def _allocate(to_move, receivers, source_feed, by_key, start, rate):
    transfers: list[TransferLock] = []
    deferred: list[tuple[str, Decimal]] = []
    cursor = start
    for load in to_move:
        outstanding = load.quantity
        ranked = sorted(
            ((key, left) for key, left in receivers.items() if left > 0),
            key=lambda kv: (by_key[kv[0]].feed == source_feed, -kv[1], kv[0]),
        )
        for key, left in ranked:
            if outstanding <= 0:
                break
            take = min(left, outstanding)
            span = move_duration(take, rate)
            transfers.append(TransferLock(load.key, load.unit, key, take, cursor, cursor + span))
            cursor += span
            receivers[key] -= take
            outstanding -= take
        if outstanding > 0:
            deferred.append((load.key, outstanding))
    return transfers, deferred, cursor


def plan(request: WindowRequest, units: Iterable[CoolingUnit], loads: Iterable[StoredLoad], existing: Iterable[WindowPlan] = ()) -> WindowPlan:
    request.validate()
    us = list(units)
    ls = list(loads)
    for u in us:
        u.validate()
    for l in ls:
        l.validate()
    by_key = {u.key: u for u in us}
    if len(by_key) != len(us):
        raise ValueError("duplicate unit key")
    if request.unit not in by_key:
        raise ValueError("unknown unit")
    for l in ls:
        if l.unit not in by_key:
            raise ValueError("load placed on unknown unit")
    target = by_key[request.unit]
    earliest = _utc(request.earliest)
    latest = _utc(request.latest)
    to_move = sorted((l for l in ls if l.unit == request.unit), key=lambda l: (-l.priority, l.deferrable, l.key))

    def bail(reasons, transfers=(), deferred=None):
        if not request.allow_partial:
            raise ValueError(reasons[0])
        if deferred is None:
            deferred = tuple((l.key, l.quantity) for l in to_move)
        result = WindowPlan(request.key, "deferred", (), tuple(transfers), tuple(deferred), tuple(reasons))
        result.validate()
        return result

    # only locked/partial windows hold resources; deferred ones are advisory
    booked = [p for p in existing if p.status != "deferred"]
    reasons: list[str] = []
    down = {request.unit}
    for prev in booked:
        for task in prev.tasks:
            if not _overlaps(earliest, latest, _utc(task.start), _utc(task.end)):
                continue
            if task.unit == request.unit:
                reasons.append(f"unit {task.unit} already locked by {prev.request}")
            elif task.kind in ("service", "qualification"):
                other = by_key.get(task.unit)
                if other is not None and other.feed == target.feed:
                    reasons.append(f"feed {target.feed} already offline at {task.unit}")
                else:
                    down.add(task.unit)
    if not feed_ok(us, down):
        reasons.append(f"feed {target.feed} would lose its last online unit")
    if not n_plus_one_ok(us, ls, down):
        reasons.append("N+1 capacity violated while unit is offline")
    if reasons:
        return bail(reasons)

    current: dict[str, Decimal] = {}
    for l in ls:
        current[l.unit] = current.get(l.unit, Decimal("0")) + l.quantity
    reserved: dict[str, Decimal] = {}
    for prev in booked:
        for tr in prev.transfers:
            if _overlaps(earliest, latest, _utc(tr.start), _utc(tr.end)):
                reserved[tr.target] = reserved.get(tr.target, Decimal("0")) + tr.quantity
    receivers: dict[str, Decimal] = {}
    for u in us:
        if u.key in down:
            continue
        free = u.capacity - current.get(u.key, Decimal("0")) - reserved.get(u.key, Decimal("0"))
        if free > 0:
            receivers[u.key] = free

    transfers, deferred, moved_end = _allocate(to_move, receivers, target.feed, by_key, earliest, request.move_minutes_per_unit)
    stuck = {key for key, _ in deferred}
    critical = sorted(l.key for l in to_move if not l.deferrable and l.key in stuck)
    if critical:
        return bail([f"critical load has no destination: {', '.join(critical)}"], transfers, tuple(deferred))

    tasks: list[LockedTask] = []
    if transfers:
        tasks.append(LockedTask("move_out", request.unit, earliest, moved_end))
    service_end = moved_end + request.service
    qual_end = service_end + request.qualification
    tasks.append(LockedTask("service", request.unit, moved_end, service_end))
    tasks.append(LockedTask("qualification", request.unit, service_end, qual_end))
    if qual_end > latest:
        return bail(["move, service and qualification exceed the window"], transfers, tuple(deferred))
    if request.restock and transfers:
        restock_end = qual_end + (moved_end - earliest)
        if restock_end <= latest:
            tasks.append(LockedTask("restock", request.unit, qual_end, restock_end))
        else:
            reasons.append("restock deferred beyond the window")
    if deferred:
        reasons.append("deferred loads: " + ", ".join(f"{k} ({q})" for k, q in deferred))

    status = "locked" if not reasons else "partial"
    if status != "locked" and not request.allow_partial:
        raise ValueError(reasons[0])
    result = WindowPlan(request.key, status, tuple(tasks), tuple(transfers), tuple(deferred), tuple(reasons))
    result.validate()
    return result


def locked_tasks(plan: WindowPlan, kind: str) -> list[LockedTask]:
    if kind not in TASK_ORDER:
        raise ValueError("unknown task kind")
    return [t for t in plan.tasks if t.kind == kind]


def summarize(plans: Iterable[WindowPlan]) -> dict[str, int]:
    out = {s: 0 for s in STATUSES}
    for p in plans:
        out[p.status] += 1
    return out


def explain(plan: WindowPlan) -> list[str]:
    lines = [f"{plan.request}: {plan.status}"]
    for t in plan.tasks:
        lines.append(f"task {t.kind} on {t.unit}: {t.start.isoformat()} -> {t.end.isoformat()}")
    for tr in plan.transfers:
        lines.append(f"move {tr.load}: {tr.source} -> {tr.target} qty {tr.quantity}")
    for key, quantity in plan.deferred_loads:
        lines.append(f"defer {key}: qty {quantity}")
    lines.extend(f"reason: {r}" for r in plan.reasons)
    return lines

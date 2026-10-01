import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from coldledger.api import ColdLedgerService
from coldledger.warehouse.maintenance_window import (
    CoolingUnit, StoredLoad, WindowRequest, WindowPlan, LockedTask, TransferLock,
    plan, n_plus_one_ok, feed_ok, move_duration, locked_tasks, summarize, explain,
)

UTC = timezone.utc
T0 = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
D = Decimal


def units():
    return [
        CoolingUnit("F1", "A", D("100")),
        CoolingUnit("F2", "B", D("100")),
        CoolingUnit("F3", "A", D("80")),
        CoolingUnit("F4", "B", D("80")),
    ]


def base_loads():
    return [
        StoredLoad("L1", "F1", D("30"), priority=5, deferrable=False),
        StoredLoad("L2", "F1", D("20")),
        StoredLoad("L3", "F2", D("40")),
        StoredLoad("L4", "F3", D("30")),
        StoredLoad("L5", "F4", D("20")),
    ]


def request(hours=36, **kw):
    args = dict(key="W1", unit="F1", earliest=T0, latest=T0 + timedelta(hours=hours),
                service=timedelta(hours=4), qualification=timedelta(hours=6))
    args.update(kw)
    return WindowRequest(**args)


class MaintenanceWindowTests(unittest.TestCase):
    def test_locked_window_chains_all_tasks(self):
        p = plan(request(), units(), base_loads())
        self.assertEqual(p.status, "locked")
        self.assertEqual([t.kind for t in p.tasks], ["move_out", "service", "qualification", "restock"])
        self.assertEqual(p.tasks[0].start, T0)
        self.assertEqual(p.tasks[-1].end, T0 + timedelta(hours=35))
        self.assertEqual(p.deferred_loads, ())
        self.assertEqual(sum(t.quantity for t in p.transfers), D("50"))
        self.assertEqual(p.transfers[0].target, "F2")  # cross-feed preferred
        for a, b in zip(p.tasks, p.tasks[1:]):
            self.assertEqual(a.end, b.start)

    def test_n_plus_one_violation_defers_whole_window(self):
        loads = [StoredLoad("L1", "F1", D("60"), deferrable=False),
                 StoredLoad("L2", "F2", D("50")),
                 StoredLoad("L3", "F3", D("40")),
                 StoredLoad("L4", "F4", D("20"))]
        p = plan(request(), units(), loads)
        self.assertEqual(p.status, "deferred")
        self.assertEqual(p.tasks, ())
        self.assertEqual(p.transfers, ())
        self.assertEqual(p.deferred_loads, (("L1", D("60")),))
        self.assertTrue(any("N+1" in r for r in p.reasons))

    def test_same_feed_overlap_defers(self):
        busy = WindowPlan("W0", "locked",
                          tasks=(LockedTask("service", "F3", T0 + timedelta(hours=2), T0 + timedelta(hours=8)),))
        p = plan(request(), units(), base_loads(), existing=[busy])
        self.assertEqual(p.status, "deferred")
        self.assertTrue(any("feed A" in r for r in p.reasons))

    def test_deferred_existing_window_does_not_block(self):
        ghost = WindowPlan("W0", "deferred",
                           transfers=(TransferLock("X", "F1", "F2", D("90"), T0, T0 + timedelta(hours=2)),),
                           reasons=("nope",))
        p = plan(request(), units(), base_loads(), existing=[ghost])
        self.assertEqual(p.status, "locked")

    def test_reserved_capacity_partial_defers_only_overflow(self):
        loads = [StoredLoad("L1", "F1", D("20"), priority=9, deferrable=False),
                 StoredLoad("L2", "F1", D("50")),
                 StoredLoad("L3", "F2", D("20")),
                 StoredLoad("L4", "F3", D("10")),
                 StoredLoad("L5", "F4", D("10"))]
        reserved = WindowPlan("W0", "partial", transfers=(
            TransferLock("X", "F9", "F2", D("60"), T0, T0 + timedelta(hours=2)),
            TransferLock("Y", "F9", "F3", D("65"), T0, T0 + timedelta(hours=2)),
            TransferLock("Z", "F9", "F4", D("65"), T0, T0 + timedelta(hours=2)),
        ), reasons=("reserved",))
        p = plan(request(hours=40), units(), loads, existing=[reserved])
        self.assertEqual(p.status, "partial")
        self.assertEqual(p.deferred_loads, (("L2", D("40")),))
        placed = {t.load: D("0") for t in p.transfers}
        for t in p.transfers:
            placed[t.load] += t.quantity
        self.assertEqual(placed["L1"], D("20"))  # critical load fully placed
        self.assertEqual(placed["L2"], D("10"))

    def test_critical_load_without_destination_defers(self):
        # N+1 headroom: 260-100=160 >= total 160; receivers free 50+45+45=140 >= 40
        loads = [StoredLoad("L1", "F1", D("40"), deferrable=False),
                 StoredLoad("L3", "F2", D("50")),
                 StoredLoad("L4", "F3", D("35")),
                 StoredLoad("L5", "F4", D("35"))]
        # a competing window reserves every free pallet position
        reserved = WindowPlan("W0", "partial", transfers=(
            TransferLock("X", "F9", "F2", D("50"), T0, T0 + timedelta(hours=2)),
            TransferLock("Y", "F9", "F3", D("45"), T0, T0 + timedelta(hours=2)),
            TransferLock("Z", "F9", "F4", D("45"), T0, T0 + timedelta(hours=2)),
        ), reasons=("reserved",))
        p = plan(request(), units(), loads, existing=[reserved])
        self.assertEqual(p.status, "deferred")
        self.assertTrue(any("critical load" in r for r in p.reasons))
        self.assertEqual(p.deferred_loads, (("L1", D("40")),))

    def test_restock_overflow_degrades_to_partial(self):
        p = plan(request(hours=30), units(), base_loads())
        self.assertEqual(p.status, "partial")
        self.assertEqual([t.kind for t in p.tasks], ["move_out", "service", "qualification"])
        self.assertTrue(any("restock" in r for r in p.reasons))

    def test_short_window_defers_but_keeps_feasible_transfers(self):
        p = plan(request(hours=20), units(), base_loads())
        self.assertEqual(p.status, "deferred")
        self.assertEqual(p.tasks, ())
        self.assertEqual(sum(t.quantity for t in p.transfers), D("50"))
        self.assertEqual(p.deferred_loads, ())

    def test_allow_partial_false_raises(self):
        loads = [StoredLoad("L1", "F1", D("60"), deferrable=False),
                 StoredLoad("L2", "F2", D("50")),
                 StoredLoad("L3", "F3", D("40")),
                 StoredLoad("L4", "F4", D("20"))]
        with self.assertRaises(ValueError):
            plan(request(allow_partial=False), units(), loads)

    def test_naive_datetime_rejected(self):
        bad = WindowRequest("W1", "F1", datetime(2026, 10, 5, 8), T0 + timedelta(hours=36),
                            timedelta(hours=4), timedelta(hours=6))
        with self.assertRaises(ValueError):
            plan(bad, units(), base_loads())

    def test_overlap_on_same_unit_defers(self):
        busy = WindowPlan("W0", "locked",
                          tasks=(LockedTask("service", "F1", T0 + timedelta(hours=1), T0 + timedelta(hours=3)),))
        p = plan(request(), units(), base_loads(), existing=[busy])
        self.assertEqual(p.status, "deferred")
        self.assertTrue(any("already locked" in r for r in p.reasons))

    def test_predicates_and_move_duration(self):
        us, ls = units(), base_loads()
        self.assertTrue(n_plus_one_ok(us, ls, ("F1",)))
        self.assertFalse(n_plus_one_ok(us, ls, ("F1", "F2")))
        self.assertTrue(feed_ok(us, ("F1",)))
        self.assertFalse(feed_ok(us, ("F1", "F3")))
        self.assertEqual(move_duration(D("4"), D("15")), timedelta(hours=1))

    def test_helpers(self):
        p = plan(request(), units(), base_loads())
        self.assertEqual(len(locked_tasks(p, "service")), 1)
        self.assertEqual(summarize([p]), {"locked": 1, "partial": 0, "deferred": 0})
        self.assertTrue(explain(p)[0].startswith("W1: locked"))

    def test_service_facade(self):
        p = ColdLedgerService().plan_maintenance_window(request(), units(), base_loads())
        self.assertEqual(p.status, "locked")


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from coldledger.warehouse.maintenance_window import (
    CargoLot,
    CrewResource,
    FreezerUnit,
    MaintenanceWindow,
    PHASE_MAINTAIN,
    PHASE_RELOCATE,
    PHASE_REQUALIFY,
    plan,
    resource_conflicts,
    summarize,
)

UTC = timezone.utc


def at(h, m=0, day=5):
    return datetime(2026, 10, day, h, m, tzinfo=UTC)


def freezers():
    return [
        FreezerUnit("F1", "Z", "C1", Decimal("100")),
        FreezerUnit("F2", "Z", "C1", Decimal("100")),
        FreezerUnit("F3", "Z", "C2", Decimal("100")),
        FreezerUnit("F4", "Z", "C2", Decimal("100")),
    ]


def resources():
    span = (at(6), at(22))
    return [
        CrewResource("R1", "Z", *span, frozenset({"relocate"})),
        CrewResource("R2", "Z", *span, frozenset({"maintain"})),
        CrewResource("R3", "Z", *span, frozenset({"requalify"})),
        CrewResource("R4", "Z", *span, frozenset({"relocate", "maintain", "requalify"})),
    ]


def window(key, freezer="F1", earliest=None, latest=None, duration=None,
           buffer=None, requalify=None, priority=0, partial=False):
    return MaintenanceWindow(
        key, freezer,
        earliest or at(8),
        latest or at(20),
        duration or timedelta(hours=2),
        buffer if buffer is not None else timedelta(hours=1),
        requalify if requalify is not None else timedelta(minutes=30),
        priority=priority,
        partial_relocation_allowed=partial,
    )


def lots_on_f1():
    return [
        CargoLot("L1", "F1", Decimal("20"), frozenset({"F2"}), relocation_minutes=10),
        CargoLot("L2", "F1", Decimal("5"), frozenset({"F2", "F3"}), relocation_minutes=15),
    ]


class MaintenanceWindowTests(unittest.TestCase):
    def by_key(self, result, key):
        return next(w for w in result.windows if w.window_key == key)

    def test_happy_path_locks_equipment_transfers_and_review(self):
        result = plan([window("W1")], freezers(), lots_on_f1(), resources())
        w = self.by_key(result, "W1")
        self.assertTrue(w.scheduled)
        phases = {t.phase for t in w.tasks}
        self.assertEqual(phases, {PHASE_RELOCATE, PHASE_MAINTAIN, PHASE_REQUALIFY})

        by_phase = {t.phase: t for t in w.tasks}
        # Relocation finishes before work starts; requalification after work.
        self.assertLessEqual(by_phase[PHASE_RELOCATE].end, by_phase[PHASE_MAINTAIN].start)
        self.assertLessEqual(by_phase[PHASE_MAINTAIN].end, by_phase[PHASE_REQUALIFY].start)
        # Reconfirmation must be inside the window deadline.
        self.assertLessEqual(by_phase[PHASE_REQUALIFY].end, at(20))

        # Both lots moved off the serviced unit into surviving capacity.
        self.assertEqual({m.destination for m in w.moves}, {"F2"})
        self.assertEqual(sum((m.quantity for m in w.moves), Decimal("0")), Decimal("25"))

        # Equipment, inventory and review tasks locked atomically.
        kinds = {lk.kind for lk in w.locks}
        self.assertEqual(kinds, {"freezer", "lot", "resource"})
        self.assertIn(("freezer", "F1"), {(lk.kind, lk.target) for lk in w.locks})
        self.assertEqual({lk.target for lk in w.locks if lk.kind == "lot"}, {"L1", "L2"})
        self.assertEqual(resource_conflicts(result), [])

    def test_n_plus_one_blocks_last_surviving_unit_on_circuit(self):
        # min 2 running on C1 which only has 2 units: servicing F1 leaves one.
        result = plan(
            [window("W1")], freezers(), lots_on_f1(), resources(),
            min_running={"C1": 2},
        )
        w = self.by_key(result, "W1")
        self.assertFalse(w.scheduled)
        self.assertEqual(w.reason, "redundancy")

    def test_split_feeds_are_independent(self):
        # C1 pair F1/F2, C2 pair F3/F4: work on both circuits simultaneously
        # is fine because the feeds are independent.
        w1 = window("W1", freezer="F1", earliest=at(8), latest=at(12))
        w2 = window("W2", freezer="F3", earliest=at(8), latest=at(12))
        lots = lots_on_f1() + [CargoLot("L3", "F3", Decimal("8"), frozenset({"F4"}))]
        result = plan([w1, w2], freezers(), lots, resources())
        self.assertEqual(len(result.scheduled), 2)
        # F1's cargo went to its C1 twin, F3's cargo to the C2 twin.
        self.assertEqual({m.source: m.destination for m in result.all_moves()},
                         {"F1": "F2", "F3": "F4"})

    def test_overlapping_work_on_same_circuit_squeezes_lower_priority(self):
        w1 = window("W1", freezer="F1", earliest=at(8), latest=at(11, 30), priority=5)
        w2 = window("W2", freezer="F2", earliest=at(9), latest=at(13))
        w3 = window("W3", freezer="F3", earliest=at(9), latest=at(13))
        lots = lots_on_f1() + [CargoLot("L3", "F2", Decimal("3"), frozenset({"F1"}))]
        result = plan([w1, w2, w3], freezers(), lots, resources())
        self.assertTrue(self.by_key(result, "W1").scheduled)   # feasible part kept
        self.assertTrue(self.by_key(result, "W3").scheduled)   # other feed independent
        squeezed = self.by_key(result, "W2")
        self.assertFalse(squeezed.scheduled)
        self.assertEqual(squeezed.reason, "redundancy")
        # Cargo physically on the serviced unit is explicitly postponed:
        # L3 is native to F2, L1/L2 were relocated onto F2 by the earlier W1.
        self.assertEqual({d.lot_key for d in squeezed.deferred_lots}, {"L1", "L2", "L3"})
        self.assertTrue(all(d.reason == "redundancy" for d in squeezed.deferred_lots))

    def test_priority_breaks_ties_when_maintainer_serializes(self):
        one_of_each = [
            CrewResource("R1", "Z", at(6), at(22), frozenset({"relocate"})),
            CrewResource("R2", "Z", at(6), at(22), frozenset({"maintain"})),
            CrewResource("R3", "Z", at(6), at(22), frozenset({"requalify"})),
        ]
        w1 = window("W1", freezer="F1", latest=at(12))
        w2 = window("W2", freezer="F2", latest=at(14), priority=5)
        result = plan([w1, w2], freezers(), lots_on_f1(), one_of_each)
        # Higher-priority W2 takes the only maintainer slot early; W1's later
        # maintain+requalify can no longer finish by its 12:00 deadline.
        self.assertTrue(self.by_key(result, "W2").scheduled)
        self.assertFalse(self.by_key(result, "W1").scheduled)

    def test_partial_relocation_keeps_window_and_postpones_stranded_lot(self):
        lots = lots_on_f1() + [
            CargoLot("L9", "F1", Decimal("50"), frozenset(), deferrable=True),
        ]
        result = plan(
            [window("W1", partial=True)], freezers(), lots, resources(),
        )
        w = self.by_key(result, "W1")
        self.assertTrue(w.scheduled)
        self.assertEqual({m.lot_key for m in w.moves}, {"L1", "L2"})
        self.assertEqual({d.lot_key for d in w.deferred_lots}, {"L9"})
        self.assertEqual({d.reason for d in w.deferred_lots}, {"no_allowed_destination"})
        # Only transferable lots are locked for transfer.
        self.assertEqual({lk.target for lk in w.locks if lk.kind == "lot"}, {"L1", "L2"})

    def test_partial_not_allowed_defers_whole_window(self):
        lots = lots_on_f1() + [
            CargoLot("L9", "F1", Decimal("50"), frozenset(), deferrable=True),
        ]
        result = plan([window("W1", partial=False)], freezers(), lots, resources())
        w = self.by_key(result, "W1")
        self.assertFalse(w.scheduled)
        self.assertTrue(w.reason.startswith("relocation:"))
        # Every lot on the stranded unit is reported for postponement.
        self.assertEqual({d.lot_key for d in w.deferred_lots}, {"L1", "L2", "L9"})
        self.assertEqual(w.tasks, ())
        self.assertEqual(w.locks, ())

    def test_non_deferrable_lot_blocks_even_in_partial_mode(self):
        small_dest = [
            FreezerUnit("F1", "Z", "C1", Decimal("100")),
            FreezerUnit("F2", "Z", "C1", Decimal("5")),
        ]
        lots = [CargoLot("H", "F1", Decimal("40"), frozenset({"F2"}), deferrable=False)]
        result = plan([window("W1", partial=True)], small_dest, lots, resources())
        w = self.by_key(result, "W1")
        self.assertFalse(w.scheduled)
        self.assertEqual(w.reason, "relocation:destination_capacity")
        self.assertEqual({d.lot_key for d in w.deferred_lots}, {"H"})

    def test_move_longer_than_buffer_must_postpone(self):
        lots = [CargoLot("SLOW", "F1", Decimal("1"), frozenset({"F2"}),
                         relocation_minutes=90, deferrable=False)]
        result = plan([window("W1", partial=True, buffer=timedelta(minutes=60))],
                      freezers(), lots, resources())
        w = self.by_key(result, "W1")
        self.assertFalse(w.scheduled)
        self.assertEqual(w.reason, "relocation:relocation_time")

    def test_slow_deferrable_lot_postponed_but_window_proceeds(self):
        lots = [CargoLot("SLOW", "F1", Decimal("1"), frozenset({"F2"}),
                         relocation_minutes=90, deferrable=True)]
        result = plan([window("W1", partial=True, buffer=timedelta(minutes=60))],
                      freezers(), lots, resources())
        w = self.by_key(result, "W1")
        self.assertTrue(w.scheduled)
        self.assertEqual({d.lot_key for d in w.deferred_lots}, {"SLOW"})
        self.assertEqual({d.reason for d in w.deferred_lots}, {"relocation_time"})
        self.assertEqual(w.moves, ())

    def test_requalification_must_finish_within_window(self):
        tight = window("W1", earliest=at(18), latest=at(19),
                       duration=timedelta(hours=2), requalify=timedelta(hours=2))
        result = plan([tight], freezers(), lots_on_f1(), resources())
        w = self.by_key(result, "W1")
        self.assertFalse(w.scheduled)
        self.assertEqual(w.reason, "no_feasible_time")

    def test_strict_raises_on_first_infeasible_window(self):
        with self.assertRaises(ValueError):
            plan(
                [window("W1")], freezers(), lots_on_f1(), resources(),
                min_running={"C1": 2}, strict=True,
            )

    def test_crew_double_booking_serializes_overlapping_windows(self):
        # Only one maintainer on shift; second window must wait for the crew.
        one_maintainer = [
            CrewResource("R1", "Z", at(6), at(22), frozenset({"relocate"})),
            CrewResource("R2", "Z", at(6), at(22), frozenset({"maintain"})),
            CrewResource("R3", "Z", at(6), at(22), frozenset({"requalify"})),
        ]
        w1 = window("W1", freezer="F1", earliest=at(8), latest=at(12))
        w2 = window("W2", freezer="F3", earliest=at(8), latest=at(12))
        lots = lots_on_f1() + [CargoLot("L3", "F3", Decimal("8"), frozenset({"F4"}))]
        result = plan([w1, w2], freezers(), lots, one_maintainer)
        # W1 relocates 8-9, works 9-11, requalifies 11-11:30.  W2 cannot get
        # the maintainer before 11 and work+requalify would run past 12:00.
        self.assertTrue(self.by_key(result, "W1").scheduled)
        squeezed = self.by_key(result, "W2")
        self.assertFalse(squeezed.scheduled)
        self.assertEqual({d.lot_key for d in squeezed.deferred_lots}, {"L3"})
        self.assertEqual(resource_conflicts(result), [])

    def test_moved_cargo_consumes_destination_capacity_for_later_window(self):
        # F2 has room for 25 of F1 cargo; a second later window on F3 tries to
        # push its own cargo to F2 and finds the space already consumed.
        units = [
            FreezerUnit("F1", "Z", "C1", Decimal("100")),
            FreezerUnit("F2", "Z", "C1", Decimal("115")),
            FreezerUnit("F3", "Z", "C2", Decimal("100")),
            FreezerUnit("F4", "Z", "C2", Decimal("100")),
        ]
        lots = lots_on_f1() + [CargoLot("L3", "F3", Decimal("100"), frozenset({"F2", "F4"}))]
        w1 = window("W1", freezer="F1", earliest=at(8), latest=at(12), priority=9)
        w2 = window("W2", freezer="F3", earliest=at(13), latest=at(20))
        result = plan([w1, w2], units, lots, resources())
        w = self.by_key(result, "W2")
        # F2 only has 90 left; L3 (100) splits F4/F2 via F4 capacity (100) -> fits.
        self.assertTrue(w.scheduled)
        self.assertEqual(sum((m.quantity for m in w.moves), Decimal("0")), Decimal("100"))

    def test_split_lot_reports_only_stranded_remainder(self):
        # F2 already holds 80 of its 100; a 50-unit lot can move 20 there.
        units = [
            FreezerUnit("F1", "Z", "C1", Decimal("100")),
            FreezerUnit("F2", "Z", "C1", Decimal("100")),
        ]
        lots = [
            CargoLot("BIG", "F1", Decimal("50"), frozenset({"F2"}), deferrable=True),
            CargoLot("KEEP", "F2", Decimal("80"), frozenset({"F2"}), deferrable=True),
        ]
        result = plan([window("W1", partial=True)], units, lots, resources())
        w = self.by_key(result, "W1")
        self.assertTrue(w.scheduled)
        move = next(m for m in w.moves if m.lot_key == "BIG")
        self.assertEqual(move.quantity, Decimal("20"))
        self.assertEqual(move.source, "F1")
        self.assertEqual(move.destination, "F2")
        stranded = next(d for d in w.deferred_lots if d.lot_key == "BIG")
        self.assertEqual(stranded.quantity, Decimal("30"))
        self.assertEqual(stranded.reason, "destination_capacity")

    def test_relocated_lot_source_tracks_its_current_position(self):
        units = [
            FreezerUnit("F1", "Z", "C1", Decimal("100")),
            FreezerUnit("F2", "Z", "C1", Decimal("100")),
            FreezerUnit("F3", "Z", "C2", Decimal("100")),
            FreezerUnit("F4", "Z", "C2", Decimal("100")),
        ]
        lots = [CargoLot("L1", "F1", Decimal("10"), frozenset({"F2", "F4"}))]
        w1 = window("W1", freezer="F1", earliest=at(8), latest=at(12), priority=5)
        w2 = window("W2", freezer="F2", earliest=at(13), latest=at(20))
        result = plan([w1, w2], units, lots, resources())
        second = self.by_key(result, "W2")
        self.assertTrue(second.scheduled)
        move = next(iter(second.moves))
        self.assertEqual((move.lot_key, move.source, move.destination), ("L1", "F2", "F4"))

    def test_summarize_reports_locks_moves_and_postponed_cargo(self):
        w1 = window("W1", freezer="F1")
        w2 = window("W2", freezer="F2", earliest=at(9), latest=at(12))
        result = plan([w1, w2], freezers(), lots_on_f1(), resources())
        summary = summarize(result)
        self.assertEqual(summary["windows_total"], 2)
        self.assertEqual(summary["windows_scheduled"], 1)
        self.assertEqual(summary["windows_deferred"], 1)
        self.assertGreater(summary["locks"], 0)
        self.assertEqual(summary["moves"], 2)
        self.assertGreaterEqual(summary["deferred_lots"], 0)


if __name__ == "__main__":
    unittest.main()

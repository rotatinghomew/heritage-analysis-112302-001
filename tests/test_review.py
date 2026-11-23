import csv
import datetime as dt
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from shelter_monitor.allocate import (
    Schedule,
    audit_schedules,
    plan_next_round,
    reopen_analysis,
)
from shelter_monitor.data import load_dataset
from shelter_monitor.review import run_review

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "round2"
FIXTURES = ROOT / "fixtures"


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class TempFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class ReviewStructureTests(unittest.TestCase):
    def test_real_fixtures_load_and_report(self) -> None:
        report = run_review(load_dataset(FIXTURES))
        self.assertEqual(report["tool"], "shelter-monitor")
        self.assertIn(report["recommendation"]["decision"], {
            "not_supported",
            "unknown_insufficient",
            "hold_neighbour_risk",
            "partial_extension_supported",
        })
        # 缺样绝不能被写成"安全"或"有效"。
        self.assertNotIn("safe", json.dumps(report, ensure_ascii=False).lower())

    def test_as_treated_never_replaces_assignment(self) -> None:
        report = run_review(load_dataset(FIXTURES))
        arms = {cell["assigned"] for cell in report["compliance"]["cells"]}
        self.assertEqual(arms, {"shelter", "control"})
        # W5 抽签为 shelter 但观察 record 也只有 shelter，不能凭空出现 as-treated 组。
        self.assertNotIn("as_treated", json.dumps(report["intention_comparison"]))


class EventClusteringTests(TempFixture):
    def _write_primary(self, rows: list[list[str]]) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review"],
            rows,
        )

    def test_same_rain_event_is_one_cluster(self) -> None:
        # 同一场雨 R1 内，G1 有遮护/对照，G2 也有遮护/对照。
        rows = [
            ["A1", "G1", "shelter", "S1", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable"],
            ["A2", "G1", "control", "S2", "valid", "2026-05-01", "0.50", "0.40", "R1", "stable"],
            ["A3", "G2", "shelter", "S3", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable"],
            ["A4", "G2", "control", "S4", "valid", "2026-05-01", "0.50", "0.40", "R1", "stable"],
        ]
        self._write_primary(rows)
        report = run_review(load_dataset(self.dir))
        overall = report["intention_comparison"]["overall"]
        self.assertEqual(overall["status"], "not_comparable")
        self.assertEqual(overall["comparable_clusters"], 1)
        self.assertIn("同场雨", overall["why"])

    def test_two_events_with_both_arms_are_estimable(self) -> None:
        rows = []
        for index, event in enumerate(["R1", "R2"]):
            day = f"2026-05-{1 + index * 10:02d}"
            rows.extend([
                [f"A1", "G1", "shelter", "S1", "valid", day, "0.10", "0.40", event, "stable"],
                [f"A2", "G1", "control", "S2", "valid", day, "0.50", "0.40", event, "stable"],
            ])
        self._write_primary(rows)
        overall = run_review(load_dataset(self.dir))["intention_comparison"]["overall"]
        self.assertEqual(overall["status"], "estimated")
        self.assertEqual(overall["comparable_clusters"], 2)
        self.assertLess(overall["point"], 0)

    def test_cross_group_only_event_is_excluded(self) -> None:
        # R1 内两臂都在 G1（可比）；R2 内遮护只在 G1、对照只在 G2（跨组，不可比）。
        rows = [
            ["A1", "G1", "shelter", "S1", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable"],
            ["A2", "G1", "control", "S2", "valid", "2026-05-01", "0.50", "0.40", "R1", "stable"],
            ["A3", "G1", "shelter", "S1", "valid", "2026-05-11", "0.10", "0.40", "R2", "stable"],
            ["A4", "G2", "control", "S2", "valid", "2026-05-11", "0.50", "0.40", "R2", "stable"],
        ]
        self._write_primary(rows)
        overall = run_review(load_dataset(self.dir))["intention_comparison"]["overall"]
        self.assertEqual(overall["status"], "not_comparable")
        self.assertEqual(overall["comparable_clusters"], 1)
        self.assertIn("R2", overall["cross_group_only_events"])

    def test_repeated_readings_in_same_wall_event_collapse(self) -> None:
        rows = []
        for index, event in enumerate(["R1", "R2"]):
            day = f"2026-05-{1 + index * 10:02d}"
            rows.extend([
                ["A1", "G1", "shelter", "S1", "valid", day, "0.10", "0.40", event, "stable"],
                ["A1", "G1", "shelter", "S1", "valid", day, "0.12", "0.40", event, "stable"],
                ["A1", "G1", "shelter", "S1", "valid", day, "0.11", "0.40", event, "stable"],
                ["A2", "G1", "control", "S2", "valid", day, "0.50", "0.40", event, "stable"],
            ])
        self._write_primary(rows)
        report = run_review(load_dataset(self.dir))
        # 第 1 条保留；第 2、3 条是同设备同日的重复登记，记为设备重叠并剔除。
        self.assertTrue(report["device_overlaps"])
        self.assertEqual(report["device_overlaps"][0]["retained_row"], 2)
        self.assertEqual(report["device_overlaps"][0]["excluded_rows"], [3, 4])
        self.assertEqual(report["intention_comparison"]["overall"]["status"], "estimated")


class CalibrationTests(TempFixture):
    def test_uncovered_calibration_is_gap_not_dry(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review", "model"],
            [["A1", "G1", "shelter", "S9", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable", "M1"]],
        )
        _write(
            self.dir / "calibrations.csv",
            ["sensor", "model", "valid_from", "valid_to", "calibration"],
            [["S1", "M1", "2026-01-01", "2026-12-31", "valid"]],
        )
        report = run_review(load_dataset(self.dir))
        issues = report["walls"]["A1"]["evidence_issues"]
        self.assertIn("calibration_uncovered", issues)
        self.assertIn("no_usable_moisture", issues)
        self.assertEqual(report["intention_comparison"]["overall"]["status"], "no_data")

    def test_calibration_only_valid_on_declared_dates(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review", "model"],
            [["A1", "G1", "shelter", "S1", "valid", "2026-08-01", "0.10", "0.40", "R1", "stable", "M1"]],
        )
        _write(
            self.dir / "calibrations.csv",
            ["sensor", "model", "valid_from", "valid_to", "calibration"],
            [["S1", "M1", "2026-01-01", "2026-06-30", "valid"]],
        )
        report = run_review(load_dataset(self.dir))
        self.assertIn("calibration_uncovered", report["walls"]["A1"]["evidence_issues"])

    def test_invalid_calibration_counts_as_gap(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review"],
            [["A1", "G1", "shelter", "S1", "invalid", "2026-05-01", "0.10", "0.40", "R1", "unknown"]],
        )
        report = run_review(load_dataset(self.dir))
        issues = report["walls"]["A1"]["evidence_issues"]
        self.assertIn("calibration_invalid", issues)


class DamageReviewTests(TempFixture):
    def _primary(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review"],
            [
                ["A1", "G1", "shelter", "S1", "valid", "2026-05-01", "0.10", "0.40", "R1", "unknown"],
                ["A2", "G1", "control", "S2", "valid", "2026-05-01", "0.50", "0.40", "R1", "stable"],
            ],
        )

    def test_missing_review_is_gap_not_intact(self) -> None:
        self._primary()
        report = run_review(load_dataset(self.dir))
        self.assertIn("A1", report["damage"]["unreviewed_walls"])
        self.assertIn("no_independent_damage_review", report["walls"]["A1"]["evidence_issues"])
        # unknown 不能进 stable 集合。
        self.assertNotIn("A1", report["damage"]["reviewed_walls"])

    def test_independent_review_table_overrides_unknown(self) -> None:
        self._primary()
        _write(
            self.dir / "damage_reviews.csv",
            ["review_id", "wall", "reviewed_at", "outcome", "independent"],
            [["D1", "A1", "2026-05-05", "stable", "yes"]],
        )
        report = run_review(load_dataset(self.dir))
        self.assertIn("A1", report["damage"]["reviewed_walls"])

    def test_same_review_window_is_one_cluster(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review"],
            [["A1", "G1", "shelter", "S1", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable"],
             ["A2", "G1", "control", "S2", "valid", "2026-05-01", "0.50", "0.40", "R1", "stable"]],
        )
        # 三条复查都在同一 14 天窗口内：只能形成 1 个簇，不能估计。
        _write(
            self.dir / "damage_reviews.csv",
            ["review_id", "wall", "reviewed_at", "outcome", "independent"],
            [
                ["D1", "A1", "2026-05-05", "stable", "yes"],
                ["D2", "A2", "2026-05-06", "new_crack", "yes"],
                ["D3", "A1", "2026-05-07", "stable", "yes"],
            ],
        )
        damage = run_review(load_dataset(self.dir))["damage"]["comparison"]
        self.assertEqual(damage["status"], "not_comparable")


class NeighbourAndComplianceTests(TempFixture):
    def test_neighbour_exposure_uses_actual_shelter_dates(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review"],
            [
                ["A1", "G1", "shelter", "S1", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable"],
                ["A2", "G1", "control", "S2", "valid", "2026-05-01", "0.50", "0.40", "R1", "stable"],
                ["A2", "G1", "control", "S2", "valid", "2026-05-11", "0.30", "0.40", "R2", "stable"],
            ],
        )
        _write(
            self.dir / "walls.csv",
            ["wall", "material", "side", "emergency", "neighbours", "sheltered", "shelter_start"],
            [
                ["A1", "brick", "north", "no", "A2", "yes", "2026-05-01"],
                ["A2", "brick", "north", "no", "A1", "no", ""],
            ],
        )
        report = run_review(load_dataset(self.dir))
        # 对照 A2 在 R1 有遮护邻墙（暴露），在 R2 仍暴露；但不能与 R1 拆成独立证据。
        self.assertEqual(report["compliance"]["walls"]["A2"]["assigned"], "control")

    def test_unknown_actual_shelter_is_gap(self) -> None:
        _write(
            self.dir / "primary.csv",
            ["wall", "drain_group", "assigned", "sensor", "calibration",
             "observed_at", "moisture", "warning", "rain", "damage_review"],
            [["A1", "G1", "shelter", "S1", "valid", "2026-05-01", "0.10", "0.40", "R1", "stable"]],
        )
        _write(
            self.dir / "walls.csv",
            ["wall", "material", "side", "emergency", "neighbours", "sheltered", "shelter_start"],
            [["A1", "brick", "north", "no", "", "", ""]],
        )
        report = run_review(load_dataset(self.dir))
        self.assertIn("actual_shelter_unknown", report["walls"]["A1"]["evidence_issues"])


class PlanTests(TempFixture):
    def test_protected_schedule_blocks_candidate(self) -> None:
        dataset = load_dataset(EXAMPLE)
        report = run_review(dataset)
        schedules = [
            Schedule("S-X", "W8", "shelter", dt.date(2026, 6, 15), dt.date(2026, 9, 15), "confirmed")
        ]
        plan = plan_next_round(dataset, report, schedules)
        option_walls = [
            {a["wall"] for a in option["assignments"]}
            for option in plan["selected_options"] + plan["alternatives_blocked"]
        ]
        self.assertFalse(any("W8" in walls for walls in option_walls))
        self.assertIn("W8", {s["wall"] for s in plan["constraints"]["protected_schedules"]})

    def test_adjacent_neighbour_control_is_blocked(self) -> None:
        dataset = load_dataset(EXAMPLE)
        dataset.candidates.append(
            {
                "wall": "W20",
                "material": "brick",
                "side": "north",
                "drain_group": "G1",
                "available_from": dt.date(2026, 6, 15),
                "available_to": dt.date(2026, 9, 15),
                "neighbours": ("W21",),
                "emergency": None,
                "note": None,
                "row": 99,
            }
        )
        dataset.candidates.append(
            {
                "wall": "W21",
                "material": "brick",
                "side": "north",
                "drain_group": "G1",
                "available_from": dt.date(2026, 6, 15),
                "available_to": dt.date(2026, 9, 15),
                "neighbours": ("W20",),
                "emergency": None,
                "note": None,
                "row": 100,
            }
        )
        report = run_review(dataset)
        plan = plan_next_round(dataset, report)
        blocked_pairs = [
            {a["wall"] for a in option["assignments"]}
            for option in plan["alternatives_blocked"]
        ]
        self.assertIn({"W20", "W21"}, blocked_pairs)
        reason_types = {
            conflict["type"]
            for option in plan["alternatives_blocked"]
            if {a["wall"] for a in option["assignments"]} == {"W20", "W21"}
            for conflict in option["conflicts"]
        }
        self.assertIn("adjacent_arms", reason_types)

    def test_emergency_wall_never_used_as_control(self) -> None:
        dataset = load_dataset(EXAMPLE)
        report = run_review(dataset)
        plan = plan_next_round(dataset, report)
        assigned_walls = {
            a["wall"]
            for option in plan["selected_options"] + plan["arm_extension_options"]
            for a in option["assignments"]
        }
        self.assertNotIn("W5", assigned_walls)
        self.assertIn("W5", plan["constraints"]["emergency_walls"])

    def test_unfillable_gaps_are_reported(self) -> None:
        dataset = load_dataset(EXAMPLE)
        report = run_review(dataset)
        plan = plan_next_round(dataset, report)
        # W6/W7 的设备重叠没有可补的校准记录，必须如实列为补不齐。
        self.assertTrue(plan["unfillable_gaps"])
        self.assertEqual(
            plan["summary"]["gaps_total"],
            plan["summary"]["gaps_coverable"] + plan["summary"]["gaps_unfillable"],
        )


class AuditAndReopenTests(TempFixture):
    def _schedules(self) -> list[Schedule]:
        return [
            Schedule(
                "S-001",
                "W1",
                "shelter",
                dt.date(2026, 5, 1),
                dt.date(2026, 9, 15),
                "confirmed",
                dt.date(2026, 4, 20),
                {"condition": "brick/north", "decision": "unknown_insufficient"},
            ),
            Schedule(
                "S-002",
                "W10",
                "control",
                dt.date(2026, 6, 15),
                dt.date(2026, 9, 15),
                "signed",
                dt.date(2026, 6, 1),
                {"condition": "stone/west", "decision": "unknown_insufficient"},
            ),
        ]

    def test_audit_keeps_signing_and_current_comparison(self) -> None:
        dataset = load_dataset(EXAMPLE)
        report = run_review(dataset)
        audits = audit_schedules(self._schedules(), report)
        self.assertEqual(len(audits), 2)
        first = audits[0]
        self.assertEqual(first["basis_at_signing"]["condition"], "brick/north")
        self.assertIsNotNone(first["current_comparison"])
        self.assertTrue(first["still_protected"])
        # 旧条件键 brick/north 能解析到新的材料级 brick，并带有当前决定。
        self.assertIn("decision", first["current_comparison"])

    def test_reopen_lists_changed_conditions_and_protected_schedules(self) -> None:
        previous = {
            "recommendation": {
                "conditions": {
                    "brick": {"decision": "unknown_insufficient"},
                    "stone": {"decision": "not_supported"},
                }
            }
        }
        current = {
            "recommendation": {
                "conditions": {
                    "brick": {"decision": "extension_supported"},
                    "stone": {"decision": "not_supported"},
                }
            }
        }
        result = reopen_analysis(previous, current, self._schedules())
        self.assertEqual([c["condition"] for c in result["changed_conditions"]], ["brick"])
        affected = {s["schedule_id"] for s in result["protected_but_affected_schedules"]}
        unaffected = {s["schedule_id"] for s in result["unaffected_schedules"]}
        self.assertEqual(affected, {"S-001"})
        self.assertEqual(unaffected, {"S-002"})
        self.assertTrue(all("仍受保护" in s["treatment"] for s in result["protected_but_affected_schedules"]))


class ExampleFixtureTests(unittest.TestCase):
    def test_example_runs_end_to_end(self) -> None:
        dataset = load_dataset(EXAMPLE)
        report = run_review(dataset)
        plan = plan_next_round(dataset, report)
        self.assertTrue(report["event_coverage"])
        self.assertIn("summary", plan)


if __name__ == "__main__":
    unittest.main()

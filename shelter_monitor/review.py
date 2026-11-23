"""按抽签组做的因果复盘。

本模块只回答规程允许的问题：
  1. 按原抽签组（intention-to-treat）比较遮护与对照；
  2. 以"每场连续降雨"为一个共同事件；同一事件内先按排水组汇总，
     再对排水组等权平均，最后以事件为簇做区间与置换；
     ——同场雨、同排水组内的墙段与时段都不算独立证据；
  3. 实际是否搭棚只用来解释遵从，不用于重划组；
  4. 邻墙暴露单独作为解释变量，只在对照意向墙内比较；
  5. 校准未覆盖、设备重叠、无独立病害复查、无有效受雨时段的墙段
     一律记为证据缺口，不记为干燥或无损。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import __version__
from .data import CalibrationEvent, Dataset, Observation, Wall
from .stats import event_risk_difference

DEFAULT_NEIGHBOUR_MARGIN = 0.05
"""工具默认的邻墙风险界限。规程没有给数值时必须由业务确认，不能当成既定政策。"""

FAILURE_DAMAGE = {"new_crack", "urgent", "failure", "collapse", "deteriorating"}
STABLE_DAMAGE = {"stable", "none", "intact", "no_change"}
URGENT_DAMAGE = {"urgent", "collapse"}
DAMAGE_WINDOW_DAYS = 14
"""独立复查的聚类窗口：同一窗口内的复查不是独立重复。"""


def _review_window_key(when: dt.date) -> str:
    """把复查日期归入固定 14 天桶，避免相隔一两天的复查被当成独立窗口。"""
    bucket = when.toordinal() // DAMAGE_WINDOW_DAYS
    start = dt.date.fromordinal(bucket * DAMAGE_WINDOW_DAYS)
    return start.isoformat()


def _sha256_files(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        try:
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    digest.update(chunk)
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationStatus:
    status: str  # "valid" / "invalid" / "uncovered" / "missing"
    detail: str


def calibration_status(obs: Observation, events: list[CalibrationEvent]) -> CalibrationStatus:
    """校准只对文件声明的设备与日期生效。

    有校准文件时，任何未被覆盖的读数都记为缺口，而不是默认有效。
    没有校准文件时，退回主表逐行声明；未声明的仍记为缺口。
    """
    if events:
        covering = [
            e
            for e in events
            if e.covers(obs.observed_at, obs.sensor)
            and (obs.model is None or e.model is None or e.model == obs.model)
        ]
        if not covering:
            return CalibrationStatus("uncovered", "校准文件未覆盖该设备/日期")
        if any(e.calibration == "invalid" for e in covering):
            return CalibrationStatus("invalid", "校准文件声明该日期/设备无效")
        if any(e.calibration == "valid" for e in covering):
            return CalibrationStatus("valid", "校准文件覆盖且有效")
        return CalibrationStatus("uncovered", "校准文件覆盖但未声明有效")
    if obs.calibration == "valid":
        return CalibrationStatus("valid", "主表逐行声明有效（无独立校准文件）")
    if obs.calibration == "invalid":
        return CalibrationStatus("invalid", "主表逐行声明无效")
    return CalibrationStatus("missing", "主表未声明校准状态")


def _wall_shelter_active(wall: Wall, when: dt.date | None) -> bool | None:
    """观察日该墙是否在实际遮护期。

    sheltered=False 明确表示从不遮护；shelter_start/end 给出施工窗口。
    两者都缺时返回 None（未知），不能当成"无棚"。
    """
    if wall.sheltered is False:
        return False
    if when is None or wall.shelter_start is None:
        return None
    if when < wall.shelter_start:
        return False
    if wall.shelter_end is not None and when > wall.shelter_end:
        return False
    return True


@dataclass
class DamageReview:
    wall: str
    reviewed_at: dt.date | None
    outcome: str | None
    independent: bool
    source: str
    review_id: str | None = None

    @property
    def counted(self) -> bool:
        return (
            self.independent
            and self.outcome is not None
            and (self.outcome in FAILURE_DAMAGE or self.outcome in STABLE_DAMAGE)
        )


@dataclass
class PreparedObservation:
    obs: Observation
    calibration: CalibrationStatus
    device_overlap: bool
    moisture_usable: bool
    exceeds_warning: bool | None
    actual_shelter: bool | None
    neighbour_exposed: bool | None
    material: str
    side: str

    @property
    def event(self) -> str:
        return self.obs.rain_event or "no_event"

    @property
    def arm(self) -> str:
        return self.obs.assigned

    @property
    def cluster_key(self) -> str:
        return f"{self.event}|{self.obs.drain_group}"


@dataclass
class ReviewResult:
    dataset: Dataset
    prepared: list[PreparedObservation]
    damage_reviews: list[DamageReview]
    device_overlaps: list[dict[str, Any]]
    neighbour_margin: float
    margin_source: str
    generated_at: str
    input_hash: str

    report: dict[str, Any] = field(default_factory=dict)


def _detect_device_overlaps(observations: list[Observation]) -> tuple[list[dict[str, Any]], set[int]]:
    """同一设备同一日期的多条记录：共享/重复读数，不能当独立证据。

    区分两种情况：
      * 多个墙段共用一台设备：全部剔除，无法判断各自读数归属；
      * 同一墙段同设备同日的重复登记：保留首条，其余作为重复剔除。
    """
    groups: dict[tuple[str, str], list[Observation]] = {}
    for obs in observations:
        if obs.sensor and obs.observed_at is not None:
            groups.setdefault((obs.sensor, obs.observed_at.isoformat()), []).append(obs)
    overlaps: list[dict[str, Any]] = []
    excluded_rows: set[int] = set()
    for (sensor, date), rows in sorted(groups.items()):
        walls = sorted({r.wall for r in rows})
        if len(rows) == 1:
            continue
        if len(walls) == 1:
            ordered = sorted(rows, key=lambda r: r.row)
            excluded = {r.row for r in ordered[1:]}
            overlaps.append(
                {
                    "sensor": sensor,
                    "observed_at": date,
                    "walls": walls,
                    "rows": [r.row for r in rows],
                    "retained_row": ordered[0].row,
                    "excluded_rows": sorted(excluded),
                    "reason": "同一墙段同一设备同日的重复读数，仅保留首条",
                    "not_independent": True,
                }
            )
            excluded_rows |= excluded
        else:
            overlaps.append(
                {
                    "sensor": sensor,
                    "observed_at": date,
                    "walls": walls,
                    "rows": [r.row for r in rows],
                    "retained_row": None,
                    "excluded_rows": sorted(r.row for r in rows),
                    "reason": "同一设备同日被多个墙段共用，无法归属，全部剔除",
                    "not_independent": True,
                }
            )
            excluded_rows |= {r.row for r in rows}
    return overlaps, excluded_rows


def _load_damage_reviews(dataset: Dataset) -> tuple[list[DamageReview], list[str]]:
    """优先用独立复查表；没有时退回主表，但明确标为同记录复查。"""
    notes: list[str] = []
    reviews: list[DamageReview] = []
    if dataset.damage_review_rows:
        for row in dataset.damage_review_rows:
            reviews.append(
                DamageReview(
                    wall=row["wall"],
                    reviewed_at=row.get("reviewed_at"),
                    outcome=row.get("outcome"),
                    independent=row.get("independent") is True,
                    source="damage_reviews.csv",
                    review_id=row.get("review_id"),
                )
            )
        notes.append("使用独立复查表 damage_reviews.csv；同一复查窗口内的记录按一个簇处理。")
        return reviews, notes

    seen: set[tuple[str, str]] = set()
    for obs in dataset.observations:
        if obs.damage_review is None:
            continue
        key = (obs.wall, obs.observed_at.isoformat() if obs.observed_at else "")
        if key in seen:
            continue
        seen.add(key)
        reviews.append(
            DamageReview(
                wall=obs.wall,
                reviewed_at=obs.observed_at,
                outcome=obs.damage_review,
                independent=obs.damage_review != "unknown",
                source="primary.csv:damage_review",
                review_id=None,
            )
        )
    notes.append(
        "未提供独立复查表；damage_review 视为同一记录内的复查窗口，unknown 记为缺口而非无损。"
    )
    return reviews, notes


def _neighbour_exposed(dataset: Dataset, wall: Wall, when: dt.date | None) -> bool | None:
    """邻墙在观察日是否处于实际遮护期。

    True：至少一个邻墙在遮护；False：所有已知邻墙都未遮护（含"确实无邻墙"）；
    None：相邻关系或施工日期缺失，无法判断——记为暴露不确定，不默认未暴露。
    """
    if not wall.neighbours:
        return False if wall.neighbours_known else None
    any_unknown = False
    for neighbour in wall.neighbours:
        state = _wall_shelter_active(dataset.wall(neighbour), when)
        if state is True:
            return True
        if state is None:
            any_unknown = True
    return None if any_unknown else False


def prepare(dataset: Dataset, neighbour_margin: float | None = None) -> ReviewResult:
    overlaps, overlap_rows = _detect_device_overlaps(dataset.observations)
    prepared: list[PreparedObservation] = []
    for obs in dataset.observations:
        cal = calibration_status(obs, dataset.calibration_events)
        overlap = obs.row in overlap_rows
        usable = (
            cal.status == "valid"
            and not overlap
            and obs.moisture is not None
            and obs.warning is not None
            and obs.observed_at is not None
            and obs.rain_event is not None
        )
        exceeds = None
        if usable and obs.moisture is not None and obs.warning is not None:
            exceeds = obs.moisture > obs.warning
        wall = dataset.wall(obs.wall)
        prepared.append(
            PreparedObservation(
                obs=obs,
                calibration=cal,
                device_overlap=overlap,
                moisture_usable=usable,
                exceeds_warning=exceeds,
                actual_shelter=_wall_shelter_active(wall, obs.observed_at),
                neighbour_exposed=_neighbour_exposed(dataset, wall, obs.observed_at),
                material=wall.material or "unknown_material",
                side=wall.side or "unknown_side",
            )
        )

    damage_reviews, damage_notes = _load_damage_reviews(dataset)
    dataset.damage_review_notes = damage_notes  # type: ignore[attr-defined]

    margin = neighbour_margin
    source = "命令行/调用参数"
    if margin is None:
        margin, source = _margin_from_policy(dataset)
    return ReviewResult(
        dataset=dataset,
        prepared=prepared,
        damage_reviews=damage_reviews,
        device_overlaps=overlaps,
        neighbour_margin=margin,
        margin_source=source,
        generated_at=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        input_hash=_sha256_files(dataset.input_paths),
    )


def _margin_from_policy(dataset: Dataset) -> tuple[float, str]:
    for record in dataset.policy:
        if record.item in {"neighbour_risk_margin", "neighbour_margin"}:
            try:
                return float(record.rule), f"support-2.csv:{record.policy}"
            except ValueError:
                continue
    return DEFAULT_NEIGHBOUR_MARGIN, "工具默认值（规程未给出数值，需业务确认）"


# ---------------------------------------------------------------------------
# 结果折叠：墙 → 事件，避免同一墙的多个时段重复计数。
# ---------------------------------------------------------------------------


def _collapse_moisture(
    rows: list[PreparedObservation],
) -> list[tuple[str, str, str, bool]]:
    """(event, drain_group, arm, wall_success) — 每墙每事件只取一次。

    墙在该事件内任一有效受雨时段超过材料警戒线，即记为 1。
    """
    buckets: dict[tuple[str, str, str, str], list[bool]] = {}
    for p in rows:
        if not p.moisture_usable or p.exceeds_warning is None:
            continue
        buckets.setdefault((p.event, p.obs.drain_group, p.arm, p.obs.wall), []).append(
            p.exceeds_warning
        )
    return [
        (event, group, arm, any(values))
        for (event, group, arm, _wall), values in buckets.items()
    ]


def _collapse_by_flag(
    rows: list[PreparedObservation], flag: str
) -> list[tuple[str, str, str, bool]]:
    """邻墙暴露分析：flag 取 'exposed'/'not_exposed'，只在对照意向墙内。"""
    buckets: dict[tuple[str, str, str, str], list[bool]] = {}
    for p in rows:
        if p.arm != "control" or not p.moisture_usable or p.exceeds_warning is None:
            continue
        if flag == "exposed":
            if p.neighbour_exposed is not True:
                continue
            arm = "exposed"
        else:
            if p.neighbour_exposed is not False:
                continue
            arm = "not_exposed"
        buckets.setdefault((p.event, p.obs.drain_group, arm, p.obs.wall), []).append(
            p.exceeds_warning
        )
    return [
        (event, group, arm, any(values))
        for (event, group, arm, _wall), values in buckets.items()
    ]


def _neighbour_analysis(rows: list[PreparedObservation]) -> dict[str, Any]:
    """邻墙暴露 vs 未暴露（仅对照意向墙），按事件聚类。"""
    collapsed = _collapse_by_flag(rows, "exposed") + _collapse_by_flag(rows, "not_exposed")
    return event_risk_difference(collapsed, "exposed", "not_exposed")


def _damage_contrasts(
    dataset: Dataset, reviews: list[DamageReview]
) -> dict[str, Any]:
    """病害结局：只用可计数的独立复查，按复查窗口聚类。

    没有复查的墙段不进入分母，也不记为无损。
    与含水量不同，病害复查是独立观察窗口，因此按"复查窗口 × 排水组"聚类，
    不要求与降雨事件同时出现。
    """
    assignment: dict[str, str] = {}
    drain_group: dict[str, str] = {}
    for obs in dataset.observations:
        assignment[obs.wall] = obs.assigned
        if obs.drain_group:
            drain_group.setdefault(obs.wall, obs.drain_group)

    counted = [r for r in reviews if r.counted and r.reviewed_at is not None]
    buckets: dict[tuple[str, str, str, str], list[bool]] = {}
    for review in counted:
        arm = assignment.get(review.wall)
        if arm not in {"shelter", "control"}:
            continue
        window_key = _review_window_key(review.reviewed_at)  # type: ignore[arg-type]
        group = drain_group.get(review.wall, "unknown_group")
        success = review.outcome in FAILURE_DAMAGE
        buckets.setdefault((window_key, group, arm, review.wall), []).append(success)

    collapsed = [
        (window, group, arm, any(values))
        for (window, group, arm, _wall), values in buckets.items()
    ]
    result = event_risk_difference(collapsed, "shelter", "control")
    result["outcome"] = "独立复查出现新裂缝/恶化/紧急病害"
    result["window_days"] = DAMAGE_WINDOW_DAYS
    result["reviewed_walls"] = sorted({r.wall for r in counted})
    result["reviewed_but_unknown"] = sorted(
        {r.wall for r in reviews if r.independent and not r.counted}
    )
    result["unreviewed_walls"] = sorted(set(dataset.walls) - {r.wall for r in counted})
    return result


def _condition_name(condition: tuple[str, str] | str) -> str:
    if isinstance(condition, str):
        return condition
    return f"{condition[0]}/{condition[1]}"


def _wall_evidence(
    dataset: Dataset,
    prepared: list[PreparedObservation],
    damage: list[DamageReview],
) -> dict[str, Any]:
    by_wall: dict[str, list[PreparedObservation]] = {}
    for p in prepared:
        by_wall.setdefault(p.obs.wall, []).append(p)
    damage_by_wall: dict[str, list[DamageReview]] = {}
    for review in damage:
        damage_by_wall.setdefault(review.wall, []).append(review)

    walls_out: dict[str, Any] = {}
    for name in sorted(set(dataset.walls) | set(by_wall)):
        wall = dataset.wall(name)
        rows = by_wall.get(name, [])
        issues: list[str] = []
        notes: list[str] = []
        if not rows:
            issues.append("no_observations")
        usable = [p for p in rows if p.moisture_usable]
        if rows and not usable:
            if any(p.device_overlap for p in rows):
                issues.append("device_overlap_only")
                notes.append("仅有的读数来自设备重叠或重复登记，不能作为独立证据")
            if any(p.obs.rain_event is None for p in rows):
                issues.append("missing_rain_event")
                notes.append("缺少降雨事件标识，无法判定独立受雨时段")
            if any(p.calibration.status == "uncovered" for p in rows):
                issues.append("calibration_uncovered")
                notes.append("校准文件未覆盖该设备/日期")
            elif any(p.calibration.status == "invalid" for p in rows):
                issues.append("calibration_invalid")
                notes.append("校准声明无效")
            elif any(p.calibration.status == "missing" for p in rows):
                issues.append("calibration_missing")
            if not any(not p.device_overlap for p in rows):
                if "device_overlap_only" not in issues:
                    issues.append("no_independent_moisture")
            elif "device_overlap_only" not in issues:
                issues.append("no_usable_moisture")
        if wall.neighbours and any(
            dataset.wall(n).sheltered is None and dataset.wall(n).shelter_start is None
            for n in wall.neighbours
        ):
            issues.append("neighbour_construction_unknown")
        actual_shelter_known = wall.sheltered is False or wall.shelter_start is not None
        if not actual_shelter_known:
            issues.append("actual_shelter_unknown")
            notes.append("不知道是否/何时实际搭棚，遵从无法解释")
        counted = [r for r in damage_by_wall.get(name, []) if r.counted]
        if not counted:
            issues.append("no_independent_damage_review")
            if any(r.outcome == "unknown" for r in damage_by_wall.get(name, [])):
                notes.append("病害记录为 unknown，不能当作无损")
        statuses = sorted({r.outcome for r in counted if r.outcome})
        urgent = wall.emergency is True or any(s in URGENT_DAMAGE for s in statuses)
        walls_out[name] = {
            "assigned": sorted({p.obs.assigned for p in rows}),
            "material": wall.material,
            "side": wall.side,
            "neighbours": list(wall.neighbours),
            "shelter_start": wall.shelter_start.isoformat() if wall.shelter_start else None,
            "shelter_end": wall.shelter_end.isoformat() if wall.shelter_end else None,
            "sheltered": wall.sheltered,
            "actual_shelter_known": actual_shelter_known,
            "emergency": urgent,
            "damage_statuses": statuses,
            "evidence_issues": issues,
            "evidence_notes": notes,
            "usable_moisture_periods": len(usable),
        }
    return walls_out


def _compliance_summary(prepared: list[PreparedObservation]) -> dict[str, Any]:
    cells: dict[tuple[str, str], int] = {}
    per_wall: dict[str, dict[str, Any]] = {}
    for p in prepared:
        actual = p.actual_shelter
        label = "unknown" if actual is None else ("sheltered" if actual else "open")
        cells[(p.obs.assigned, label)] = cells.get((p.obs.assigned, label), 0) + 1
        entry = per_wall.setdefault(
            p.obs.wall,
            {"assigned": p.obs.assigned, "actual_dates_known": actual is not None, "states": set()},
        )
        entry["states"].add(label)
    for entry in per_wall.values():
        entry["states"] = sorted(entry["states"])
    return {
        "cells": [
            {"assigned": a, "actual": b, "observations": n}
            for (a, b), n in sorted(cells.items())
        ],
        "walls": per_wall,
        "note": "实际搭棚只解释遵从；不得据此重划原实验组，也不得用 as-treated 比较替代抽签组比较。",
    }


def _emergency_walls(dataset: Dataset, damage: list[DamageReview]) -> list[dict[str, Any]]:
    urgent_walls = {r.wall for r in damage if r.counted and r.outcome in URGENT_DAMAGE}
    out = []
    for name in sorted(set(dataset.walls) | urgent_walls):
        wall = dataset.wall(name)
        if wall.emergency is True or name in urgent_walls:
            out.append(
                {
                    "wall": name,
                    "reason": "support-1.emergency" if wall.emergency is True else "urgent_damage_review",
                    "action": "保护人员立即干预；不得为补对照而放回无棚组",
                }
            )
    return out


def _recommend(
    overall_direct: dict[str, Any],
    conditions: dict[str, Any],
    neighbour_overall: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    def direct_supported(result: dict[str, Any]) -> bool:
        return (
            result.get("status") == "estimated"
            and result.get("high") is not None
            and result["high"] < 0
            and result.get("comparable_clusters", 0) >= 2
        )

    def neighbour_excluded(result: dict[str, Any]) -> bool:
        return (
            result.get("status") == "estimated"
            and result.get("high") is not None
            and result["high"] <= margin
            and result.get("comparable_clusters", 0) >= 2
        )

    condition_decisions: dict[str, Any] = {}
    for name, entry in conditions.items():
        direct = entry["direct"]
        neigh = entry["neighbour"]
        if entry.get("usable_observations", 0) == 0:
            decision = "unknown_insufficient"
            reason = "该条件没有可用观察；缺样不能默认安全。"
        elif direct.get("status") == "single_arm":
            decision = "unknown_insufficient"
            reason = "该条件只有一个意向臂有可用观察，无法比较。"
        elif direct.get("status") != "estimated":
            decision = "unknown_insufficient"
            reason = "直接收益无法估计（缺少至少两个同时含两臂的降雨事件）。"
        elif entry.get("side_confounded"):
            decision = "unknown_insufficient"
            reason = (
                "该材料内遮护与对照的方位完全错开，材料效应无法与朝向效应分开；"
                "不能把差异归因于遮护。"
            )
        elif not direct_supported(direct):
            decision = "not_supported"
            reason = "直接收益区间未完全低于 0，不能扩展。"
        elif neigh.get("status") == "no_data" and neighbour_overall.get("status") == "estimated":
            decision = "benefit_supported_neighbour_unresolved"
            reason = "直接收益有支持，但该条件缺少邻墙暴露信息，无法排除邻墙风险。"
        elif not neighbour_excluded(neigh):
            decision = "benefit_supported_neighbour_unresolved"
            reason = "直接收益有支持，但邻墙风险无法排除到规程界限内。"
        else:
            decision = "extension_supported"
            reason = "直接收益有支持且邻墙风险上界在规程界限内；仅适用于同类条件。"
        condition_decisions[name] = {
            "decision": decision,
            "reason": reason,
            "direct": direct,
            "neighbour": neigh,
            "usable_observations": entry.get("usable_observations", 0),
            "events_with_both_arms": direct.get("comparable_clusters", 0),
        }

    if not condition_decisions:
        overall_decision = "unknown_insufficient"
        reason = "没有任何条件具备可用比较。"
    elif any(d["decision"] == "extension_supported" for d in condition_decisions.values()):
        overall_decision = "partial_extension_supported"
        reason = "仅部分条件满足直接收益与邻墙界限；不得外推到其他材料/朝向。"
    elif any(d["decision"] == "benefit_supported_neighbour_unresolved" for d in condition_decisions.values()):
        overall_decision = "hold_neighbour_risk"
        reason = "存在直接收益但邻墙风险未排除的条件，暂不扩展。"
    elif any(d["decision"] == "not_supported" for d in condition_decisions.values()):
        overall_decision = "not_supported"
        reason = "已有条件的直接收益未被证据支持。"
    else:
        overall_decision = "unknown_insufficient"
        reason = "所有条件都缺可比证据，不能默认安全。"

    return {
        "decision": overall_decision,
        "reason": reason,
        "neighbour_margin": margin,
        "conditions": condition_decisions,
    }


def _coverage(prepared: list[PreparedObservation]) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for p in prepared:
        entry = events.setdefault(
            p.event, {"drain_groups": set(), "walls": set(), "usable": 0}
        )
        entry["drain_groups"].add(p.obs.drain_group)
        entry["walls"].add(p.obs.wall)
        if p.moisture_usable:
            entry["usable"] += 1
    return [
        {
            "event": event,
            "drain_groups": sorted(entry["drain_groups"]),
            "clusters": len(entry["drain_groups"]),
            "walls": sorted(entry["walls"]),
            "usable_periods": entry["usable"],
        }
        for event, entry in sorted(events.items())
    ]


def build_report(result: ReviewResult) -> dict[str, Any]:
    dataset = result.dataset
    prepared = result.prepared
    damage = result.damage_reviews
    wall_info = _wall_evidence(dataset, prepared, damage)

    overall_rows = _collapse_moisture(prepared)
    overall_direct = event_risk_difference(overall_rows, "shelter", "control")
    overall_direct["outcome"] = "有效受雨观察时段内含水量超过该墙段材料警戒线"
    overall_direct["unit"] = "每墙每场雨折叠为一次二值结果；事件内按排水组等权"

    conditions: dict[str, Any] = {}
    all_materials = sorted({p.material for p in prepared})
    for material in all_materials:
        rows = [p for p in prepared if p.material == material]
        direct = event_risk_difference(_collapse_moisture(rows), "shelter", "control")
        direct["outcome"] = "有效受雨观察时段内含水量超过该墙段材料警戒线"
        neigh = _neighbour_analysis(rows)
        sides_shelter = sorted({p.side for p in rows if p.arm == "shelter"})
        sides_control = sorted({p.side for p in rows if p.arm == "control"})
        confounded = bool(sides_shelter) and bool(sides_control) and not (
            set(sides_shelter) & set(sides_control)
        )
        conditions[material] = {
            "material": material,
            "sides": sorted({p.side for p in rows}),
            "shelter_sides": sides_shelter,
            "control_sides": sides_control,
            "side_confounded": confounded,
            "usable_observations": len([p for p in rows if p.moisture_usable]),
            "walls": sorted({p.obs.wall for p in rows}),
            "events": sorted({p.event for p in rows if p.moisture_usable}),
            "direct": direct,
            "neighbour": neigh,
        }

    # 墙段表里没有观察的材料也要出现在缺口里。
    for name, info in wall_info.items():
        material = info.get("material") or "unknown_material"
        if material not in conditions:
            conditions[material] = {
                "material": material,
                "sides": [info.get("side")] if info.get("side") else [],
                "shelter_sides": [],
                "control_sides": [],
                "side_confounded": False,
                "usable_observations": 0,
                "walls": [name],
                "events": [],
                "direct": {"status": "no_data", "clusters": 0, "comparable_clusters": 0},
                "neighbour": {"status": "no_data", "clusters": 0, "comparable_clusters": 0},
            }

    neighbour_overall = _neighbour_analysis(prepared)
    recommendation = _recommend(overall_direct, conditions, neighbour_overall, result.neighbour_margin)

    damage_result = _damage_contrasts(dataset, damage)

    gaps: list[dict[str, Any]] = []
    for name, info in sorted(wall_info.items()):
        for issue in info["evidence_issues"]:
            gaps.append(
                {
                    "wall": name,
                    "issue": issue,
                    "counted_as": "evidence_gap",
                    "detail": "；".join(info["evidence_notes"]),
                }
            )
    for overlap in result.device_overlaps:
        gaps.append(
            {
                "wall": ",".join(overlap["walls"]),
                "issue": "device_overlap",
                "counted_as": "evidence_gap",
                "detail": overlap,
            }
        )

    report = {
        "tool": "shelter-monitor",
        "version": __version__,
        "generated_at": result.generated_at,
        "input_hash": result.input_hash,
        "inputs": dataset.sources,
        "policy": {
            "records": [
                {"policy": r.policy, "item": r.item, "rule": r.rule} for r in dataset.policy
            ],
            "neighbour_margin": result.neighbour_margin,
            "neighbour_margin_source": result.margin_source,
            "grouping_rule": "按抽签 assigned 组评估；实际搭棚仅解释遵从。",
            "independence_rule": "同一场连续降雨为一个共同事件；事件内按排水组等权，事件为簇。",
        },
        "event_coverage": _coverage(prepared),
        "intention_comparison": {
            "overall": overall_direct,
            "by_condition": {name: entry["direct"] for name, entry in conditions.items()},
            "outcome": overall_direct["outcome"],
        },
        "neighbour_exposure": {
            "overall": neighbour_overall,
            "by_condition": {name: entry["neighbour"] for name, entry in conditions.items()},
            "definition": "对照意向墙中，观察日有相邻墙处于实际遮护期者视为暴露；相邻关系缺失记为不确定。",
        },
        "damage": {
            "comparison": damage_result,
            "reviewed_walls": damage_result["reviewed_walls"],
            "unreviewed_walls": damage_result["unreviewed_walls"],
            "notes": getattr(dataset, "damage_review_notes", []),
            "rule": "病害只在独立复查窗口内计数；unknown 与无复查记为证据缺口，不记为无损。",
        },
        "compliance": _compliance_summary(prepared),
        "emergency": _emergency_walls(dataset, damage),
        "walls": wall_info,
        "conditions": conditions,
        "recommendation": recommendation,
        "evidence_gaps": gaps,
        "device_overlaps": result.device_overlaps,
    }
    result.report = report
    return report


def run_review(dataset: Dataset, neighbour_margin: float | None = None) -> dict[str, Any]:
    return build_report(prepare(dataset, neighbour_margin))


def _fmt_risk(result: dict[str, Any] | None) -> str:
    if not result:
        return "无数据"
    status = result.get("status")
    if status == "estimated":
        return (
            f"{result['point']:+.3f} [{result['low']:+.3f}, {result['high']:+.3f}]"
            f"，可比事件={result.get('comparable_clusters', 0)}，p={result.get('p', 1):.3f}"
        )
    if status == "single_arm":
        return f"只有单一意向臂（缺 {', '.join(result.get('missing_arm', []))}），不可比较"
    if status == "not_comparable":
        detail = result.get("why", "同场雨不能当独立重复")
        return f"不可估计：可比的组内对照事件 {result.get('comparable_clusters', 0)} 个（{detail}）"
    if status == "no_data":
        return "无可用数据"
    return f"无法估计（{status}）"


format_risk = _fmt_risk
"""公开的格式化函数，供 CLI 复用同一种证据表述。"""


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 遮护政策复盘（按抽签组）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 输入：{', '.join(report['inputs'])}")
    lines.append(
        f"- 邻墙风险界限：{report['policy']['neighbour_margin']}"
        f"（{report['policy']['neighbour_margin_source']}）"
    )
    lines.append(f"- 结论：**{report['recommendation']['decision']}** — {report['recommendation']['reason']}")
    lines.append(f"- 数据指纹：`{report['input_hash'][:16]}`")
    lines.append("")
    lines.append("## 主比较（意向组，按降雨事件聚类）")
    lines.append(f"- 风险差（遮护 − 对照）：{_fmt_risk(report['intention_comparison']['overall'])}")
    lines.append(f"- 结局：{report['intention_comparison']['outcome']}")
    lines.append("")
    lines.append("## 邻墙暴露（仅对照意向墙，单独解释）")
    lines.append(f"- 风险差（暴露 − 未暴露）：{_fmt_risk(report['neighbour_exposure']['overall'])}")
    lines.append(f"- 定义：{report['neighbour_exposure']['definition']}")
    lines.append("")
    lines.append("## 表面病害（独立复查窗口）")
    damage = report["damage"]
    lines.append(f"- 风险差（遮护 − 对照）：{_fmt_risk(damage['comparison'])}")
    lines.append(f"- 已复查：{', '.join(damage['reviewed_walls']) or '无'}")
    if damage.get("reviewed_but_unknown"):
        lines.append(f"- 有复查但结果未确认（记为缺口）：{', '.join(damage['reviewed_but_unknown'])}")
    lines.append(f"- 未复查（记为缺口，不记为无损）：{', '.join(damage['unreviewed_walls']) or '无'}")
    for note in damage["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 实际遵从（只解释遵从，不重划组）")
    if not report["compliance"]["cells"]:
        lines.append("- 无观察")
    for cell in report["compliance"]["cells"]:
        lines.append(f"- 抽签 {cell['assigned']}，实际 {cell['actual']}：{cell['observations']} 条观察")
    lines.append(f"- {report['compliance']['note']}")
    lines.append("")
    lines.append("## 条件结论")
    for name, entry in sorted(report["recommendation"]["conditions"].items()):
        lines.append(f"- `{name}`：**{entry['decision']}**")
        lines.append(f"  - 直接收益：{_fmt_risk(entry['direct'])}")
        lines.append(f"  - 邻墙暴露：{_fmt_risk(entry['neighbour'])}")
        lines.append(f"  - {entry['reason']}")
    lines.append("")
    lines.append("## 事件覆盖")
    if not report["event_coverage"]:
        lines.append("- 无事件")
    for coverage in report["event_coverage"]:
        lines.append(
            f"- {coverage['event']}：排水组 {', '.join(coverage['drain_groups'])}；"
            f"有效时段 {coverage['usable_periods']}；墙段 {', '.join(coverage['walls'])}"
        )
    lines.append("")
    lines.append("## 证据缺口（不记为干燥/无损）")
    if not report["evidence_gaps"]:
        lines.append("- 无")
    else:
        seen: set[tuple[str, str]] = set()
        for gap in report["evidence_gaps"]:
            key = (gap["wall"], gap["issue"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{gap['wall']}`：{gap['issue']}")
    lines.append("")
    if report["emergency"]:
        lines.append("## 紧急线（立即干预）")
        for item in report["emergency"]:
            lines.append(f"- `{item['wall']}`：{item['reason']}；{item['action']}")
        lines.append("")
    lines.append("## 设备重叠读数（不构成独立证据）")
    if not report["device_overlaps"]:
        lines.append("- 无")
    else:
        for overlap in report["device_overlaps"]:
            lines.append(
                f"- 设备 {overlap['sensor']} 于 {overlap['observed_at']} 出现在 "
                f"{', '.join(overlap['walls'])}：{overlap['reason']}"
            )
    lines.append("")
    return "\n".join(lines)


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)

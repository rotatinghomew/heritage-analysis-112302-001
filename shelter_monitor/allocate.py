"""下一轮布点：从候选墙段与现有工期中找能填补证据缺口的观察组合。

约束优先级：
  1. 已达紧急线或已确认的保护作业不可挪动；
  2. 不得把墙段放回无棚组来补对照；
  3. 相邻墙段的安排若会污染已有对照，必须显式标为冲突而不是静默采用；
  4. 同一设备同期的重叠读数不构成新的独立证据；
  5. 剩余资源补不齐的缺口要如实列出，不能记为"暂无风险"。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .data import Dataset


@dataclass
class Schedule:
    schedule_id: str
    wall: str
    arm: str  # "shelter" / "control"
    start: dt.date
    end: dt.date
    status: str  # "signed" / "confirmed"
    signed_at: dt.date | None = None
    comparison_snapshot: dict[str, Any] = field(default_factory=dict)
    note: str | None = None

    @property
    def protected(self) -> bool:
        return self.status in {"signed", "confirmed"}


def schedules_from_rows(rows: list[dict]) -> list[Schedule]:
    out: list[Schedule] = []
    for row in rows:
        start = row.get("start")
        end = row.get("end")
        if start is None or end is None:
            continue
        out.append(
            Schedule(
                schedule_id=row["schedule_id"],
                wall=row["wall"],
                arm=row["arm"],
                start=start,
                end=end,
                status=row["status"],
                signed_at=row.get("signed_at"),
                comparison_snapshot=row.get("comparison_snapshot") or {},
                note=row.get("note"),
            )
        )
    return out


def _windows_overlap(a_from, a_to, b_from, b_to) -> bool:
    if a_from is None or a_to is None or b_from is None or b_to is None:
        return True  # 工期未知时保守视为重叠
    return not (a_to < b_from or b_to < a_from)


def _condition_key(material: str | None, side: str | None = None) -> str:
    """与复盘报告一致：材料是"同类条件"的主键。"""
    return material or "unknown_material"


def _iso(value: dt.date | None) -> str | None:
    return value.isoformat() if value else None


def _candidate_conflicts(
    dataset: Dataset,
    candidate: dict,
    arm: str,
    schedules: list[Schedule],
    chosen: dict[str, str],
) -> list[dict[str, str]]:
    """候选墙加入某一臂时的全部冲突。空列表表示可执行。"""
    wall = candidate["wall"]
    conflicts: list[dict[str, str]] = []
    info = dataset.wall(wall)
    neighbours = set(info.neighbours) | set(candidate.get("neighbours") or ())
    start = candidate.get("available_from")
    end = candidate.get("available_to")

    if info.emergency is True or candidate.get("emergency") is True:
        conflicts.append(
            {"type": "emergency_wall", "detail": "已达紧急线；只能由保护人员立即干预，不得用于补对照。"}
        )

    for schedule in schedules:
        if schedule.wall != wall or not schedule.protected:
            continue
        if _windows_overlap(start, end, schedule.start, schedule.end):
            conflicts.append(
                {
                    "type": "protected_schedule",
                    "detail": f"已签排期 {schedule.schedule_id}（{schedule.status}）占用该工期，不可挪动。",
                }
            )

    for neighbour in neighbours:
        for schedule in schedules:
            if schedule.wall != neighbour or not schedule.protected:
                continue
            if not _windows_overlap(start, end, schedule.start, schedule.end):
                continue
            if arm == "control" and schedule.arm == "shelter":
                conflicts.append(
                    {
                        "type": "neighbour_runoff_control",
                        "detail": f"邻墙 {neighbour} 已签遮护排期 {schedule.schedule_id}；本墙作对照会被径流污染。",
                    }
                )
            if arm == "shelter" and schedule.arm == "control":
                conflicts.append(
                    {
                        "type": "destroys_existing_control",
                        "detail": f"邻墙 {neighbour} 是已签对照排期 {schedule.schedule_id}；本墙遮护会摧毁该对照。",
                    }
                )
        if arm == "control" and chosen.get(neighbour) == "shelter":
            conflicts.append(
                {
                    "type": "neighbour_runoff_control",
                    "detail": f"同批计划把邻墙 {neighbour} 设为遮护，本墙作对照不独立。",
                }
            )
        if arm == "shelter" and chosen.get(neighbour) == "control":
            conflicts.append(
                {
                    "type": "destroys_existing_control",
                    "detail": f"同批计划把邻墙 {neighbour} 设为对照；本墙遮护会摧毁该对照。",
                }
            )
    return conflicts


def _open_gaps(report: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(gap: dict[str, Any]) -> None:
        if gap["gap_id"] in seen:
            return
        seen.add(gap["gap_id"])
        gaps.append(gap)

    for name, entry in sorted(report.get("recommendation", {}).get("conditions", {}).items()):
        if entry["decision"] != "extension_supported":
            add(
                {
                    "gap_id": f"comparison:{name}",
                    "kind": "comparison",
                    "condition": name,
                    "reason": entry["reason"],
                    "needs": "同排水组、同一工期窗内同时具备遮护与对照，且至少两个降雨事件的有效观察",
                }
            )

    reviewed = set(report.get("damage", {}).get("reviewed_walls", []))
    for name, info in sorted(report.get("walls", {}).items()):
        if name not in reviewed:
            add(
                {
                    "gap_id": f"damage_review:{name}",
                    "kind": "damage_review",
                    "wall": name,
                    "reason": "没有独立复查窗口的病害记录",
                    "needs": "独立复查；不能把 unknown 记为无损",
                }
            )
        if not info.get("actual_shelter_known", info.get("shelter_start") is not None):
            add(
                {
                    "gap_id": f"actual_shelter:{name}",
                    "kind": "actual_shelter",
                    "wall": name,
                    "reason": "实际是否/何时搭棚未知，遵从解释不完整",
                    "needs": "补登记实际遮护状态与日期（不能默认无棚）",
                }
            )
        for issue in info.get("evidence_issues", []):
            if issue.startswith("calibration") or issue in {"no_calibrated_moisture", "device_overlap_only"}:
                add(
                    {
                        "gap_id": f"calibration:{name}",
                        "kind": "calibration",
                        "wall": name,
                        "reason": issue,
                        "needs": "覆盖该设备/日期的换机校准记录",
                    }
                )
    return gaps


def plan_next_round(
    dataset: Dataset,
    report: dict[str, Any],
    schedules: list[Schedule] | None = None,
) -> dict[str, Any]:
    schedules = schedules or []
    candidates = getattr(dataset, "candidates", [])
    gaps = _open_gaps(report)

    protected_walls = {s.wall for s in schedules if s.protected}
    emergency_walls = {name for name, info in report.get("walls", {}).items() if info.get("emergency")}
    blocked_walls = protected_walls | emergency_walls

    by_condition: dict[str, list[dict]] = {}
    for candidate in candidates:
        key = _condition_key(candidate.get("material"))
        by_condition.setdefault(key, []).append(candidate)

    options: list[dict[str, Any]] = []
    for gap in gaps:
        if gap["kind"] != "comparison":
            continue
        condition = gap["condition"]
        pool = [
            c
            for c in by_condition.get(condition, [])
            if c["wall"] not in blocked_walls
        ]
        groups: dict[str, list[dict]] = {}
        for candidate in pool:
            groups.setdefault(candidate.get("drain_group") or "unknown_group", []).append(candidate)

        for group, members in sorted(groups.items()):
            for shelter in members:
                for control in members:
                    if shelter["wall"] == control["wall"]:
                        continue
                    if not _windows_overlap(
                        shelter.get("available_from"),
                        shelter.get("available_to"),
                        control.get("available_from"),
                        control.get("available_to"),
                    ):
                        continue
                    chosen = {shelter["wall"]: "shelter", control["wall"]: "control"}
                    conflicts: list[dict[str, str]] = []
                    conflicts.extend(_candidate_conflicts(dataset, shelter, "shelter", schedules, chosen))
                    conflicts.extend(_candidate_conflicts(dataset, control, "control", schedules, chosen))
                    if control["wall"] in (
                        set(dataset.wall(shelter["wall"]).neighbours)
                        | set(shelter.get("neighbours") or ())
                    ):
                        conflicts.append(
                            {
                                "type": "adjacent_arms",
                                "detail": "同批遮护/对照两墙相邻，径流会转移，不能当作独立对照。",
                            }
                        )
                    start = max(
                        shelter.get("available_from") or dt.date.min,
                        control.get("available_from") or dt.date.min,
                    )
                    end = min(
                        shelter.get("available_to") or dt.date.max,
                        control.get("available_to") or dt.date.max,
                    )
                    options.append(
                        {
                            "option_id": f"comparison:|{condition}|{group}|{shelter['wall']}+{control['wall']}",
                            "fills": [gap["gap_id"]],
                            "kind": "comparison",
                            "condition": condition,
                            "drain_group": group,
                            "window": {"start": _iso(start), "end": _iso(end)},
                            "assignments": [
                                {
                                    "wall": shelter["wall"],
                                    "arm": "shelter",
                                    "available_from": _iso(shelter.get("available_from")),
                                    "available_to": _iso(shelter.get("available_to")),
                                },
                                {
                                    "wall": control["wall"],
                                    "arm": "control",
                                    "available_from": _iso(control.get("available_from")),
                                    "available_to": _iso(control.get("available_to")),
                                },
                            ],
                            "conflicts": conflicts,
                            "feasible": not conflicts,
                            "why": (
                                "可在同一排水组、同一工期窗内形成意向臂对照。"
                                if not conflicts
                                else "存在冲突，需先解除或改配。"
                            ),
                        }
                    )

    # 同一对墙若同时出现可行与不可行，保留可行的那条。
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for option in options:
        walls = tuple(sorted(a["wall"] for a in option["assignments"]))
        grouped.setdefault((option["drain_group"], *walls), []).append(option)
    options = [sorted(v, key=lambda o: not o["feasible"])[0] for v in grouped.values()]

    feasible = [o for o in options if o["feasible"]]
    blocked = [o for o in options if not o["feasible"]]
    feasible.sort(key=lambda o: (o["drain_group"], o["option_id"]))

    # 贪心：先按条件覆盖，再列出剩余。
    filled: set[str] = set()
    selected: list[dict[str, Any]] = []
    used_walls: set[str] = set()
    for option in feasible:
        walls = {a["wall"] for a in option["assignments"]}
        if walls & used_walls:
            continue
        if all(g in filled for g in option["fills"]):
            continue
        selected.append(option)
        used_walls |= walls
        filled.update(option["fills"])

    # 单臂延伸：能改善证据但不足以单独补齐可比条件。
    extensions: list[dict[str, Any]] = []
    for gap in gaps:
        if gap["kind"] != "comparison":
            continue
        for candidate in by_condition.get(gap["condition"], []):
            if candidate["wall"] in blocked_walls | used_walls:
                continue
            for arm in ("shelter", "control"):
                conflicts = _candidate_conflicts(dataset, candidate, arm, schedules, {})
                extensions.append(
                    {
                        "option_id": f"arm-extension|{gap['condition']}|{candidate['wall']}|{arm}",
                        "fills": [],
                        "kind": "arm_extension",
                        "condition": gap["condition"],
                        "assignments": [
                            {
                                "wall": candidate["wall"],
                                "arm": arm,
                                "available_from": _iso(candidate.get("available_from")),
                                "available_to": _iso(candidate.get("available_to")),
                            }
                        ],
                        "conflicts": conflicts,
                        "feasible": not conflicts,
                        "why": "只补一条意向臂，仍不足以形成同簇可比对照。",
                    }
                )

    comparison_gaps = [g for g in gaps if g["kind"] == "comparison"]
    unfillable = [
        gap
        for gap in gaps
        if gap["gap_id"] not in filled
    ]

    return {
        "as_of": report.get("generated_at"),
        "constraints": {
            "protected_schedules": [
                {
                    "schedule_id": s.schedule_id,
                    "wall": s.wall,
                    "arm": s.arm,
                    "start": s.start.isoformat(),
                    "end": s.end.isoformat(),
                    "status": s.status,
                }
                for s in schedules
                if s.protected
            ],
            "emergency_walls": sorted(emergency_walls),
            "rule": "紧急墙与已确认保护作业不可挪动；不得把墙段放回无棚组补对照。",
        },
        "open_gaps": gaps,
        "selected_options": selected,
        "alternatives_blocked": blocked,
        "arm_extension_options": [o for o in extensions if o["feasible"]],
        "unfillable_gaps": unfillable,
        "summary": {
            "comparison_gaps": len(comparison_gaps),
            "comparison_gaps_covered": len([g for g in comparison_gaps if g["gap_id"] in filled]),
            "blocked_alternatives": len(blocked),
            "gaps_total": len(gaps),
            "gaps_coverable": len(filled),
            "gaps_unfillable": len(unfillable),
        },
    }


# ---------------------------------------------------------------------------
# 签排期审计：保护人员能回查"签排期当时依赖的比较"与"当前比较"。
# ---------------------------------------------------------------------------


def _resolve_condition(report: dict[str, Any], condition: str | None) -> Any:
    """解析签署快照里的条件键。

    兼容两种写法：新报告的 "brick"，以及旧快照的 "brick/north"。
    旧键先按完整键找，找不到再退回材料键，绝不把不同材料混在一起。
    """
    conditions = report.get("recommendation", {}).get("conditions", {})
    if not condition:
        return None
    if condition in conditions:
        return conditions[condition]
    material = condition.split("/", 1)[0]
    return conditions.get(material)


def audit_schedules(
    schedules: list[Schedule],
    report: dict[str, Any],
    schedule_ids: list[str] | None = None,
    *,
    structural_change: bool = False,
) -> list[dict[str, Any]]:
    """逐份已签排期列出签署时的比较快照与当前比较，允许两者并存。"""
    selected = [
        s
        for s in schedules
        if s.protected and (schedule_ids is None or s.schedule_id in schedule_ids)
    ]
    out: list[dict[str, Any]] = []
    for schedule in selected:
        condition = schedule.comparison_snapshot.get("condition")
        if condition:
            current = _resolve_condition(report, condition)
        else:
            current = report.get("intention_comparison", {}).get("overall")
        out.append(
            {
                "schedule_id": schedule.schedule_id,
                "wall": schedule.wall,
                "arm": schedule.arm,
                "window": {"start": schedule.start.isoformat(), "end": schedule.end.isoformat()},
                "status": schedule.status,
                "signed_at": _iso(schedule.signed_at),
                "basis_at_signing": schedule.comparison_snapshot,
                "current_comparison": current,
                "snapshot_condition_resolved": condition,
                "snapshot_condition_note": (
                    "签署快照的条件键按材料解析；旧键如 brick/north 会先精确匹配再退回 brick。"
                    if condition
                    else "签署快照未记录条件，退回报表总体比较。"
                ),
                "current_input_hash": report.get("input_hash"),
                "devices_changed_since_signing": report.get("device_overlaps", []),
                "structural_change_since_signing": structural_change,
                "still_protected": True,
                "note": (
                    "签署时的比较与当前比较并列保留；当前证据变化不自动撤销已签排期，"
                    "但需由保护人员按规程复核。"
                ),
            }
        )
    return out


def reopen_analysis(
    previous: dict[str, Any],
    current: dict[str, Any],
    schedules: list[Schedule],
    *,
    trigger: str = "",
) -> dict[str, Any]:
    """新报告改变适用范围时，列出应重开的建议与被保护的排期。"""
    prev_conditions = previous.get("recommendation", {}).get("conditions", {})
    curr_conditions = current.get("recommendation", {}).get("conditions", {})

    changed: list[dict[str, Any]] = []
    for name in sorted(set(prev_conditions) | set(curr_conditions)):
        before = prev_conditions.get(name, {}).get("decision", "absent")
        after = curr_conditions.get(name, {}).get("decision", "absent")
        if before != after:
            changed.append(
                {
                    "condition": name,
                    "previous_decision": before,
                    "current_decision": after,
                    "action": "重开该项建议；按当前比较重新评估",
                }
            )

    changed_conditions = {c["condition"] for c in changed}

    def _material_of(condition: str | None) -> str | None:
        if not condition:
            return None
        return condition.split("/", 1)[0]

    affected_schedules = []
    unaffected_schedules = []
    for schedule in schedules:
        if not schedule.protected:
            continue
        condition = schedule.comparison_snapshot.get("condition")
        entry = {
            "schedule_id": schedule.schedule_id,
            "wall": schedule.wall,
            "arm": schedule.arm,
            "condition": condition,
            "status": schedule.status,
            "basis_at_signing": schedule.comparison_snapshot,
        }
        if _material_of(condition) in changed_conditions:
            entry["treatment"] = "已签排期仍受保护；不因新报告自动撤销，由保护人员复核"
            affected_schedules.append(entry)
        else:
            entry["treatment"] = "不在变化范围内，维持原排期"
            unaffected_schedules.append(entry)

    return {
        "trigger": trigger,
        "changed_conditions": changed,
        "protected_but_affected_schedules": affected_schedules,
        "unaffected_schedules": unaffected_schedules,
        "rule": "新报告只重开建议，不自动挪动已签排期；两者并存供保护人员回查。",
    }

"""按降雨事件聚类的区间估计与置换检验。

关键约束：
  * 同一场连续降雨是共同事件，不能把事件内多个墙段/多个时段当独立重复；
  * 同一排水组共享外溢，组内读数也不独立；
  * 因此先在同一事件内按排水组汇总，再对排水组等权平均，最后以事件为簇
    做重抽样/置换。事件数就是有效样本量，事件少时区间自然很宽。

工具不把 p 值当唯一结论；区间宽度与"可比事件数"本身就是证据强度的一部分。
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .data import Dataset  # noqa: F401  (类型标注用，避免循环导入)

Row = tuple[str, str, str, bool]  # (event, drain_group, arm, success)


@dataclass(frozen=True)
class EventContrast:
    event: str
    treated_rate: float
    comparison_rate: float
    diff: float
    treated_cells: int
    comparison_cells: int
    groups: int


def _event_group_rates(
    rows: Iterable[Row],
    treated: str,
    comparison: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, set[str]]]:
    """(event, group, arm) -> 组内成功率；同时记录墙段数与事件内排水组集合。"""
    buckets: dict[tuple[str, str, str], list[bool]] = {}
    for event, group, arm, success in rows:
        if arm not in (treated, comparison):
            continue
        buckets.setdefault((event, group, arm), []).append(success)

    group_rates: dict[str, dict[str, float]] = {}
    cell_counts: dict[str, dict[str, int]] = {}
    event_groups: dict[str, set[str]] = {}
    for (event, group, arm), values in buckets.items():
        group_rates.setdefault(event, {})[(group, arm)] = sum(values) / len(values)
        cell_counts.setdefault(event, {})[arm] = cell_counts.setdefault(event, {}).get(arm, 0) + len(values)
        event_groups.setdefault(event, set()).add(group)
    return group_rates, cell_counts, event_groups


def event_contrasts(
    rows: Iterable[Row],
    treated: str,
    comparison: str,
) -> tuple[list[EventContrast], list[str]]:
    """仅在同一降雨事件、同一排水组内做两臂对比。

    返回 (可用的组内对比, 只能跨组对比因而被排除的事件)。
    跨组对比会把排水组的外溢/地形差异算成处理效应，因此不进入估计。
    同一事件内的多个排水组先各自求差，再对排水组等权平均。
    """
    group_rates, cell_counts, event_groups = _event_group_rates(rows, treated, comparison)
    out: list[EventContrast] = []
    cross_group_only: list[str] = []
    for event in sorted(group_rates):
        rates = group_rates[event]
        groups = sorted({g for (g, _arm) in rates})
        group_diffs: list[float] = []
        for group in groups:
            tr = rates.get((group, treated))
            cr = rates.get((group, comparison))
            if tr is not None and cr is not None:
                group_diffs.append(tr - cr)
        if not group_diffs:
            # 两臂都有，但从不在同一排水组内出现：不能当组内比较。
            if any(arm == treated for (_g, arm) in rates) and any(
                arm == comparison for (_g, arm) in rates
            ):
                cross_group_only.append(event)
            continue
        out.append(
            EventContrast(
                event=event,
                treated_rate=sum(
                    rates[(g, treated)] for g in groups if (g, treated) in rates
                )
                / max(1, len([g for g in groups if (g, treated) in rates])),
                comparison_rate=sum(
                    rates[(g, comparison)] for g in groups if (g, comparison) in rates
                )
                / max(1, len([g for g in groups if (g, comparison) in rates])),
                diff=sum(group_diffs) / len(group_diffs),
                treated_cells=cell_counts[event].get(treated, 0),
                comparison_cells=cell_counts[event].get(comparison, 0),
                groups=len(groups),
            )
        )
    return out, cross_group_only


def _bootstrap_mean(values: Sequence[float], iterations: int, seed: int) -> tuple[float, float, float]:
    point = sum(values) / len(values)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low = samples[max(0, int(0.025 * len(samples)) - 1)]
    high = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return point, max(-1.0, low), min(1.0, high)


def _permutation_p(values: Sequence[float], max_exact: int, seed: int) -> float:
    observed = abs(sum(values))
    total = 0
    extreme = 0
    if 2 ** len(values) <= max_exact:
        for signs in itertools.product((1, -1), repeat=len(values)):
            if abs(sum(s * v for s, v in zip(signs, values))) >= observed - 1e-12:
                extreme += 1
            total += 1
    else:
        rng = random.Random(seed)
        total = max_exact
        for _ in range(total):
            if abs(sum((1 if rng.random() < 0.5 else -1) * v for v in values)) >= observed - 1e-12:
                extreme += 1
    return extreme / total


def _resolution_floor(values: Sequence[float]) -> float:
    """事件级分辨率下限：每个事件贡献一个 [-1, 1] 的率差，n 个事件无法分辨
    比 1/n 更细的差异。

    当各事件差异完全一致时，bootstrap 会塌缩成零宽度；该下限保证区间
    至少反映"只有这么少事件"这一事实，避免把少数一致事件当成精确估计。
    这是分辨率说明，不是把不确定性人为放大。
    """
    n = max(1, len(values))
    return 1.0 / n


def event_risk_difference(
    rows: Iterable[Row],
    treated: str,
    comparison: str,
    *,
    iterations: int = 4000,
    max_exact: int = 4096,
    seed: int = 20260503,
) -> dict:
    """事件级风险差。返回可序列化字典，状态区分单臂/不可比/可估计。

    * single_arm：只有一个臂出现，不能比较；
    * not_comparable：两臂都有，但没有至少两个事件同时含两臂；
    * estimated：至少两个可比事件。
    """
    rows = list(rows)
    all_arms = {arm for _, _, arm, _ in rows if arm in (treated, comparison)}
    missing = [arm for arm in (treated, comparison) if arm not in all_arms]
    contrasts, cross_group_only = event_contrasts(rows, treated, comparison)
    events_all = {event for event, _, arm, _ in rows if arm in (treated, comparison)}
    groups_all = {group for _, group, arm, _ in rows if arm in (treated, comparison)}

    if not rows or not all_arms:
        return {
            "status": "no_data",
            "clusters": 0,
            "comparable_clusters": 0,
            "events": 0,
            "drain_groups": 0,
            "treated": treated,
            "comparison": comparison,
        }
    if missing:
        return {
            "status": "single_arm",
            "missing_arm": missing,
            "clusters": 0,
            "comparable_clusters": 0,
            "events": len(events_all),
            "drain_groups": len(groups_all),
            "cross_group_only_events": cross_group_only,
            "treated": treated,
            "comparison": comparison,
        }
    if len(contrasts) < 2:
        why = "没有至少两个同一排水组内同时含两臂的降雨事件；同场雨或跨排水组都不能当独立重复。"
        if cross_group_only:
            why += f" 以下事件两臂仅分处不同排水组，已排除：{', '.join(cross_group_only)}。"
        return {
            "status": "not_comparable",
            "clusters": len(contrasts),
            "comparable_clusters": len(contrasts),
            "events": len(events_all),
            "drain_groups": len(groups_all),
            "cross_group_only_events": cross_group_only,
            "treated": treated,
            "comparison": comparison,
            "why": why,
        }

    values = [c.diff for c in contrasts]
    point, boot_low, boot_high = _bootstrap_mean(values, iterations, seed)
    floor = _resolution_floor(values)
    # 取 bootstrap 与事件分辨率下限中更宽的一侧；两者都是保守方向。
    low = min(boot_low, point - floor)
    high = max(boot_high, point + floor)
    low = max(-1.0, low)
    high = min(1.0, high)
    p = _permutation_p(values, max_exact, seed)
    return {
        "status": "estimated",
        "point": round(point, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "interval_method": "event-cluster bootstrap，下限叠加事件分辨率 1/n",
        "resolution": round(floor, 6),
        "clusters": len(contrasts),
        "comparable_clusters": len(contrasts),
        "events": len(events_all),
        "drain_groups": len(groups_all),
        "cross_group_only_events": cross_group_only,
        "p": round(p, 6),
        "treated": treated,
        "comparison": comparison,
        "per_event": [
            {
                "event": c.event,
                "diff": round(c.diff, 6),
                "treated_rate": round(c.treated_rate, 6),
                "comparison_rate": round(c.comparison_rate, 6),
                "treated_cells": c.treated_cells,
                "comparison_cells": c.comparison_cells,
                "groups": c.groups,
            }
            for c in contrasts
        ],
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项 Wilson 区间，仅用于事件数很少时的描述性辅助。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

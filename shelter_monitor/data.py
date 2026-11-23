"""输入模型与读取。

只读取业务记录本身；不把不同表中的同一标识当成独立观测。
空值保持为 None（来源未给值），与显式的 0 / "unknown" 区分开。
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class InputError(ValueError):
    """输入结构或取值违反规程，需要业务人员修正。"""


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def _parse_date(value: str | None, where: str) -> dt.date | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{where}: 无法解析日期 {value!r}") from exc


def _parse_float(value: str | None, where: str) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise InputError(f"{where}: 无法解析数值 {value!r}") from exc


def _parse_bool(value: str | None, where: str) -> bool | None:
    value = _clean(value)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"yes", "true", "1", "y"}:
        return True
    if lowered in {"no", "false", "0", "n"}:
        return False
    raise InputError(f"{where}: 无法解析 yes/no 值 {value!r}")


@dataclass(frozen=True)
class Observation:
    row: int
    wall: str
    drain_group: str
    assigned: str  # "shelter" / "control" — 抽签登记的意向组
    sensor: str
    calibration: str  # "valid" / "invalid" / None（未声明）
    model: str | None
    observed_at: dt.date | None
    moisture: float | None
    warning: float | None
    rain_event: str | None
    damage_review: str | None

    @property
    def flagged_usable(self) -> bool:
        """主表自身声明可用的有效受雨观察。最终是否可用还要看校准覆盖。"""
        return (
            self.calibration == "valid"
            and self.moisture is not None
            and self.warning is not None
            and self.observed_at is not None
            and self.rain_event is not None
        )

    @property
    def exceeds_warning(self) -> bool | None:
        if not self.flagged_usable:
            return None
        return self.moisture > self.warning  # type: ignore[operator]


@dataclass(frozen=True)
class Wall:
    wall: str
    material: str | None
    side: str | None
    emergency: bool | None
    neighbours: tuple[str, ...] = ()
    shelter_start: dt.date | None = None
    shelter_end: dt.date | None = None
    sheltered: bool | None = None
    """是否属于遮护墙段。True/False 为明确记录，None 为来源未给值。

    False 表示从不遮护——用于区分"从未搭棚"与"搭棚日期未知"。
    """

    neighbours_known: bool = False
    """是否明确登记过相邻关系。True 且 neighbours 为空 = 确实无相邻墙。"""


@dataclass(frozen=True)
class PolicyRecord:
    policy: str
    item: str
    rule: str


@dataclass
class Dataset:
    observations: list[Observation] = field(default_factory=list)
    walls: dict[str, Wall] = field(default_factory=dict)
    policy: list[PolicyRecord] = field(default_factory=list)
    calibration_events: list["CalibrationEvent"] = field(default_factory=list)
    damage_review_rows: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    schedule_rows: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    input_paths: list[str] = field(default_factory=list)

    def wall(self, name: str) -> Wall:
        return self.walls.get(name, Wall(name, None, None, None))

    def shelter_start(self, name: str) -> dt.date | None:
        return self.wall(name).shelter_start


@dataclass(frozen=True)
class CalibrationEvent:
    """换机校准：某设备型号在某段日期内有效。"""

    sensor: str
    model: str | None
    valid_from: dt.date | None
    valid_to: dt.date | None
    calibration: str
    row: int

    def covers(self, observed_at: dt.date | None, sensor: str) -> bool:
        if observed_at is None or sensor != self.sensor:
            return False
        if self.valid_from is not None and observed_at < self.valid_from:
            return False
        if self.valid_to is not None and observed_at > self.valid_to:
            return False
        return True


_KNOWN_ASSIGNMENTS = {"shelter", "control"}


def read_primary(path: Path) -> list[Observation]:
    out: list[Observation] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "wall",
            "drain_group",
            "assigned",
            "sensor",
            "calibration",
            "observed_at",
            "moisture",
            "warning",
            "rain",
            "damage_review",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise InputError(f"{path}: 缺少字段 {sorted(missing)}")
        for index, raw in enumerate(reader, start=2):
            assigned = _clean(raw.get("assigned"))
            if assigned not in _KNOWN_ASSIGNMENTS:
                raise InputError(f"{path}:{index}: assigned 必须是 shelter/control，得到 {assigned!r}")
            calibration = _clean(raw.get("calibration"))
            if calibration is not None and calibration not in {"valid", "invalid"}:
                raise InputError(f"{path}:{index}: calibration 只能是 valid/invalid 或留空")
            out.append(
                Observation(
                    row=index,
                    wall=_clean(raw.get("wall")) or "",
                    drain_group=_clean(raw.get("drain_group")) or "",
                    assigned=assigned,
                    sensor=_clean(raw.get("sensor")) or "",
                    calibration=calibration,
                    model=_clean(raw.get("model")),
                    observed_at=_parse_date(raw.get("observed_at"), f"{path}:{index}"),
                    moisture=_parse_float(raw.get("moisture"), f"{path}:{index}"),
                    warning=_parse_float(raw.get("warning"), f"{path}:{index}"),
                    rain_event=_clean(raw.get("rain")),
                    damage_review=_clean(raw.get("damage_review")),
                )
            )
    return out


def read_walls(path: Path) -> dict[str, Wall]:
    """读取墙段表：材料、方位、紧急线、相邻关系、实际遮护起止日期。

    除 wall 外字段均可留空。相邻关系可写成 "W2|W3" 或 "W2;W3"。
    实际施工日期只用于解释遵从，绝不用于重划评估组。
    """
    walls: dict[str, Wall] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        if "wall" not in headers:
            raise InputError(f"{path}: 缺少 wall 字段")
        for index, raw in enumerate(reader, start=2):
            name = _clean(raw.get("wall"))
            if not name:
                raise InputError(f"{path}:{index}: wall 不能为空")
            neighbours_raw = _clean(raw.get("neighbours")) or ""
            neighbours = tuple(
                n for n in (part.strip() for part in neighbours_raw.replace(";", "|").split("|")) if n
            )
            walls[name] = Wall(
                wall=name,
                material=_clean(raw.get("material")),
                side=_clean(raw.get("side")),
                emergency=_parse_bool(raw.get("emergency"), f"{path}:{index}"),
                neighbours=neighbours,
                shelter_start=_parse_date(raw.get("shelter_start"), f"{path}:{index}"),
                shelter_end=_parse_date(raw.get("shelter_end"), f"{path}:{index}"),
                sheltered=_parse_bool(raw.get("sheltered"), f"{path}:{index}"),
                neighbours_known="neighbours" in headers,
            )
    return walls


def read_policy(path: Path) -> list[PolicyRecord]:
    out: list[PolicyRecord] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            policy = _clean(raw.get("policy"))
            item = _clean(raw.get("item"))
            rule = _clean(raw.get("rule"))
            if policy and item:
                out.append(PolicyRecord(policy, item, rule or ""))
    return out


_CAL_FIELDS = {"sensor", "model", "valid_from", "valid_to", "calibration"}


def read_calibrations(path: Path) -> list[CalibrationEvent]:
    events: list[CalibrationEvent] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        missing = _CAL_FIELDS - headers
        if missing:
            raise InputError(f"{path}: 缺少字段 {sorted(missing)}")
        for index, raw in enumerate(reader, start=2):
            calibration = _clean(raw.get("calibration")) or "valid"
            if calibration not in {"valid", "invalid"}:
                raise InputError(f"{path}:{index}: calibration 只能是 valid/invalid")
            events.append(
                CalibrationEvent(
                    sensor=_clean(raw.get("sensor")) or "",
                    model=_clean(raw.get("model")),
                    valid_from=_parse_date(raw.get("valid_from"), f"{path}:{index}"),
                    valid_to=_parse_date(raw.get("valid_to"), f"{path}:{index}"),
                    calibration=calibration,
                    row=index,
                )
            )
    return events


def _infer_adjacency(observations: Iterable[Observation]) -> dict[str, tuple[str, ...]]:
    """没有邻接表时，默认同一排水组内相邻的墙段互相邻接。

    这是保守的兜底：宁可在组内多算一次暴露，也不把邻墙当独立对照。
    """
    groups: dict[str, set[str]] = {}
    for obs in observations:
        groups.setdefault(obs.drain_group, set()).add(obs.wall)
    adjacency: dict[str, set[str]] = {}
    for walls in groups.values():
        for wall in walls:
            adjacency.setdefault(wall, set()).update(walls - {wall})
    return {k: tuple(sorted(v)) for k, v in adjacency.items()}


def read_damage_reviews(path: Path) -> list[dict]:
    """独立病害复查。wall, reviewed_at, outcome, independent（可选）。"""
    out: list[dict] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or ())
        if "wall" not in headers or "outcome" not in headers:
            raise InputError(f"{path}: 至少需要 wall 与 outcome 字段")
        for index, raw in enumerate(reader, start=2):
            outcome = _clean(raw.get("outcome"))
            if outcome is None:
                continue
            independent_raw = _clean(raw.get("independent"))
            independent = True if independent_raw is None else _parse_bool(independent_raw, f"{path}:{index}")
            out.append(
                {
                    "wall": _clean(raw.get("wall")) or "",
                    "reviewed_at": _parse_date(raw.get("reviewed_at"), f"{path}:{index}"),
                    "outcome": outcome.lower(),
                    "independent": independent,
                    "review_id": _clean(raw.get("review_id")),
                    "row": index,
                }
            )
    return out


def load_dataset(fixtures_dir: Path) -> Dataset:
    fixtures_dir = Path(fixtures_dir)
    if not fixtures_dir.is_dir():
        raise InputError(f"输入目录不存在: {fixtures_dir}")
    primary = fixtures_dir / "primary.csv"
    if not primary.is_file():
        raise InputError(f"缺少主表: {primary}")

    dataset = Dataset(sources=["primary.csv"], input_paths=[str(primary)])
    dataset.observations = read_primary(primary)

    walls_path = fixtures_dir / "walls.csv"
    if walls_path.is_file():
        dataset.walls = read_walls(walls_path)
        dataset.sources.append("walls.csv")
        dataset.input_paths.append(str(walls_path))
    else:
        # 兼容 support-1.csv 的墙段静态字段；无邻接列时按排水组兜底。
        support = fixtures_dir / "support-1.csv"
        if support.is_file():
            dataset.walls = read_walls(support)
            dataset.sources.append("support-1.csv")
            dataset.input_paths.append(str(support))
        inferred = _infer_adjacency(dataset.observations)
        for name, wall in list(dataset.walls.items()):
            if not wall.neighbours and name in inferred:
                dataset.walls[name] = Wall(
                    wall=wall.wall,
                    material=wall.material,
                    side=wall.side,
                    emergency=wall.emergency,
                    neighbours=inferred[name],
                    shelter_start=wall.shelter_start,
                    shelter_end=wall.shelter_end,
                    sheltered=wall.sheltered,
                    neighbours_known=False,
                )
        for name in inferred:
            dataset.walls.setdefault(name, Wall(name, None, None, None, inferred[name]))

    policy_path = fixtures_dir / "support-2.csv"
    if policy_path.is_file():
        dataset.policy = read_policy(policy_path)
        dataset.sources.append("support-2.csv")
        dataset.input_paths.append(str(policy_path))

    cal_path = fixtures_dir / "calibrations.csv"
    if cal_path.is_file():
        dataset.calibration_events = read_calibrations(cal_path)
        dataset.sources.append("calibrations.csv")
        dataset.input_paths.append(str(cal_path))

    reviews_path = fixtures_dir / "damage_reviews.csv"
    if reviews_path.is_file():
        dataset.damage_review_rows = read_damage_reviews(reviews_path)
        dataset.sources.append("damage_reviews.csv")
        dataset.input_paths.append(str(reviews_path))

    # 候选墙段与工期：下一轮布点用，不影响本轮评估组。
    candidates_path = fixtures_dir / "candidates.csv"
    if candidates_path.is_file():
        dataset.candidates = read_candidates(candidates_path)
        dataset.sources.append("candidates.csv")
        dataset.input_paths.append(str(candidates_path))

    schedules_path = fixtures_dir / "schedules.csv"
    if schedules_path.is_file():
        dataset.schedule_rows = read_schedules(schedules_path)
        dataset.sources.append("schedules.csv")
        dataset.input_paths.append(str(schedules_path))

    for obs in dataset.observations:
        if obs.wall not in dataset.walls:
            dataset.walls[obs.wall] = Wall(obs.wall, None, None, None)

    # 候选墙段不并入 dataset.walls：它们不属于本轮已建墙段，
    # 否则会被误列进本轮证据缺口。布点模块直接读 candidates 的字段。
    return dataset


def read_schedules(path: Path) -> list[dict]:
    """已签/已确认排期。wall, arm, start, end, status, signed_at, condition, note。

    comparison_snapshot 可写成 JSON 字符串，保留签署当时依赖的比较。
    """
    out: list[dict] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "wall" not in (reader.fieldnames or []):
            raise InputError(f"{path}: 缺少 wall 字段")
        for index, raw in enumerate(reader, start=2):
            wall = _clean(raw.get("wall"))
            if not wall:
                continue
            arm = _clean(raw.get("arm"))
            if arm not in {"shelter", "control"}:
                raise InputError(f"{path}:{index}: arm 必须是 shelter/control")
            status = _clean(raw.get("status")) or "signed"
            if status not in {"signed", "confirmed", "draft"}:
                raise InputError(f"{path}:{index}: status 必须是 signed/confirmed/draft")
            snapshot: dict = {}
            raw_snapshot = _clean(raw.get("comparison_snapshot"))
            if raw_snapshot:
                import json

                try:
                    snapshot = json.loads(raw_snapshot)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path}:{index}: comparison_snapshot 不是合法 JSON") from exc
            condition = _clean(raw.get("condition"))
            if condition and "condition" not in snapshot:
                snapshot["condition"] = condition
            out.append(
                {
                    "schedule_id": _clean(raw.get("schedule_id")) or f"S{index}",
                    "wall": wall,
                    "arm": arm,
                    "start": _parse_date(raw.get("start"), f"{path}:{index}"),
                    "end": _parse_date(raw.get("end"), f"{path}:{index}"),
                    "status": status,
                    "signed_at": _parse_date(raw.get("signed_at"), f"{path}:{index}"),
                    "comparison_snapshot": snapshot,
                    "note": _clean(raw.get("note")),
                    "row": index,
                }
            )
    return out


def read_candidates(path: Path) -> list[dict]:
    """下一轮候选墙段与可用工期。

    字段：wall, material, side, drain_group, available_from, available_to, note。
    drain_group 用于保证遮护与对照落在同一排水组内比较。
    """
    out: list[dict] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "wall" not in (reader.fieldnames or []):
            raise InputError(f"{path}: 缺少 wall 字段")
        for index, raw in enumerate(reader, start=2):
            name = _clean(raw.get("wall"))
            if not name:
                continue
            out.append(
                {
                    "wall": name,
                    "material": _clean(raw.get("material")),
                    "side": _clean(raw.get("side")),
                    "drain_group": _clean(raw.get("drain_group")),
                    "available_from": _parse_date(raw.get("available_from"), f"{path}:{index}"),
                    "available_to": _parse_date(raw.get("available_to"), f"{path}:{index}"),
                    "neighbours": tuple(
                        n
                        for n in (
                            part.strip()
                            for part in (_clean(raw.get("neighbours")) or "")
                            .replace(";", "|")
                            .split("|")
                        )
                        if n
                    ),
                    "emergency": _parse_bool(raw.get("emergency"), f"{path}:{index}"),
                    "note": _clean(raw.get("note")),
                    "row": index,
                }
            )
    return out

"""命令行入口。

    python3 -m shelter_monitor review  --fixtures fixtures [--margin 0.05] [--format md|json]
    python3 -m shelter_monitor plan    --fixtures fixtures [--format md|json]
    python3 -m shelter_monitor audit   --fixtures fixtures [--schedule S1] [--format md|json]

每次收到新观测，重新运行 `review` 即可；`plan` 会基于当前报告重算布点，
`audit` 并列签署时与当前比较。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .allocate import audit_schedules, plan_next_round, reopen_analysis, schedules_from_rows
from .data import InputError, load_dataset
from .review import format_risk, report_to_json, run_review, to_markdown


def _fmt_risk(result: dict | None) -> str:
    return format_risk(result)


def _plan_markdown(plan: dict) -> str:
    lines = ["# 下一轮布点", ""]
    c = plan["constraints"]
    lines.append("## 不可挪动")
    if c["protected_schedules"]:
        for s in c["protected_schedules"]:
            lines.append(f"- 排期 {s['schedule_id']}：`{s['wall']}` → {s['arm']}（{s['start']} 至 {s['end']}）")
    else:
        lines.append("- 无已签/已确认排期")
    if c["emergency_walls"]:
        lines.append(f"- 紧急线墙段：{', '.join(c['emergency_walls'])}（就地干预，不用于补对照）")
    lines.append("")
    lines.append("## 选中的观察组合")
    if not plan["selected_options"]:
        lines.append("- 现有候选与工期内没有可行的同组遮护/对照组合")
    else:
        for option in plan["selected_options"]:
            arms = "；".join(f"`{a['wall']}`→{a['arm']}" for a in option["assignments"])
            lines.append(f"- {arms}；排水组 {option['drain_group']}；窗口 {option['window']['start']} 至 {option['window']['end']}")
            lines.append(f"  - 填补：{', '.join(option['fills'])}")
    lines.append("")
    lines.append("## 被阻止的组合")
    if not plan["alternatives_blocked"]:
        lines.append("- 无")
    else:
        for option in plan["alternatives_blocked"]:
            arms = "；".join(f"`{a['wall']}`→{a['arm']}" for a in option["assignments"])
            reasons = "；".join(cf["detail"] for cf in option["conflicts"])
            lines.append(f"- {arms}：{reasons}")
    lines.append("")
    lines.append("## 单臂延伸（不单独补齐可比条件）")
    if not plan["arm_extension_options"]:
        lines.append("- 无")
    else:
        for option in plan["arm_extension_options"]:
            a = option["assignments"][0]
            lines.append(f"- `{a['wall']}`→{a['arm']}（{option['condition']}）：{option['why']}")
    lines.append("")
    lines.append("## 剩余资源也补不齐的缺口")
    if not plan["unfillable_gaps"]:
        lines.append("- 无")
    else:
        for gap in plan["unfillable_gaps"]:
            lines.append(f"- {gap['gap_id']}：{gap['reason']}；需要：{gap['needs']}")
    lines.append("")
    return "\n".join(lines)


def _audit_markdown(audits: list[dict]) -> str:
    lines = ["# 签排期审计", ""]
    if not audits:
        lines.append("- 没有已签/已确认排期")
        return "\n".join(lines)
    for audit in audits:
        lines.append(f"## {audit['schedule_id']} · `{audit['wall']}` → {audit['arm']}")
        lines.append(f"- 窗口：{audit['window']['start']} 至 {audit['window']['end']}（{audit['status']}）")
        lines.append(f"- 签署时间：{audit['signed_at'] or '未记录'}")
        lines.append(f"- 签署时依赖的比较：{json.dumps(audit['basis_at_signing'], ensure_ascii=False, sort_keys=True)}")
        current = audit["current_comparison"] or {}
        if isinstance(current, dict) and "decision" in current:
            lines.append(
                f"- 当前比较：{_fmt_risk(current.get('direct'))}；邻墙 {_fmt_risk(current.get('neighbour'))}"
            )
        else:
            lines.append(f"- 当前比较：{_fmt_risk(current if isinstance(current, dict) else None)}")
        lines.append(f"- {audit['note']}")
        lines.append("")
    return "\n".join(lines)


def _reopen_markdown(result: dict) -> str:
    lines = ["# 新报告影响范围", ""]
    if not result["changed_conditions"]:
        lines.append("- 条件结论没有变化；无需重开建议。")
    else:
        for change in result["changed_conditions"]:
            lines.append(
                f"- `{change['condition']}`：{change['previous_decision']} → {change['current_decision']}；{change['action']}"
            )
    lines.append("")
    lines.append(f"- 已签排期仍受保护：{len(result['protected_but_affected_schedules'])} 份受变化影响，{len(result['unaffected_schedules'])} 份不受影响")
    if result["protected_but_affected_schedules"]:
        lines.append("")
        lines.append("## 受变化影响但仍受保护的排期")
        for entry in result["protected_but_affected_schedules"]:
            lines.append(
                f"- {entry['schedule_id']} · `{entry['wall']}` → {entry['arm']}"
                f"（条件 {entry['condition']}，{entry['status']}）：{entry['treatment']}"
            )
    if result["unaffected_schedules"]:
        lines.append("")
        lines.append("## 不受影响的排期")
        for entry in result["unaffected_schedules"]:
            lines.append(
                f"- {entry['schedule_id']} · `{entry['wall']}` → {entry['arm']}（条件 {entry['condition']}）"
            )
    lines.append("")
    lines.append(f"- {result['rule']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shelter-monitor", description="雨季遗址遮护的因果复盘与下一轮布点")
    parser.add_argument("command", choices=["review", "plan", "audit", "reopen"])
    parser.add_argument("--fixtures", default="fixtures", help="输入目录（默认 fixtures/）")
    parser.add_argument("--margin", type=float, default=None, help="邻墙风险界限；未给时读规程或使用保守默认")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    parser.add_argument("--schedule", action="append", default=None, help="audit 只查这些 schedule_id")
    parser.add_argument("--previous", default=None, help="reopen 所需的上一次报告 JSON 路径")
    parser.add_argument("--out", default=None, help="另存 JSON 报告路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(Path(args.fixtures))
        report = run_review(dataset, args.margin)
    except InputError as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2

    schedules = schedules_from_rows(dataset.schedule_rows)

    if args.command == "review":
        payload = report
        text = report_to_json(report) if args.format == "json" else to_markdown(report)
    elif args.command == "plan":
        payload = plan_next_round(dataset, report, schedules)
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) if args.format == "json" else _plan_markdown(payload)
    elif args.command == "audit":
        payload = {
            "generated_at": report["generated_at"],
            "input_hash": report["input_hash"],
            "audits": audit_schedules(schedules, report, args.schedule),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) if args.format == "json" else _audit_markdown(payload["audits"])
    elif args.command == "reopen":
        if not args.previous:
            parser.error("reopen 需要 --previous <旧报告.json>")
        try:
            with open(args.previous, encoding="utf-8") as stream:
                previous = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"无法读取旧报告：{exc}", file=sys.stderr)
            return 2
        payload = reopen_analysis(previous, report, schedules, trigger=args.previous)
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) if args.format == "json" else _reopen_markdown(payload)
    else:  # pragma: no cover
        parser.error("未知命令")

    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0

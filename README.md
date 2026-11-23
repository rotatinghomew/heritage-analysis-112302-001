# 雨季遗址遮护的因果复盘与下一轮布点

本地数据与术语材料供业务人员整理分析资料使用，所有记录均为合成样例。

工具按原抽签组复盘遮护政策，并把"下一轮布点"做成可重复运行的分析；
每次收到新观测重跑即可，不需要手工重算比较。

## 目录

- `fixtures/` 合成输入（最小示例）
- `examples/round2/` 完整示例：独立复查、校准、候选墙段、已签排期
- `docs/业务规程.md` 业务规则
- `docs/数据说明.md` 字段与实体关联
- `docs/工具说明.md` 使用方法、结论等级与证据缺口口径
- `shelter_monitor/` 分析工具
- `tests/` 输入结构与业务约束测试

## 环境和测试

Python 3.11 标准库，无需第三方依赖。

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q .
```

## 使用

```bash
python3 -m shelter_monitor review --fixtures fixtures          # 季中复盘
python3 -m shelter_monitor plan   --fixtures fixtures          # 下一轮布点
python3 -m shelter_monitor audit  --fixtures fixtures          # 签排期审计
python3 -m shelter_monitor reopen --fixtures fixtures --previous old.json
```

加 `--format json --out report.json` 可存档报告，供下一轮 `reopen` 回查。
详见 `docs/工具说明.md`。

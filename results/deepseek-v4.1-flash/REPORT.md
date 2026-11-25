# DeepSeek-V4.1-Flash（思考关闭）· OBM-VLM-Task2 结果汇总

## 图表设计（对应 README 协议）

1. **题型总览条图**：只报告 target / critical / near / unrelated 四类指标，不用总准确率。
2. **三因素分面图**（02–04）：每次固定另外两个因素，单独考察 context_newlines、overlap、repetition。
3. **三因素平均图**（07–09）：每张图只变化一个因素，另外两个因素在每个 seed 内先取平均。
4. **全条件热图**：展示全因子网格，便于对照交互。
5. **Target–Critical 散点**：区分「一般遗忘」（target 降）与「绑定错误」（target 稳、critical 升）。

误差条：分面图在每个完整条件下先算每个 test seed 的题型均值，再对 18 个 seed 求均值与近似 95% CI。平均图先在每个 seed 内对非对比因素取平均，再跨 seed 计算。热图与散点仍按完整条件、跨 seed 计算。

## 总体（test, n=7776）

| 题型 | 指标 | n |
|------|------|---|
| target | 1.0000 | 2916 |
| critical_recombination | 0.0031 | 2909 |
| near_lure | 0.0000 | 970 |
| unrelated_lure | 0.0000 | 972 |

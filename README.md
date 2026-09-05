# HMM-GRU 选股因子挖掘

基于 **隐马尔可夫模型（HMM）因子** 与 **GRU 时序选股模型** 的两阶段量化因子挖掘项目：
先用 HMM 从价量序列中提炼"上涨概率"风格的结构化因子，再将其作为时序特征输入 GRU，
输出端到端的选股因子 `gru_score`，并在 RiceQuant（rqdatac）数据上完成 IC 检验与
分组回测（含交易成本、基准对冲）。

## 方法概览

```
原始价量数据 (7因子)                HMM 因子                    GRU 因子
close/open/high/low ──┐      ┌─ UP 模型  log P(O|λ_up) ──┐
turnover/cap/ret1d  ──┼─► HMM┤                          ├─► GRU(40日窗口×9特征) ─► gru_score ─► 调仓回测
                      │      └─ DOWN 模型 log P(O|λ_dn) ─┘
        剔除 ST/停牌, 截面 winsorize + z-score      时序/截面标准化, 日截面排序损失
```

### 阶段一：HMM 因子（`src/hmm_factor.py`）

复现广发证券《再探西蒙斯投资之道：基于隐马尔科夫模型的选股策略研究》：

- **观测特征**：`close, open, high, low, turnover, returnPast1d, cap` 7 维，预处理为
  前向填充 → 截面 winsorize（均值±3σ）→ 截面 z-score；
- **模型设定**：隐状态数 N=3、高斯混合分模型数 M=2、观测序列长度 Q=10、调仓周期 L=5
  （hmmlearn 的 GMMHMM；`hmmlearn>=0.3` 已移除 GMMHMM，脚本内置了等价的
  `BaseHMM + GaussianMixture` 实现自动降级）；
- **完全滚动训练**：每个调仓日 T 重新训练 UP / DOWN 两个 HMM——往前取 10 个周期
  （约 5000 个样本），按区间收益符号划分上涨/下跌样本，各自训练；
- **因子定义**：股票在 (T-9)~T 的观测序列在 UP 模型下的对数观测概率
  `hmm_factor = log P(O|λ_up)`（同时输出 `logp_down` 供 GRU 使用）。

### 阶段二：GRU 选股模型（`src/gru_train.py`）

依据申万宏源《GRU 选股模型的时序分域探索》的统一模型设定：

- **输入**：过去 40 个交易日的 9 维特征序列
  （7 个原始因子 + `hmm_factor, logp_up, logp_down`），
  先做滚动窗口（40 日）时序 z-score，再做每日截面 z-score；
- **结构**：GRU(hidden=16, 1 层, dropout=0.2) + Linear；
- **损失**：按日截面排序损失（负 Pearson 相关系数，直接最大化截面 IC），
  按日组织 batch；
- **训练/验证/测试**：2019–2025 训练，训练窗口尾部 20% 交易日验证（早停指标 =
  验证集 RankIC），2026 年为严格样本外测试；
- **输出**：`results/gru_factor.csv`（全历史因子，含各股票期末无标签的 5 个交易日，
  供回测/平台上传）与 `results/gru_predictions.csv`（2026 测试集预测）。

### 阶段三：回测（`notebooks/ricequant_Backtesting.ipynb`）

基于 rqdatac 后复权收盘价：周频调仓（持有 5 日、T+1 建仓），
每期按因子分 5 组，计入佣金（万 2 双边）、印花税（万 5 卖出）、滑点（千 1 双边），
并与中证 500 基准做超额对比。

## 回测结果

GRU 因子，回测区间 2021-05 ~ 2026-08（周频调仓，约 774 只股票，剔除 ST/停牌）：

| 指标 | Pearson IC | Rank IC |
|---|---|---|
| 均值 | 0.0494 | **0.0649** |
| ICIR | 0.3747 | **0.5728** |
| IC > 0 占比 | 64.98% | **69.26%** |

| 组合（扣费后） | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 胜率 |
|---|---|---|---|---|---|
| 多头 G5（top 20%） | 17.08% | 21.92% | 0.78 | -24.55% | 56.03% |
| 多空 L-S（G5-G1） | **30.70%** | 13.66% | **2.25** | -17.80% | 62.65% |

- 多头 G5 相对中证 500（扣费）：年化超额 **12.10%**，信息比率 1.21，超额胜率 62.65%；
- 分组单调性好：扣费后 G1 ~ G5 年化分别为 -9.69% / 1.40% / 6.54% / 7.08% / **18.07%**。

> ⚠️ 注意：全历史因子 `gru_factor.csv` 包含 2019–2025 训练期（样本内），
> 上表区间大部分为样本内表现，收益会偏乐观；**严格样本外请看 2026 年测试段**
> （`results/gru_predictions.csv`，脚本训练日志中也会打印 2026 年 RankIC/ICIR）。

## 快速开始

```bash
git clone <your-repo-url>
cd hmm-gru-factor
pip install -r requirements.txt
```

**1. 准备数据**：将 `raw_factors.pkl`、`stock_pool.pkl` 放入 `data/`
（格式见 [data/README.md](data/README.md)，行情与股票池数据需通过 rqdatac 等授权数据源准备）。

**2. 计算 HMM 因子并合并原始特征**：

```bash
python src/hmm_factor.py --data ./data --out ./data/hmm_factor.pkl
python src/merge_hmm_features.py --factor ./data/hmm_factor.pkl --raw ./data/raw_factors.pkl --out ./data/hmm_factor.csv
```

**3. 训练 GRU 并生成因子**：

```bash
python src/gru_train.py        # 读取 data/hmm_factor.csv，输出至 results/ 与 models/
```

**4. 回测**：打开 `notebooks/ricequant_Backtesting.ipynb`，修改变量区
`FACTOR_FILE` / `FACTOR_COL`（如 `gru_factor.csv` / `gru_score`），
需要已配置许可的 rqdatac 环境，逐 cell 运行即可复现 IC 报告、分组净值与基准对比图。

## 项目结构

```
hmm-gru-factor/
├── src/
│   ├── hmm_factor.py          # 阶段一：滚动训练 HMM，输出 HMM 因子
│   ├── gru_train.py           # 阶段二：GRU 选股模型训练与因子生成
│   └── merge_hmm_features.py  # 工具：合并 HMM 因子与原始 7 因子
├── notebooks/
│   └── ricequant_Backtesting.ipynb  # 阶段三：IC / 分组 / 基准对冲回测
├── models/
│   └── gru_model.pt           # 验证集最优 GRU 权重（hidden=16，约 7.5KB）
├── data/                      # 数据目录（大文件不入库，见 data/README.md）
├── results/                   # 训练输出（gru_factor.csv / gru_predictions.csv）
└── docs/
    ├── plan.md                # 复现方案与研报方法要点
    └── REFERENCES.md          # 参考文献
```

## 环境依赖

见 [requirements.txt](requirements.txt)。核心：`pandas / numpy / scipy / scikit-learn /
torch / hmmlearn / matplotlib`，回测需 `rqdatac`（RiceQuant 数据接口，需许可）。

> `hmmlearn` 请安装 `0.3` 以下版本以使用原生 `GMMHMM`；若环境为新版，
> 本项目的 `hmm_factor.py` 会自动切换为内置的等价实现，无需降级。

## 数据来源与声明

- 行情与股票池数据来自 **RiceQuant（rqdatac）**。
- 数据文件（`raw_factors.pkl`、`stock_pool.pkl` 等）因体积与许可限制未随仓库发布，
  请通过授权渠道获取后按 [data/README.md](data/README.md) 准备。
- 本仓库仅用于学习与研究交流，不构成任何投资建议；回测结果为历史数据拟合结果，
  样本内表现不代表未来收益。

# 复现广发证券《再探西蒙斯投资之道：基于隐马尔科夫模型的选股策略研究》

## 目标
交付两个 Python 脚本：
1. `fetch_data_rqdatac.py` —— 从 RiceQuant (rqdatac) 下载原始数据并构造研报所需的 7 个原始股票因子
2. `hmm_factor.py` —— 按研报方法滚动训练 HMM（UP/DOWN 模型），输出每只股票每个调仓日的 HMM 因子值

## 研报方法要点（已从 PDF 提取）
- 股票池：中证500成份股（剔除ST、停牌）
- 原始因子（7个）：close, open, high, low, turnover(换手率), returnPast1d(1日涨跌幅), cap(流通市值)
- 预处理：缺失值用上一期填充 → 截面 winsorize（均值±3倍标准差）→ 截面 z-score 标准化
- 调仓周期 L=5 个交易日；观测序列长度 Q=10；隐状态数 N=3；高斯混合分模型数 M=2
- 训练方式：完全滚动。在调仓日 T：
  - 训练集：往前 10 个周期，第 k 期(k=0..9)基准日 T-5k，
    标签 = close(T-5k)/close(T-5k-5)-1 的符号；观测序列 = (T-5k-14)~(T-5k-5) 共10天
  - 上涨样本训练 UP 模型，下跌样本训练 DOWN 模型（约 5000 个样本）
- 预测：在 T 日取 (T-9)~T 的10日观测序列，HMM因子 = 该序列在 UP 模型下的对数观测概率 log P(O|λ_up)
- HMM 库：hmmlearn 的 GMMHMM（高斯混合发射概率），需 hmmlearn<0.3（0.3 起移除了 GMMHMM），缺省时降级为 GaussianHMM

## 阶段
- Stage 1 — 写 `fetch_data_rqdatac.py`：rqdatac 拉取中证500历史成份、行情(OHLC)、换手率、流通股本→流通市值、ST/停牌状态，计算 returnPast1d，落地 parquet
- Stage 2 — 写 `hmm_factor.py`：预处理 + 滚动训练 + 因子计算，输出因子表 parquet
- Stage 3 — 本地验证：用合成数据跑通 Stage 2 核心逻辑（语法、HMM训练、因子输出格式）；Stage 1 因无 rqdatac 许可无法实测，做语法检查

# -*- coding: utf-8 -*-
"""
hmm_factor.py
=============
复现广发证券《再探西蒙斯投资之道：基于隐马尔科夫模型的选股策略研究》(2018-09-05)
—— 第 2 步：基于 HMM 模型计算每只股票在每个调仓日的 HMM 因子值。

研报方法（第 3 章）：
  * 观测向量（每天 6~7 维价量特征，见 FEATURES）
  * 预处理：缺失值用上一期填充 -> 截面 winsorize(均值±3σ) -> 截面 z-score 标准化
  * 超参数：隐状态数 N=3，观测序列长度 Q=10，高斯混合分模型数 M=2，
            调仓/预测周期 L=5，训练样本取往前 10 个周期（约 5000 个样本）
  * 完全滚动训练：每个调仓日 T 重新训练 UP / DOWN 两个 HMM
      - 第 k 期(k=0..9)基准日 T-5k：标签 = close(T-5k)/close(T-5k-5)-1 的符号
        观测序列 = (T-5k-14)~(T-5k-5) 共 Q=10 天
      - 上涨样本训练 UP 模型，下跌样本训练 DOWN 模型
  * HMM 因子 = 股票 (T-9)~T 的观测序列在 UP 模型下的对数观测概率 log P(O|λ_up)

依赖：hmmlearn。注意 hmmlearn>=0.3 移除了 GMMHMM，推荐 pip install "hmmlearn<0.3"；
     若环境只有新版 hmmlearn，本脚本自动降级为 GaussianHMM（单高斯发射，M=1）。

用法：
    python hmm_factor.py --data ./hmm_data --out ./hmm_data/hmm_factor.pkl
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- 模型超参数（研报 3.3 节网格搜索结果）
L = 5            # 调仓/预测周期（交易日）
Q = 10           # 观测序列长度
N_STATES = 3     # 隐状态数
N_MIX = 2        # 高斯混合分模型数
N_PERIODS = 10   # 训练样本往前取的周期数
N_ITER = 50      # Baum-Welch 最大迭代次数

# 观测特征（研报公式 3-2 列出的 7 个原始因子；研报正文写"维度为6"，实际列出7个，默认全用）
FEATURES = ["close", "open", "high", "low", "turnover", "returnPast1d", "cap"]

# 训练所需的最长历史窗口：k=9 期 -> 起点 T-5*9-14 = T-59
MAX_LOOKBACK = L * (N_PERIODS - 1) + (Q + L - 1)   # = 59

try:
    # 旧版 hmmlearn(<0.3) 自带 GMMHMM
    from hmmlearn.hmm import GMMHMM as _GMMHMM
    _HAS_GMMHMM = True
except ImportError:
    _HAS_GMMHMM = False


if not _HAS_GMMHMM:
    # hmmlearn>=0.3 移除了 GMMHMM，这里基于 BaseHMM + sklearn GaussianMixture
    # 自行实现"高斯混合发射概率"的 HMM，严格对应研报 M=2 的设定。
    from hmmlearn.base import BaseHMM
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    class _GMMHMM(BaseHMM):
        """GMM 发射概率的连续 HMM（适配 hmmlearn>=0.3）。
        每个隐状态对应一个 n_mix 分量的 GaussianMixture，EM 训练。"""

        def __init__(self, n_components=3, n_mix=2, covariance_type="diag",
                     n_iter=50, tol=1e-3, random_state=None, reg_covar=1e-3):
            super().__init__(n_components=n_components, n_iter=n_iter, tol=tol,
                             random_state=random_state, init_params="st", params="st")
            self.n_mix = n_mix
            self.covariance_type = covariance_type
            self.reg_covar = reg_covar

        def _init(self, X, lengths):
            super()._init(X, lengths)
            # KMeans 将全部观测帧粗分为 n_components 类，作为各状态 GMM 的初始值
            km = KMeans(n_clusters=self.n_components, n_init=10,
                        random_state=self.random_state).fit(X)
            self.gmms_ = []
            for j in range(self.n_components):
                gmm = GaussianMixture(n_components=self.n_mix,
                                      covariance_type=self.covariance_type,
                                      reg_covar=self.reg_covar,
                                      random_state=self.random_state)
                Xj = X[km.labels_ == j]
                if len(Xj) < self.n_mix:
                    Xj = X
                gmm.fit(Xj)
                self.gmms_.append(gmm)

        def _compute_log_likelihood(self, X):
            # 每帧在"各状态 GMM"下的对数密度，shape=(n_frames, n_components)
            return np.column_stack([g.score_samples(X) for g in self.gmms_])

        def _m_step(self, X, stats):
            post = stats["post"]          # 每帧属于各隐状态的后验概率
            for j in range(self.n_components):
                w = post[:, j]
                if w.sum() < 10 * np.finfo(float).eps:
                    continue
                gmm = GaussianMixture(n_components=self.n_mix,
                                      covariance_type=self.covariance_type,
                                      reg_covar=self.reg_covar,
                                      random_state=self.random_state)
                gmm.fit(X, sample_weight=w)
                self.gmms_[j] = gmm

        def _generate_sample_from_state(self, state, random_state):
            x, _ = self.gmms_[state].sample(1, random_state=random_state)
            return x[0]


def make_hmm(random_state: int = 42):
    """构造研报设定的 HMM：N=3 隐状态，高斯混合 M=2 发射概率。"""
    return _GMMHMM(n_components=N_STATES, n_mix=N_MIX,
                   covariance_type="diag", n_iter=N_ITER,
                   random_state=random_state, tol=1e-3)


# ================================================================ 数据加载
def _read_table(path_no_ext: str) -> pd.DataFrame:
    """优先读 pickle，其次 parquet（兼容旧输出）。"""
    if os.path.exists(path_no_ext + ".pkl"):
        return pd.read_pickle(path_no_ext + ".pkl")
    return pd.read_parquet(path_no_ext + ".parquet")


def load_data(data_dir: str):
    factors = _read_table(os.path.join(data_dir, "raw_factors"))
    pool = _read_table(os.path.join(data_dir, "stock_pool"))
    factors = factors.sort_index()
    pool["date"] = pd.to_datetime(pool["date"])
    return factors, pool


def build_pool_mask(pool: pd.DataFrame, dates, stocks,
                    exclude_st: bool = True, exclude_suspended: bool = True) -> pd.DataFrame:
    """date × stock 的布尔矩阵：当日是否在可用股票池内。"""
    p = pool.copy()
    if exclude_st and "is_st" in p.columns:
        p = p[~p["is_st"]]
    if exclude_suspended and "is_suspended" in p.columns:
        p = p[~p["is_suspended"]]
    mask = pd.DataFrame(False, index=dates, columns=stocks)
    for d, g in p.groupby("date"):
        cols = [s for s in g["order_book_id"] if s in mask.columns]
        mask.loc[d, cols] = True
    return mask


# ================================================================ 预处理（研报 3.2 节）
def preprocess(factors: pd.DataFrame, pool_mask: pd.DataFrame,
               features=FEATURES):
    """
    返回 (std_feat, close)：
      std_feat: dict[feature] -> DataFrame(date × stock) 标准化后的特征
      close:    DataFrame(date × stock) 原始收盘价（用于训练样本标签）
    步骤：1) 个股时序上前向填充缺失值
          2) 截面 winsorize：边界 = 均值 ± 3 倍标准差（仅统计当日池内股票）
          3) 截面 z-score 标准化（仅统计当日池内股票）
    """
    dates = factors.index.get_level_values("date").unique().sort_values()
    stocks = factors.index.get_level_values("order_book_id").unique()

    def _wide(feat):
        return factors[feat].unstack("order_book_id").reindex(
            index=dates, columns=stocks)

    # 1) 前向填充（按股票沿时间轴）
    wide = {f: _wide(f) for f in features + ["close"] if f in factors.columns}
    wide = {f: df.ffill() for f, df in wide.items()}
    close = wide["close"]

    # 2) + 3) 截面 winsorize 与 z-score（按行=按交易日）
    pool_mask = pool_mask.reindex(index=dates, columns=stocks, fill_value=False)
    std_feat = {}
    for f in features:
        df = wide[f]
        m = pool_mask & df.notna()
        n = m.sum(axis=1)
        mean = df.where(m).mean(axis=1)
        std = df.where(m).std(axis=1)
        up, low = mean + 3 * std, mean - 3 * std
        df = df.clip(lower=low, upper=up, axis=0)   # winsorize
        std_safe = std.replace(0, np.nan)
        z = (df.sub(mean, axis=0)).div(std_safe, axis=0)  # z-score
        z = z.where(m)                                     # 池外置 NaN
        # 池内不足 2 只或截面方差为 0 的日期，该日特征不可用
        z = z.where(n >= 2)
        std_feat[f] = z
    return std_feat, close


# ================================================================ 序列构造
def get_sequence(std_feat, pos_end: int, stock: str):
    """
    取以交易日位置 pos_end 结尾、长度 Q 的观测序列 (Q × D)。
    任一特征缺失则返回 None。
    """
    cols = []
    for f in FEATURES:
        s = std_feat[f].iloc[pos_end - Q + 1: pos_end + 1][stock]
        cols.append(s.values)
    seq = np.vstack(cols).T            # (Q, D)
    if np.isnan(seq).any():
        return None
    return seq


def fit_models(train_seqs_up, train_seqs_down):
    """训练 UP / DOWN 两个 HMM。hmmlearn 接受拼接后的样本 + lengths。"""
    def _fit(seqs, seed):
        X = np.concatenate(seqs, axis=0)
        lengths = [len(s) for s in seqs]
        model = make_hmm(random_state=seed)
        model.fit(X, lengths)
        return model

    model_up = _fit(train_seqs_up, seed=42)
    model_down = _fit(train_seqs_down, seed=43)
    return model_up, model_down


# ================================================================ 主流程：滚动计算 HMM 因子
def compute_hmm_factors(std_feat, close, pool_mask,
                        min_train_samples: int = 100):
    """
    完全滚动：每隔 L 个交易日为一个调仓日 T，重新训练 UP/DOWN 模型，
    计算当日池内每只股票的 HMM 因子 = log P(O | λ_up)。

    返回 DataFrame: [date, order_book_id, hmm_factor, logp_up, logp_down]
    """
    dates = close.index
    n_dates = len(dates)
    results = []

    for p in range(MAX_LOOKBACK, n_dates, L):
        T = dates[p]
        # -------- 构造训练集：往前 N_PERIODS 个周期 --------
        seqs_up, seqs_down = [], []
        for k in range(N_PERIODS):
            pos_label = p - L * k          # 标签区间右端 T-5k
            pos_obs = pos_label - L        # 观测序列右端 T-5k-5
            if pos_label - L < 0 or pos_obs - Q + 1 < 0:
                continue
            # 样本标签：close(T-5k) / close(T-5k-5) - 1
            ret = close.iloc[pos_label] / close.iloc[pos_label - L] - 1.0
            # 训练样本取自当期（标签右端日）的股票池
            pool_k = pool_mask.iloc[pos_label]
            for stock in pool_k[pool_k].index:
                r = ret.get(stock, np.nan)
                if not np.isfinite(r):
                    continue
                seq = get_sequence(std_feat, pos_obs, stock)
                if seq is None:
                    continue
                (seqs_up if r > 0 else seqs_down).append(seq)

        if len(seqs_up) < min_train_samples or len(seqs_down) < min_train_samples:
            print(f"[skip] {T.date()} 训练样本不足 "
                  f"(up={len(seqs_up)}, down={len(seqs_down)})")
            continue

        # -------- 训练 --------
        model_up, model_down = fit_models(seqs_up, seqs_down)

        # -------- 预测：当日池内每只股票 --------
        pool_t = pool_mask.iloc[p]
        n_valid = 0
        for stock in pool_t[pool_t].index:
            seq = get_sequence(std_feat, p, stock)
            if seq is None:
                continue
            logp_up = model_up.score(seq)      # HMM 因子：UP 模型下的对数观测概率
            logp_down = model_down.score(seq)
            results.append({
                "date": T,
                "order_book_id": stock,
                "hmm_factor": logp_up,
                "logp_up": logp_up,
                "logp_down": logp_down,
            })
            n_valid += 1
        print(f"[done] {T.date()}  训练样本 up={len(seqs_up)}/down={len(seqs_down)}"
              f"  当日因子覆盖 {n_valid} 只股票")

    factor_df = pd.DataFrame(results)
    if not factor_df.empty:
        factor_df = factor_df.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    return factor_df


def main():
    parser = argparse.ArgumentParser(description="滚动训练 HMM 并计算 HMM 因子")
    parser.add_argument("--data", default="./data", help="原始因子与股票池所在目录（raw_factors.pkl + stock_pool.pkl）")
    parser.add_argument("--out", default="./data/hmm_factor.pkl", help="因子输出路径(.pkl)")
    parser.add_argument("--include-st", action="store_true", help="不剔除 ST 股票")
    parser.add_argument("--include-suspended", action="store_true", help="不剔除停牌股票")
    # parse_known_args：兼容在 Jupyter Notebook 中用 %run 执行（忽略 -f kernel.json 等内核参数）
    args, _ = parser.parse_known_args()

    print("加载数据 ...")
    factors, pool = load_data(args.data)
    dates = factors.index.get_level_values("date").unique().sort_values()
    stocks = factors.index.get_level_values("order_book_id").unique()
    print(f"数据区间 {dates[0].date()} ~ {dates[-1].date()}，共 {len(dates)} 个交易日，{len(stocks)} 只股票")

    pool_mask = build_pool_mask(pool, dates, stocks,
                                exclude_st=not args.include_st,
                                exclude_suspended=not args.include_suspended)

    print("预处理：前向填充 -> 截面 winsorize(±3σ) -> 截面 z-score ...")
    std_feat, close = preprocess(factors, pool_mask)

    print(f"滚动计算 HMM 因子（L={L}, Q={Q}, N={N_STATES}, M={N_MIX}，完全滚动训练）...")
    factor_df = compute_hmm_factors(std_feat, close, pool_mask)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    factor_df.to_pickle(args.out)
    print(f"完成：{args.out}  ({len(factor_df)} 行，{factor_df['date'].nunique()} 个调仓日)")


if __name__ == "__main__":
    main()
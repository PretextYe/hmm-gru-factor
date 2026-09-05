# -*- coding: utf-8 -*-
"""
GRU 选股模型 —— 依据申万宏源《GRU选股模型的时序分域探索》统一模型设定
  输入: 过去 40 个交易日特征序列
  模型: GRU(hidden=16, 1层, dropout=0.2) + Linear
  损失: 按日截面排序损失 (负 Pearson 相关系数, 最大化截面 IC)
  训练: 2019-2025 (数据自2019-10起, 用户要求的2018年无数据)
  验证: 训练窗口尾部 20% 交易日, 早停指标 = 验证集 RankIC
  测试: 2026 年
  输出: gru_predictions.csv (测试集评估) + gru_factor.csv (米筐上传因子:
        date, order_book_id, gru_score, 覆盖全部历史日期 2019-10 起,
        含各股票期末无标签的 5 个交易日; 注意训练期为样本内)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr

# ---------------- 参数 (报告表1) ----------------
LOOKBACK   = 40      # 输入窗口 40 个交易日
HORIZON    = 5       # 预测未来 5 日收益
HIDDEN     = 16      # hidden size
N_LAYERS   = 1       # GRU 层数
DROPOUT    = 0.2     # dropout
LR         = 1e-3
EPOCHS     = 60
PATIENCE   = 8
MIN_STOCKS = 30      # 截面最少股票数
SEED       = 42

FEATURES = ['open', 'high', 'low', 'close', 'turnover',
            'returnPast1d', 'hmm_factor', 'logp_up', 'logp_down']

BASE = Path(__file__).resolve().parent.parent   # 项目根目录
DATA       = BASE / 'data' / 'hmm_factor.csv'          # HMM 因子 + 原始特征（见 data/README.md）
OUT_PRED   = BASE / 'results' / 'gru_predictions.csv'  # 2026 测试集预测（严格样本外评估）
OUT_FACTOR = BASE / 'results' / 'gru_factor.csv'       # 全历史因子（供回测 / 平台上传）
OUT_MODEL  = BASE / 'models' / 'gru_model.pt'          # 验证集最优模型权重
for _p in (DATA.parent, OUT_PRED.parent, OUT_FACTOR.parent, OUT_MODEL.parent):
    _p.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# ---------------- 数据加载与特征工程 ----------------
df = pd.read_csv(DATA, parse_dates=['date'])
df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)
print(f'raw rows={len(df)}, dates {df.date.min().date()} ~ {df.date.max().date()}, '
      f'stocks={df.order_book_id.nunique()}')

# 价格类特征取对数
for c in ['open', 'high', 'low', 'close']:
    df[c] = np.log(df[c].astype(float))

# 标签: 未来 HORIZON 日收盘价收益
df['label'] = df.groupby('order_book_id')['close'].shift(-HORIZON) - df['close']
df['label'] = np.expm1(df['label'])  # log差 -> 收益率

# 时序归一化: 每只股票滚动窗口内 z-score (窗口=LOOKBACK, 需完整窗口)
g = df.groupby('order_book_id', sort=False)
for c in FEATURES:
    x = df[c].astype(float)
    m = x.groupby(df['order_book_id']).transform(
        lambda s: s.rolling(LOOKBACK, min_periods=LOOKBACK).mean())
    s = x.groupby(df['order_book_id']).transform(
        lambda s: s.rolling(LOOKBACK, min_periods=LOOKBACK).std())
    df[c + '_ts'] = (x - m) / s.replace(0, np.nan)

# 截面标准化: 每个交易日横截面 z-score (对时序归一化后的特征)
for c in FEATURES:
    x = df[c + '_ts']
    df[c + '_cs'] = (x - x.groupby(df['date']).transform('mean')) / \
                    x.groupby(df['date']).transform('std').replace(0, np.nan)

FEAT_COLS = [c + '_cs' for c in FEATURES]

# 丢弃滚动窗口不完整(每只股票前39行)导致的 NaN 特征行, 防止 NaN 进入 GRU
df = df.dropna(subset=FEAT_COLS).reset_index(drop=True)
g = df.groupby('order_book_id', sort=False)

# ---------------- 构建序列样本 ----------------
def build_sequences(sub):
    """单只股票: 用滑动窗口构造 (样本数, LOOKBACK, F) 与对应日期/标签"""
    vals = sub[FEAT_COLS].to_numpy(np.float32)
    lab  = sub['label'].to_numpy(np.float32)
    dat  = sub['date'].to_numpy()
    n = len(sub)
    if n < LOOKBACK + 1:
        return None
    # 滑窗视图, 窗口终点从 LOOKBACK-1 开始
    idx = np.arange(LOOKBACK - 1, n)
    starts = idx - (LOOKBACK - 1)
    seqs = np.stack([vals[s:s + LOOKBACK] for s in starts])   # (N, T, F)
    out = pd.DataFrame({
        'date': dat[idx], 'order_book_id': sub['order_book_id'].iloc[0],
        'label': lab[idx]})
    return seqs, out

all_seqs, all_meta = [], []
for oid, sub in g:
    r = build_sequences(sub)
    if r is not None:
        all_seqs.append(r[0])
        all_meta.append(r[1])

X = np.concatenate(all_seqs)                  # (N, T, F)
meta = pd.concat(all_meta).reset_index(drop=True)
print(f'sequences={X.shape}, valid label={meta.label.notna().sum()}')

# 标签按日截面 rank 后 z-score (排序损失目标)
meta["label_rank"] = meta.groupby("date")["label"].rank(pct=True)
meta["label_z"] = (meta.groupby("date")["label_rank"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-9)))

# 无标签样本(每只股票最后5行, shift(-5) 无未来数据)不参与训练/评估(防止 NaN 损失),
# 但保留在 X/meta 中, 用于在样本期末生成因子 —— 回测必须覆盖到最后一个交易日
lab = meta['label'].notna().values
X_lab, meta_lab = X[lab], meta.loc[lab].reset_index(drop=True)

# 划分训练/验证/测试
dates_all = np.sort(meta_lab['date'].unique())
train_dates = dates_all[dates_all < np.datetime64('2026-01-01')]
val_dates   = train_dates[int(len(train_dates) * 0.8):]
tr_dates    = train_dates[:int(len(train_dates) * 0.8)]
te_dates    = dates_all[dates_all >= np.datetime64('2026-01-01')]
print(f'train days={len(tr_dates)}, val days={len(val_dates)}, test days={len(te_dates)}')

def subset(dates):
    m = meta_lab['date'].isin(dates)
    return (torch.tensor(X_lab[m.values], dtype=torch.float32, device=device),
            torch.tensor(meta_lab.loc[m, 'label_z'].to_numpy(np.float32), device=device),
            torch.tensor(meta_lab.loc[m, 'label'].to_numpy(np.float32), device=device),
            meta_lab.loc[m, ['date', 'order_book_id']].reset_index(drop=True))

Xtr, ytr, yraw_tr, met_tr = subset(tr_dates)
Xva, yva, yraw_va, met_va = subset(val_dates)
Xte, yte, yraw_te, met_te = subset(te_dates)

# 按日组织 batch: 日期 -> 样本索引
def day_index(meta_df):
    d = meta_df['date'].values
    uniq, inv = np.unique(d, return_inverse=True)
    return uniq, [np.where(inv == i)[0] for i in range(len(uniq))]

tr_uniq, tr_batches = day_index(met_tr)
va_uniq, va_batches = day_index(met_va)
te_uniq, te_batches = day_index(met_te)

# ---------------- 模型 (报告: hidden=16, 1层, dropout=0.2) ----------------
class GRUModel(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.gru = nn.GRU(n_feat, HIDDEN, num_layers=N_LAYERS,
                          batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)

model = GRUModel(X.shape[-1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def neg_corr_loss(score, target):
    """负 Pearson 相关系数 (最大化截面排序)"""
    s = score - score.mean()
    t = target - target.mean()
    return -(s * t).sum() / (s.norm() * t.norm() + 1e-9)

@torch.no_grad()
def predict(Xs, batches):
    model.eval()
    out = np.empty(len(Xs), dtype=np.float64)
    for idx in batches:
        if len(idx) < MIN_STOCKS:
            out[idx] = np.nan
            continue
        out[idx] = model(Xs[idx]).cpu().numpy()
    return out

def daily_rankic(scores, meta_df, yraw):
    df_e = meta_df.copy()
    df_e['score'] = scores
    df_e['y'] = yraw.cpu().numpy() if torch.is_tensor(yraw) else yraw
    ics = df_e.groupby('date').apply(
        lambda d: spearmanr(d['score'], d['y']).statistic
        if d['score'].notna().sum() > MIN_STOCKS else np.nan,
        include_groups=False)
    ics = ics.dropna()
    return ics

# ---------------- 训练 ----------------
best_ic, best_state, bad_epochs = -np.inf, None, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(len(tr_batches))
    tot = 0.0
    for i in perm:
        idx = tr_batches[i]
        if len(idx) < MIN_STOCKS:
            continue
        score = model(Xtr[idx])
        loss = neg_corr_loss(score, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach())
    # 验证
    va_scores = predict(Xva, va_batches)
    ics = daily_rankic(va_scores, met_va, yraw_va)
    mean_ic = ics.mean()
    icir = mean_ic / (ics.std() + 1e-12)
    print(f'epoch {epoch:3d} | train loss {-tot/len(perm):+.5f} | '
          f'val RankIC {mean_ic:.4f} | ICIR {icir:.3f}', flush=True)
    if mean_ic > best_ic:
        best_ic, bad_epochs = mean_ic, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f'early stop at epoch {epoch} (best val RankIC {best_ic:.4f})')
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), OUT_MODEL)

# ---------------- 测试集评估 ----------------
te_scores = predict(Xte, te_batches)
ics = daily_rankic(te_scores, met_te, yraw_te)
mean_ic, icir = ics.mean(), ics.mean() / (ics.std() + 1e-12)
print(f'\n===== 2026 测试集 (按日 RankIC) =====')
print(f'RankIC 均值: {mean_ic:.4f} | ICIR: {icir:.3f} | 正IC占比: {(ics>0).mean():.1%}')

# 分组收益: top20% 超额 & 多空 (未来5日收益, 日度调仓近似)
df_e = met_te.copy()
df_e['score'] = te_scores
df_e['y'] = yraw_te.cpu().numpy()
def quintile_ret(d):
    d = d.dropna(subset=['score'])
    if len(d) < MIN_STOCKS:
        return pd.Series({'long': np.nan, 'ls': np.nan})
    q = d['score'].quantile([0.2, 0.8])
    long = d.loc[d['score'] >= q[0.8], 'y'].mean() - d['y'].mean()
    ls   = d.loc[d['score'] >= q[0.8], 'y'].mean() - d.loc[d['score'] <= q[0.2], 'y'].mean()
    return pd.Series({'long': long, 'ls': ls})
rets = df_e.groupby('date').apply(quintile_ret)
ann = 244 / HORIZON  # 每年约 48 个 5 日周期
print(f"多头(top20%-全截面) 年化超额: {rets['long'].mean()*ann:.2%}")
print(f"多空(top20%-bottom20%) 年化: {rets['ls'].mean()*ann:.2%}")

# 保存测试集预测
met_te['score'] = te_scores
met_te.to_csv(OUT_PRED, index=False, encoding='utf-8-sig')
print(f'\nsaved: {OUT_PRED}')

# ---------------- 因子文件 (米筐上传格式: date, order_book_id, 因子列) ----------------
# 覆盖全部历史日期; meta 未按标签过滤, 因此包含各股票最后5个无标签交易日的因子值
# torch.from_numpy 与 X 共享内存, 不额外复制整块数据
met_fac = meta[['date', 'order_book_id']].reset_index(drop=True)
_, fac_batches = day_index(met_fac)
met_fac['gru_score'] = predict(torch.from_numpy(X).to(device), fac_batches)
factor = (met_fac.dropna(subset=['gru_score'])
                 .sort_values(['date', 'order_book_id']))
factor['date'] = factor['date'].dt.strftime('%Y-%m-%d')
factor.to_csv(OUT_FACTOR, index=False, encoding='utf-8-sig')
print(f'saved: {OUT_FACTOR}  rows={len(factor)}, '
      f'dates {factor.date.min()} ~ {factor.date.max()}, '
      f'stocks={factor.order_book_id.nunique()}')
print('注意: 2019-2025 训练期为样本内(模型见过这些标签), 该段回测收益会偏高;')
print('      严格样本外评估请看 2026 段或 gru_predictions.csv')
print('done.')

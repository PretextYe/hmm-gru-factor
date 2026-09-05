# data/

本目录存放数据文件。**数据体积较大且来源受许可限制，不随仓库发布**（见 `.gitignore`），请按以下说明自行准备。

## 需要的文件

| 文件 | 说明 | 来源 |
|---|---|---|
| `raw_factors.pkl` | 原始 7 因子表，MultiIndex `(date, order_book_id)`，列为 `close, open, high, low, turnover, returnPast1d, cap`（2019-07 起） | rqdatac 行情数据整理（RiceQuant） |
| `stock_pool.pkl` | 股票池表，列 `date, order_book_id, is_st, is_suspended` | 每日股票池（剔除 ST / 停牌） |

## 生成的文件（运行脚本后产生）

| 文件 | 生成脚本 |
|---|---|
| `hmm_factor.pkl` / `hmm_factor.csv` | `src/hmm_factor.py` → `src/merge_hmm_features.py` |
| `gru_factor.csv`、`gru_predictions.csv` | `src/gru_train.py`（输出至 `results/`） |

## 字段说明

- `turnover`：换手率；`returnPast1d`：1 日涨跌幅；`cap`：流通市值
- `hmm_factor`：HMM 因子 = 观测序列在 UP 模型下的对数观测概率 `log P(O|λ_up)`
- `logp_up` / `logp_down`：同一序列在 UP / DOWN 模型下的对数观测概率
- `gru_score`：GRU 模型输出的全历史因子值（米筐上传格式：`date, order_book_id, gru_score`）

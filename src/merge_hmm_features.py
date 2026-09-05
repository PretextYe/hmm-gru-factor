# -*- coding: utf-8 -*-
"""
merge_hmm_features.py
=====================
把 hmm_factor.py 的因子输出与原始 7 因子按 (date, order_book_id) 合并，
生成 gru_train.py 所需的训练输入 hmm_factor.csv：

    date, order_book_id, hmm_factor, logp_up, logp_down,
    close, open, high, low, turnover, returnPast1d, cap

用法：
    python src/merge_hmm_features.py \
        --factor data/hmm_factor.pkl --raw data/raw_factors.pkl --out data/hmm_factor.csv
"""
import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".pkl":
        return pd.read_pickle(p)
    return pd.read_csv(p, parse_dates=["date"])


def main():
    ap = argparse.ArgumentParser(description="合并 HMM 因子与原始特征")
    ap.add_argument("--factor", default=str(BASE / "data" / "hmm_factor.pkl"),
                    help="hmm_factor.py 的输出（.pkl 或 .csv）")
    ap.add_argument("--raw", default=str(BASE / "data" / "raw_factors.pkl"),
                    help="原始因子表（MultiIndex: date, order_book_id）")
    ap.add_argument("--out", default=str(BASE / "data" / "hmm_factor.csv"),
                    help="合并后的训练输入文件路径")
    args, _ = ap.parse_known_args()

    factor = _read(args.factor)
    raw = _read(args.raw)
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()

    out = (factor.merge(raw, on=["date", "order_book_id"], how="left")
                  .sort_values(["date", "order_book_id"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved: {args.out}  rows={len(out)}, "
          f"dates {out['date'].min().date()} ~ {out['date'].max().date()}")


if __name__ == "__main__":
    main()

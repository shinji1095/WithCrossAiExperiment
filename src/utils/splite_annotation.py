#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def stratified_split(df, label_col="signal", val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label, g in df.groupby(label_col):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Split annotation.csv into train/valid (stratified).")
    ap.add_argument("--annotation_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="preprocessed")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_missing", action="store_true", help="drop rows whose image file does not exist")
    args = ap.parse_args()

    ann = Path(args.annotation_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ann)

    # 画像存在チェック（任意）
    if args.drop_missing:
        exists_mask = df["filename"].apply(lambda p: Path(p).exists())
        missing = (~exists_mask).sum()
        if missing > 0:
            print(f"[WARN] drop {missing} rows (file not found).")
        df = df.loc[exists_mask].reset_index(drop=True)

    # 層化分割
    tr, va = stratified_split(df, label_col="signal", val_ratio=args.val_ratio, seed=args.seed)

    # 書き出し
    tr_path = out / "train.csv"
    va_path = out / "valid.csv"
    tr.to_csv(tr_path, index=False)
    va.to_csv(va_path, index=False)

    # 集計を表示
    def counts(d): 
        return d["signal"].value_counts(dropna=False).to_dict()
    print("[OK] train:", tr_path, counts(tr))
    print("[OK] valid:", va_path, counts(va))

if __name__ == "__main__":
    main()

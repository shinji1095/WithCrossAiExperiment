#!/usr/bin/env python3
# convert_annotations.py

from pathlib import Path
import pandas as pd
import numpy as np
import math
import argparse

# ── 定数 ──────────────────────────────────────────────────────────────
CLASS_TO_SIGNAL = {
    0: 0,  # Red
    1: 1,  # Green
    2: 1,  # Countdown Green → Green
    3: 2,  # Countdown Blank → None
    4: 2,  # None
}
STATE_NORMAL = 0  # すべて Normal とする

# ── 関数 ──────────────────────────────────────────────────────────────
def compute_slope_rad(row: pd.Series) -> float:
    """
    x₂→x₁ ベクトル (dx, dy) の法線 n=(dy, −dx) を取り，
    その傾き角 θ = atan2(n_y, n_x) [rad] を返す。
    右上がりを正とするため，この向きを採用している。
    """
    dx = row["x1"] - row["x2"]
    dy = row["y1"] - row["y2"]
    n_x, n_y = dy, -dx
    return math.atan2(n_y, n_x)

def convert(in_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(in_csv)
    print(df.head())

    # 必要列の存在確認
    required = {"file", "mode", "x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # 変換
    df_out = pd.DataFrame({
        "filename": df["file"],
        "signal":   df["mode"].map(CLASS_TO_SIGNAL).astype(int),
        "state":    STATE_NORMAL,
        "slope_deg": df.apply(compute_slope_rad, axis=1),
    })

    # 保存
    df_out.to_csv(out_csv, index=False)
    print(f"✔ Saved {len(df_out)} rows → {out_csv}")

# ── main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert annotation CSV to new format")
    parser.add_argument("--in_csv",  required=True, type=Path, help="Input CSV path")
    parser.add_argument("--out_csv", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    if not args.in_csv.is_file():
        raise FileNotFoundError(args.in_csv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    convert(args.in_csv, args.out_csv)

if __name__ == "__main__":
    main()

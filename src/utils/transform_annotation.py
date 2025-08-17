import pandas as pd
import numpy as np

def compute_slope(row):
    try:
        x_right = float(row['x_right'])
        y_right = float(row['y_right'])
        x_left = float(row['x_left'])
        y_left = float(row['y_left'])
        if np.isnan([x_right, y_right, x_left, y_left]).any():
            return np.nan
        if x_right == x_left:
            return np.inf  # avoid division by zero
        return (y_right - y_left) / (x_right - x_left)
    except:
        return np.nan

def convert_annotation_format(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # slope を計算し slope_deg にリネーム
    df['slope_deg'] = df.apply(compute_slope, axis=1)

    # crossing → state（int型に）
    df['state'] = pd.to_numeric(df['crossing'], errors='coerce').fillna(0).astype(int)

    # 必要なカラムだけ残してリネーム
    out_df = df.rename(columns={
        'file': 'filename',
        'mode': 'signal' # 0:Red, 1:Green, 2:Blank, 3:none
    })[['filename', 'signal', 'state', 'slope_deg']]

    # 出力
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved converted annotation to: {output_csv}")

# 使用例
convert_annotation_format("src/training_file.csv", "training.csv")

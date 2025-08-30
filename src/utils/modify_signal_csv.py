import pandas as pd
import os
from tqdm import tqdm

# 入出力パス
csv_path = r"D:\Datasets\WithCross Dataset\signal.csv"
dataset_root = r"D:\Datasets\VIDVIP"
output_csv_path = r"D:\Datasets\WithCross Dataset\signal_updated.csv"

# ラベル変換
label_map = {"RED": 0, "GREEN": 1}

# signal.csv の読み込み
df = pd.read_csv(csv_path)

# 各行を処理
rows = []
for _, row in tqdm(df.iterrows()):
    filename = row["signal_file"]
    label = label_map.get(row["label"], -1)
    if label == -1:
        print(f"未知のラベル: {row['label']} をスキップ")
        continue

    # 正しく txt 名を抽出
    base_name = filename.rsplit("_", 1)[0]
    txt_name = base_name + ".txt"

    # dataset 以下を再帰的に検索
    txt_path = None
    for root, _, files in os.walk(dataset_root):
        if txt_name in files:
            txt_path = os.path.join(root, txt_name)
            break

    if txt_path is None:
        print(f"{txt_name} が見つかりません")
        continue

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, center_x, center_y, width, height = parts
            rows.append({
                "signal_file": filename,
                "label": label,
                "class_id": int(class_id),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width": float(width),
                "height": float(height)
            })


# CSV保存
df_out = pd.DataFrame(rows)
df_out.to_csv(output_csv_path, index=False)
print(f"保存完了: {output_csv_path}")

# WithCrossAiExperiment

研究用のリポジトリです。本リポジトリでは、視覚障害者の移動支援デバイスに搭載するAI実験に関連するコードやデータを管理します。

---

## 目次
- [WithCrossAiExperiment](#withcrossaiexperiment)
  - [目次](#目次)
  - [概要](#概要)
  - [環境構築](#環境構築)
  - [インストール](#インストール)
    - [onnx2tf](#onnx2tf)
    - [データセット](#データセット)
  - [使い方](#使い方)
    - [学習](#学習)
    - [モデル変換](#モデル変換)
    - [評価](#評価)
    - [可視化](#可視化)
  - [ディレクトリ構成](#ディレクトリ構成)
  - [参考文献](#参考文献)
  - [ライセンス](#ライセンス)
  - [連絡先](#連絡先)

---

## 概要
TBD

---

## 環境構築
- OS: Ubuntu 20.04 / 22.04（推奨）
- 言語: Python 3.11
- フレームワーク: PyTorch / TensorFlow 
- 必要なライブラリは `requirements.txt` に記載

---

## インストール
```bash
git clone https://github.com/shinji1095/WithCrossAiExperiment.git
cd repository
pip install -r requirements.txt
```

### onnx2tf

このプロジェクトではtflite変換にonnx2tfを使用する．onnx2tfの推奨環境はlinuxであるためwindowsを使用している場合はWSL2を使用すること．

```shell
git clone https://github.com/PINTO0309/onnx2tf.git
```

dockerコンテナを起動する．

```shell
cd onnx2tf
docker build -t onnx2tf-image .
docker run -it --rm -v "$PWD":/app onnx2tf-image
```

コンテナに入り変換スクリプトを実行する．

```shell
docker exec -it onnx2tf-image bash
cd onnx2tf
python src/convert/convert_to_tflite.py
```

### データセット

WithCrossデータセットは現在作成中です．頻繁に格納場所が変更されリンク切れが発生する可能性があるので，その際はリポジトリ管理者に連絡してください．

- **学習データセット**
[WithCross Dataset](https://drive.google.com/file/d/1aKjXCvO9STohHB1Y5RhwzIArm0k24b1T/view?usp=drive_link)

- **テストデータセット**
[WithCross Dataset Test](https://drive.google.com/file/d/1aKjXCvO9STohHB1Y5RhwzIArm0k24b1T/view?usp=drive_link)

---

## 使い方
### 学習
```bash
python src/train.py --config config.yaml
```

### モデル変換

- 分類モデルを変換する場合

```bash
cd src
python convert/convert_to_tflite.py --task classification  --model edgenext_base.usi_in1k --image_size 320 --num_classes 3
```

- セグメンテーションモデルを変換する場合
  
```bash
cd src
python -m convert.convert_to_tflite --task segmentation  --model edgenext --num_classes 2 --image_size 320
```

### 評価
```bash
python src/test.py --config test.yaml
```

### 可視化
```bash
TBD
```

---

## ディレクトリ構成
```
.
├── data
├── src/
│   ├── config/
│   ├── loss/
│   ├── metrics/
│   ├── mixer/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   └── test.py
├── .gitignore
├── README.md
└── requirements.txt
```


---

## 参考文献
TBD

---

## ライセンス
TBD

---

## 連絡先
- eto.shinji786@mail.kyutech.jp
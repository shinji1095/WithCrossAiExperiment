# WithCrossAiExperiment

研究用のリポジトリです。本リポジトリでは、視覚障害者の移動支援デバイスに搭載するAI実験に関連するコードやデータを管理します。

---

## 目次
- [概要](#概要)
- [環境構築](#環境構築)
- [インストール](#インストール)
- [使い方](#使い方)
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

---

## 使い方
### 学習
```bash
python src/train.py --config config.yaml
```

### 評価
```bash
python src/test.py --config config.yaml
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
```

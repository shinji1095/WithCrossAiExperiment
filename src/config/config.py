import os, yaml

class TrainingConfig:
    """Parse a single training configuration block.

    New keys (all optional):
        dropout_rate     (float) : Dropout probability (default 0.0)
        label_smoothing  (float) : ε for label‑smoothing CE (default 0.0)
        mixup_alpha      (float) : α for MixUp Beta(α,α); 0.0 disables
        cutmix_alpha     (float) : α for CutMix Beta(α,α); 0.0 disables
        drop_path_rate   (float) : StochasticDepth probability (default 0.0)
        max_norm         (float) : Gradient‑clip max‑norm (0.0 disables)
    """
    def __init__(self, base_cfg: dict, model_name: str, override: dict):
        # ----- dataset / output paths -----
        base_dir = r'D:\Datasets\WithCross Dataset'
        self.train_file_dir = os.path.join(base_dir, 'csv', 'ptl_training.csv')
        self.valid_file_dir = os.path.join(base_dir, 'csv', 'ptl_validation.csv')
        self.train_img_dir  = os.path.join(base_dir, 'image')
        self.valid_img_dir  = os.path.join(base_dir, 'image')

        result_dir = 'data/ex001.7'
        self.save_path   = os.path.join(result_dir, 'weights')
        self.plot_path   = os.path.join(result_dir, 'plots')
        self.result_path = os.path.join(result_dir, 'results')

        # ----- merge common / override -----
        cfg = {**base_cfg, **override}

        # basic hyper‑parameters
        self.model_name    = model_name
        self.epochs        = int(cfg['epochs'])
        self.batch_size    = int(cfg['batch_size'])
        self.image_size    = tuple(cfg['image_size'])
        self.learning_rate = float(cfg['learning_rate'])
        self.weight_decay  = float(cfg['weight_decay'])
        self.patience      = int(cfg['patience'])
        self.min_delta     = float(cfg['min_delta'])
        self.task          = cfg['task']
        self.state_filter  = cfg.get('state_filter', None)

        # ----- NEW regularization params -----
        self.dropout_rate    = float(cfg.get('dropout_rate', 0.0))
        self.label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.mixup_alpha     = float(cfg.get('mixup_alpha', 0.0))
        self.cutmix_alpha    = float(cfg.get('cutmix_alpha', 0.0))
        self.drop_path_rate  = float(cfg.get('drop_path_rate', 0.0))
        self.max_norm        = float(cfg.get('max_norm', 0.0))

        self.LOSS = dict(
            name="focal",              # ["ce", "weighted_ce", "focal", "multitask"]
            apply_class_balance=False,  # True なら有効サンプル数に基づく重みを計算
            focal_gamma=2.0,        # FocalLoss 用
            label_smoothing=0.0,    # PyTorch >=1.10 の CrossEntropyLoss でサポート
            beta=0.9999,            # 有効サンプル数 (CB-Loss) の β
            alpha=1.0,              # multitask の重み
            beta_mse=1.0,           # multitask の MSE 側重み
        )

        self.AUGMENTATION = dict(
            name="cutmix",          # "none" | "cutmix" | "attentive_cutmix" |
                                    # "saliencymix" | "puzzlemix" | "snapmix" | "keepaugment"
            prob=1.0,               # 0.0–1.0 で実行確率
            beta=1.0,               # CutMix/Saliency 系の Beta 分布 α
            use_cam_backbone="resnet50",   # Attentive／SnapMix 用 – CAM を取る骨格
            saliency_method="grad",        # SaliencyMix／PuzzleMix 用 – "grad" | "smoothgrad"
            keepaugment_tau=0.15,          # KeepAugment の重要領域しきい値
        )

def load_all_training_configs(yaml_path: str):
    """Load YAML and create TrainingConfig for every model entry."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        exp = yaml.safe_load(f)['experiment']

    common = {k: v for k, v in exp.items() if k != 'models'}
    items  = exp['models']

    configs = []
    if isinstance(items, list):
        for elem in items:
            if isinstance(elem, str):
                configs.append(TrainingConfig(common, elem, {}))
            else:
                name = elem['name']
                override = {k: v for k, v in elem.items() if k != 'name'}
                configs.append(TrainingConfig(common, name, override))
    elif isinstance(items, dict):
        for name, override in items.items():
            configs.append(TrainingConfig(common, name, override or {}))
    else:
        raise ValueError("models のフォーマットが不正です")

    return configs

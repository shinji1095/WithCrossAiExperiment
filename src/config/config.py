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
        self.train_file_dir = cfg.get('train_file_dir', self.train_file_dir)
        self.valid_file_dir = cfg.get('valid_file_dir', self.valid_file_dir)
        self.train_img_dir  = cfg.get('train_img_dir',  self.train_img_dir)
        self.valid_img_dir  = cfg.get('valid_img_dir',  self.valid_img_dir)

        # ----- NEW regularization params -----
        self.dropout_rate    = float(cfg.get('dropout_rate', 0.0))
        self.label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.mixup_alpha     = float(cfg.get('mixup_alpha', 0.0))
        self.cutmix_alpha    = float(cfg.get('cutmix_alpha', 0.0))
        self.drop_path_rate  = float(cfg.get('drop_path_rate', 0.0))
        self.max_norm        = float(cfg.get('max_norm', 0.0))

        _loss_default = dict(
            name="focal",
            apply_class_balance=False,
            focal_gamma=2.0,
            label_smoothing=0.0,
            beta=0.9999,
            alpha=1.0,
            beta_mse=1.0,
        )
        _loss_default.update(cfg.get('LOSS', {}))
        self.LOSS = _loss_default

        # 既存の self.AUGMENTATION = dict(...) を以下に置き換え
        _aug_default = dict(
            name="cutmix",                 # "none"|"cutmix"|...|"signalmix"
            prob=1.0,
            beta=1.0,
            use_cam_backbone="resnet50",
            saliency_method="grad",
            keepaugment_tau=0.15,
        )
        _aug_default.update(cfg.get('AUGMENTATION', {}))

        # SignalMix を選んだときの既定値を補完（YAML未記載でも動くように）
        if str(_aug_default.get('name', '')).lower() == 'signalmix':
            _aug_default.setdefault('signal_dir',  r'D:\Datasets\WithCross Dataset\vidvip_signal/signal')
            _aug_default.setdefault('signal_csv',  r'D:\Datasets\WithCross Dataset\vidvip_signal/signal.csv')
            _aug_default.setdefault('none_index',  0)
            _aug_default.setdefault('use_cutmix',  True)

        self.AUGMENTATION = _aug_default

        # ★ 追加: wandb のプロジェクト名 / run の接頭語
        self.wandb_project = override.get('wandb_project', base_cfg.get('wandb_project', 'withcross-training'))
        self.run_prefix    = override.get('run_prefix',    base_cfg.get('run_prefix',    'run'))

        # ★ 追加（任意）: YAMLパスも保持しておくとログで便利
        self.config_path   = base_cfg.get('__yaml_path__', '')

def load_all_training_configs(yaml_path: str):
    """Load YAML and create TrainingConfig for every model entry."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        exp = yaml.safe_load(f)['experiment']

    common = {k: v for k, v in exp.items() if k != 'models'}
    items  = exp['models']

    # ★ 追加: YAMLパスと YAMLファイル名(stem)を共通設定に埋め込む
    common['__yaml_path__'] = yaml_path
    try:
        import os
        common.setdefault('run_prefix', os.path.splitext(os.path.basename(yaml_path))[0])
    except Exception:
        common.setdefault('run_prefix', 'run')


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

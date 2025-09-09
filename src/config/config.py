# config/config.py
import os, yaml

class TrainingConfig:
    """
    既存キーはそのまま。segmentation向けに以下を追加（すべて任意）:
        num_classes       (int)   : クラス数（既定 3 → segmentation なら2などを指定）
        train_mask_dir    (str)   : 学習マスクディレクトリ
        valid_mask_dir    (str)   : 検証マスクディレクトリ
        normalize_mean    (float) : 画像正規化の平均 (default 0.5)
        normalize_std     (float) : 画像正規化の標準偏差 (default 0.5)
        num_workers       (int)   : DataLoader workers (default: OS依存)
        prefetch_factor   (int)   : DataLoader prefetch_factor (default: 2)
    LOSS の既定値に segmentation 用パラメータを追加（未使用なら無害）:
        ignore_index      (int)   : 背景等の無視ラベル（例 255）
        class_weights     (list)  : CEに渡す重み（省略可）
    """
    def __init__(self, base_cfg: dict, model_name: str, override: dict):
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

        # basic hyper-parameters（互換）
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

        # ----- segmentation 追加（未指定なら None / 既存には無害） -----
        self.num_classes     = int(cfg.get('num_classes', 3))
        self.train_mask_dir  = cfg.get('train_mask_dir', None)
        self.valid_mask_dir  = cfg.get('valid_mask_dir', None)

        self.normalize_mean  = float(cfg.get('normalize_mean', 0.5))
        self.normalize_std   = float(cfg.get('normalize_std', 0.5))

        self.num_workers     = int(cfg.get('num_workers', 0))  # Windowsデフォルト0, 他環境はtrain_ddp側で上書きあり
        self.prefetch_factor = int(cfg.get('prefetch_factor', 2))

        # ----- regularization（互換） -----
        self.dropout_rate    = float(cfg.get('dropout_rate', 0.0))
        self.label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.mixup_alpha     = float(cfg.get('mixup_alpha', 0.0))
        self.cutmix_alpha    = float(cfg.get('cutmix_alpha', 0.0))
        self.drop_path_rate  = float(cfg.get('drop_path_rate', 0.0))
        self.max_norm        = float(cfg.get('max_norm', 0.0))

        # ----- LOSS 既定（segmentation項目を追加） -----
        _loss_default = dict(
            name="focal",
            apply_class_balance=False,
            focal_gamma=2.0,
            label_smoothing=0.0,
            beta=0.9999,
            alpha=1.0,
            beta_mse=1.0,
            # --- segmentation 用（未使用なら無害） ---
            ignore_index=255,
            class_weights=None,
        )
        _loss_default.update(cfg.get('LOSS', {}))
        self.LOSS = _loss_default

        # ----- AUGMENTATION（互換・省略） -----
        _aug_default = dict(
            name="cutmix",  # "none"|"cutmix"|...|"signalmix"
            prob=1.0,
            beta=1.0,
            use_cam_backbone="resnet50",
            saliency_method="grad",
            keepaugment_tau=0.15,
        )
        _aug_default.update(cfg.get('AUGMENTATION', {}))
        if str(_aug_default.get('name', '')).lower() == 'signalmix':
            _aug_default.setdefault('signal_dir',  r'D:\Datasets\WithCross Dataset\vidvip_signal/signal')
            _aug_default.setdefault('signal_csv',  r'D:\Datasets\WithCross Dataset\vidvip_signal/signal.csv')
            _aug_default.setdefault('none_index',  0)
            _aug_default.setdefault('use_cutmix',  True)
        self.AUGMENTATION = _aug_default

        # wandb（互換）
        self.wandb_project = override.get('wandb_project', base_cfg.get('wandb_project', 'withcross-training'))
        self.run_prefix    = override.get('run_prefix',    base_cfg.get('run_prefix',    'run'))
        self.config_path   = base_cfg.get('__yaml_path__', '')

def load_all_training_configs(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        exp = yaml.safe_load(f)['experiment']

    common = {k: v for k, v in exp.items() if k != 'models'}
    items  = exp['models']
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

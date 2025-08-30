import os
import torch
import torch.multiprocessing as mp
import argparse
from train_ddp import main_worker

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training_setting.yaml")
    args = parser.parse_args()

    # pass the config path to DDP workers via environment variable
    os.environ["TRAIN_CONFIG_PATH"] = args.config

    # -------------------------------------------------
    # DDP rendezvous 環境変数を“親プロセスで一度だけ”設定
    # -------------------------------------------------
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    world_size = torch.cuda.device_count() or 1
    mp.set_start_method("spawn", force=True)
    mp.spawn(main_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

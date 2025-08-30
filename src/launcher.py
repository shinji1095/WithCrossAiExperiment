# src/launcher.py
import os, torch, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] initialized")
    dist.destroy_process_group()

if __name__ == "__main__":
    # ----- 固定値を親で一度だけセット -----
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"     # 空いている適当なポート
    world_size = torch.cuda.device_count() or 1

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)

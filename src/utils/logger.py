# utils/logger.py
import logging, os, sys

def setup_logger(rank=0, log_dir="logs", filename="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"R{rank}")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s][%(name)s] %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)

    if rank == 0:
        fh = logging.FileHandler(os.path.join(log_dir, filename))
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

import argparse

from trainers import run_trainer_from_config
from utils.log import get_logger
import torch

import gc
# Tune GC to run less frequently.
# Thresholds: (700, 10, 10) -> (50000, 10, 10)
gc.set_threshold(500000, 10, 10)

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", action="store", choices=["train", "inference"])
parser.add_argument("--config", action="store", default="configs/flux_tiny_imagenet.yaml")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

def main() -> None:
    """Main training entry point that can work with any structured trainer."""
    # Parse command line arguments and get config path
    logger.info("Parsing command line arguments...")
    config_path = args.config

    # Run the trainer
    run_trainer_from_config(config_path, mode=args.mode)


if __name__ == "__main__":
    main()

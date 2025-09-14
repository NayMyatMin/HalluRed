import argparse

from jolt.config import load_config
from jolt.train import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()



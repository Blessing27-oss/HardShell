"""Decoupled analysis script."""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    main()

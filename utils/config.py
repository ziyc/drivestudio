import os

from omegaconf import OmegaConf


def _to_plain_config(cfg):
    data = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(data, dict):
        data.pop("base_config", None)
        data.pop("base_configs", None)
    return OmegaConf.create(data)


def load_config(config_path: str):
    config_path = os.path.abspath(config_path)
    cfg = OmegaConf.load(config_path)

    base_configs = []
    if "base_config" in cfg:
        base_configs.append(cfg.base_config)
    if "base_configs" in cfg:
        base_configs.extend(cfg.base_configs)

    merged = OmegaConf.create()
    config_dir = os.path.dirname(config_path)
    for base_config in base_configs:
        base_path = base_config
        if not os.path.isabs(base_path):
            base_path = os.path.join(config_dir, base_path)
        merged = OmegaConf.merge(merged, load_config(base_path))

    return OmegaConf.merge(merged, _to_plain_config(cfg))


def merge_optional_config(cfg, config_path: str | None):
    if not config_path:
        return cfg
    return OmegaConf.merge(cfg, load_config(config_path))

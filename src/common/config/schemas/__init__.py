from pydantic import BaseModel

from common.config.schemas.GPT import GPTConfig
from common.config.schemas.transformer import TransformerConfig
from common.config.schemas.training import TrainingConfig
from common.config.schemas.run import RunConfig

CONFIG_REGISTRY: dict[str, type[BaseModel]] = {
    cls.__name__: cls for cls in (GPTConfig, TransformerConfig, TrainingConfig, RunConfig)
}


def config_class(name: str) -> type[BaseModel]:
    """Returns the config class registered under name, as stored by save_checkpoint."""
    try:
        return CONFIG_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown config class, received: {name}") from None


__all__ = ["GPTConfig", "TransformerConfig", "TrainingConfig", "RunConfig", "CONFIG_REGISTRY", "config_class"]

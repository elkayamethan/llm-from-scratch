from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from common.config.schemas.GPT import GPTConfig
from common.config.schemas.training import TrainingConfig


class RunConfig(BaseModel):
    """Configuration of one pretraining run: model, training, and run-level fields"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_name: str = Field(min_length=1)
    seed: int = Field(ge=0, lt=2**31)
    num_workers: int = Field(ge=0, default=0)
    keep_last: int = Field(gt=0, default=3)
    pin_memory: bool = False
    resume_from: Path | None = None
    device: str | None = None
    data_dir: Path
    checkpoint_dir: Path

    model: GPTConfig
    training: TrainingConfig

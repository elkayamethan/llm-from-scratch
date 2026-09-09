from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARCHITECTURE_CONFIGS_DIR = CONFIGS_DIR / "architecture"
TRAINING_CONFIGS_DIR = CONFIGS_DIR / "training"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

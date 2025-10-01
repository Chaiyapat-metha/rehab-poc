# File: .\backend\app\config.py

import yaml
from pathlib import Path
from functools import lru_cache

# (config.py -> app -> backend -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

CONFIG_DIR = PROJECT_ROOT / "backend" / "training_configs"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
EXERCISES_CONFIG_PATH = CONFIG_DIR / "exercises.yaml"
AUGMENTATION_CONFIG_PATH = CONFIG_DIR / "augmentation.yaml"

@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Loads and caches all necessary YAML configurations, including LLM settings.
    """
    config = {}

    # --- 1. โหลด Training Configs ---
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config['model_config'] = yaml.safe_load(f)

    with open(EXERCISES_CONFIG_PATH, 'r') as f:
        config['exercises'] = yaml.safe_load(f)
        
    with open(AUGMENTATION_CONFIG_PATH, 'r') as f:
        config['augmentation'] = yaml.safe_load(f)
        
    # --- 2. โหลด models.yaml (LLM และ ONNX Paths) ---
    MODELS_YAML_PATH = PROJECT_ROOT / "backend" / "models.yaml"
    with open(MODELS_YAML_PATH, 'r') as f:
        models_config = yaml.safe_load(f)

    config['llm'] = models_config.get('llm', {}) 
    
    config['model_paths'] = models_config # เก็บ paths อื่นๆ ที่เหลือ

    return config


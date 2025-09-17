import yaml
import importlib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "models.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    """โหลดไฟล์ YAML config"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_class_from_path(path: str):
    """Dynamic import class จาก string path e.g., 'app.models.MyModel'"""
    try:
        module_name, cls_name = path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class from path: {path}") from e

def get_model_instance(model_name: str):
    """สร้าง instance ของโมเดลจาก config"""
    config = load_config()
    model_config = config.get(model_name)
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found in config.")

    cls = load_class_from_path(model_config['class_path'])
    instance = cls(**model_config.get('params', {}))
    
    # โหลด model weights ถ้ามี path ระบุไว้
    model_path = model_config.get('model_path')
    if model_path:
        instance.load(model_path)
        
    return instance

# โหลด config ทั้งหมดเมื่อ module ถูก import
app_config = load_config()
import json
from pathlib import Path

class My_Config():
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value) # 将 JSON 属性映射为类属性
    def save(self):
        data = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                if isinstance(v, Path):
                    data[k] = str(v)
                else:
                    data[k] = v
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)




import json
from pathlib import Path
from typing import Dict, Any, List

from src.config.settings import settings


class ModelRegistry:
    def __init__(self, registry_file: str = "models/registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Any] = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"models": []}

    def _save_registry(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=4)

    def register_model(self, version: str, model_type: str, metadata: Dict[str, Any], status: str = "candidate") -> None:
        model_entry = {
            "version": version,
            "type": model_type,
            "status": status,
            "metadata": metadata,
        }
        
        # Check if version exists, update it if so
        updated = False
        for i, m in enumerate(self.registry["models"]):
            if m["version"] == version:
                self.registry["models"][i] = model_entry
                updated = True
                break
                
        if not updated:
            self.registry["models"].append(model_entry)
            
        self._save_registry()

    def get_model(self, version: str) -> Dict[str, Any] | None:
        for m in self.registry["models"]:
            if m["version"] == version:
                return m
        return None

    def list_models(self, status: str | None = None) -> List[Dict[str, Any]]:
        if status:
            return [m for m in self.registry["models"] if m["status"] == status]
        return self.registry["models"]

# model_config.py

import os

class ModelConfig:
    def __init__(self, model_name="mnist_model.keras", model_dir="model"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.full_path = os.path.join(self.model_dir, self.model_name)

        os.makedirs(self.model_dir, exist_ok=True)

    def get_model_path(self):
        return self.full_path

    def model_exists(self):
        return os.path.exists(self.full_path)
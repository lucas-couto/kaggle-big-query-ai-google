from keras import layers, models
from .base_model import BaseModel
from keras.applications import DenseNet121

class Densenet(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/densenet")

    def build_model(self):
        model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model.trainable = False  # Pode destravar depois de algumas épocas para fine-tuning

        return model
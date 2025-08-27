from .base_model import BaseModel
from keras.applications import Xception

class XceptionModel(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/xception")

    def build_model(self):
        model = Xception(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model.trainable = False

        return model
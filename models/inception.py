from .base_model import BaseModel
from keras.applications import InceptionResNetV2

class Inception(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/inception")

    def build_model(self):
        model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model.trainable = False
        
        return model
from .base_model import BaseModel
from keras.applications import NASNetLarge

class Nasnet(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/nasnet")

    def build_model(self):
        model = NASNetLarge(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model.trainable = False
        
        return model
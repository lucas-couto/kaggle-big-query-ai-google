from .base_model import BaseModel
from keras.applications import ResNet152

class Resnet152(BaseModel):
    def __init__(self, config, data):
        super().__init__(config, data, checkpoint_dir="checkpoints/resnet152")
        
    def build_model(self):
        model = ResNet152(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model.trainable = False
        
        return model
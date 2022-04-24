from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend as K


class OverwriteMasking(layers.Layer):
    """Set values at spasific positions to zero.
    Args:
        inputs (list): value tensor and mask tensor.
    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * K.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

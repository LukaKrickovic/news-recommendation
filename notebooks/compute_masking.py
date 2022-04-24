from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend as K


class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.
    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape

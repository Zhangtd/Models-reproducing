import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras import activations, initializers

theta = 0.8


class LocalAttention(Layer):
    """
        generate local attention maps based on input score tensor.
        input: score tensor
        output: attention weights tensor(0/1)

        input shape: (batch_size, seq_length)
        output shape: (batch_size, seq_length)
    """
    def __init__(self, **kwargs):
        super(LocalAttention, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        super(LocalAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        max_score = K.max(x)
        min_score = K.min(x)
        threhold = theta*min_score + (1-theta)*max_score
        threholds = threhold * K.ones_like(x)
        output = K.relu(x-threholds)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape


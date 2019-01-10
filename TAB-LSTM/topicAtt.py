from keras.models import Model
from keras.layers import Input
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras import activations, initializers


class topicAttention(Layer):
    """
    self defined topical attention layer.
    input: [hiddenStates, topicDistribution]
    input_shape: [(batch_size, seq_len, embedding_size), (batch_size, topic_num)]
    output: topical_text_feature
    output shape: (batch_size, embedding_size)
    """
    def __init__(self, **kwargs):
        super(topicAttention, self).__init__(**kwargs)
        # self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        self.embedding_size = input_shape[0][-1]
        self.topic_num = input_shape[1][-1]
        self.seq_len = input_shape[0][1]

        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(name='w',
                                 shape=(self.embedding_size, self.topic_num),
                                 initializer='random_normal',
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.embedding_size, 1),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(name='u',
                                 shape=(self.embedding_size, self.embedding_size),
                                 initializer='random_normal',
                                 trainable=True)
        super(topicAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        h = x[0]
        theta = x[1]

        theta_w = K.dot(theta, K.transpose(self.w))
        theta_w = K.repeat(theta_w, self.seq_len)
        h_ = K.reshape(h, [-1, self.embedding_size])
        h_u = K.dot(h_, self.u)
        h_u = K.reshape(h_u, [-1, self.seq_len, self.embedding_size])

        g = K.dot(K.tanh(theta_w+h_u), self.v)
        weight = K.softmax(K.squeeze(g, axis=-1))
        weight = K.expand_dims(weight, axis=-1)
        weight = K.repeat_elements(weight, self.embedding_size, axis=-1)
        vec = weight * h
        vec = K.sum(vec, axis=1)

        return vec

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][-1])
        return output_shape


if __name__ == "__main__":
    input1 = Input(batch_shape=(10, 25, 50))
    input2 = Input(batch_shape=(10, 20))

    topic_h = topicAttention()([input1, input2])
    print(topic_h)

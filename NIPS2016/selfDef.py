import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras import activations, initializers, regularizers, constraints

top_K = 1
REMOVE_FACTOR = -10000


class text_attention(Layer):
    """
    self defined text attention layer.
    input: hidden text feature
    output: summarized text feature with attention mechanism

    input shape: (batch_size, seq_length, embedding_size)
    output shape: (batch_size, embedding_size)
    """
    def __init__(self, units, return_alphas=False, **kwargs):
        super(text_attention, self).__init__(**kwargs)
        self.units = units
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True
        self.return_alphas = return_alphas

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        self.w_omega = self.add_weight(name='w_omega',
                                       shape=(input_dim, self.units),
                                       initializer='random_normal',
                                       trainable=True)
        self.b_omega = self.add_weight(name='b_omega',
                                       shape=(self.units,),
                                       initializer='zeros',
                                       trainable=True)
        self.u_omega = self.add_weight(name='u_omega',
                                       shape=(self.units,),
                                       initializer='random_normal',
                                       trainable=True)
        super(text_attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        input_dim = K.shape(x)[-1]
        v = K.tanh(K.dot(K.reshape(x, [-1, input_dim]), self.w_omega) + K.expand_dims(self.b_omega, 0))
        vu = K.dot(v, K.expand_dims(self.u_omega, -1))
        vu = K.reshape(vu, K.shape(x)[:2])
        m = K.cast(mask, dtype='float32')
        m = m - 1
        m = m * REMOVE_FACTOR
        vu = vu + m
        alphas = K.softmax(vu)
        output = K.sum(x * K.expand_dims(alphas, -1), 1)
        if self.return_alphas:
            return [output] + [alphas]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        if self.return_alphas:
            alphas_shape = [(input_shape[0], input_shape[1])]
            return [output_shape] + alphas_shape
        else:
            return output_shape

    def get_config(self):
        return super(text_attention, self).get_config()


class coAttention_alt(Layer):
    """
    self defined co-attention layer.
    alternative co-attention
    inputs: [image feature tensor, hidden text feature tensor, summarized text feature tensor(after attention)]
    output: co-Attention feature of image and text

    input dimensions:[(batchSize, num_region, CNN_dimension),
                    (batchSize, seq_length, CNN_dimension),(batchSize, CNN_dimension)]
    output dimension: batch_size*CNN_dimension
    """
    def __init__(self, dim_k, **kwargs):
        super(coAttention_alt, self).__init__(**kwargs)
        self.dim_k = dim_k  # internal tensor dimension
        # self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A Co-Attention_alt layer should be called '
                             'on a list of inputs.')
        if len(input_shape) != 3:
            raise ValueError('A Co-Attention_alt layer should be called on a list of 3 inputs.'
                             'Got '+str(len(input_shape))+'inputs.')
        self.num_imgRegion = input_shape[0][1]
        self.seq_len = input_shape[1][1]
        self.output_dim = input_shape[0][2]

        """trainable variables naming rule:
            w/b + '_Dense_' + Vi/Vt + '_' + 0/1
            w: weight
            b: bias
            Vi: about image feature
            Vt: about text feature
            0: phase 0
            1: phase 1
        """
        self.w_Dense_Vi_0 = self.add_weight(name='w_Dense_Vi_0',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Vt_0 = self.add_weight(name='w_Dense_Vt_0',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Pi_0 = self.add_weight(name='w_Dense_Pi_0',
                                            shape=(2*self.dim_k, 1),
                                            initializer='random_normal',
                                            trainable=True)
        self.b_Dense_Pi_0 = self.add_weight(name='b_Dense_Pi_0',
                                            shape=(self.num_imgRegion,),
                                            initializer='zeros',
                                            trainable=True)

        self.w_Dense_Vi_1 = self.add_weight(name='w_Dense_Vi_1',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Vt_1 = self.add_weight(name='w_Dense_Vt_1',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Pi_1 = self.add_weight(name='w_Dense_Pi_1',
                                            shape=(2*self.dim_k, 1),
                                            initializer='random_normal',
                                            trainable=True)
        self.b_Dense_Pi_1 = self.add_weight(name='b_Dense_Pi_1',
                                            shape=(self.seq_len,),
                                            initializer='zeros',
                                            trainable=True)

        super(coAttention_alt, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        ifeature = x[0]
        tfeature_h = x[1]
        tfeature = x[2]
        output_dim = self.output_dim
        num_imgRegion = self.num_imgRegion
        dim_k = self.dim_k
        seq_len = self.seq_len

        # phase 0: text-guided image feature computation
        w_Vi_0 = K.dot(K.reshape(ifeature, [-1, output_dim]), self.w_Dense_Vi_0)
        # shape=((batchSize*num_imgRegion),dim_k)
        w_Vi_0 = K.reshape(w_Vi_0, [-1, num_imgRegion, dim_k])  # shape=(batchSize,num_imgRegion,dim_k)
        w_Vt_0 = K.repeat(K.dot(tfeature, self.w_Dense_Vt_0), num_imgRegion)  # shape=(batchSize,num_imgRegion,dim_k)
        Vi_Vt_0 = K.concatenate([w_Vi_0, w_Vt_0], axis=-1)  # shape=(batchSize,num_imgRegion,2*dim_k)
        Hi = K.tanh(Vi_Vt_0)
        # Hi_w = K.squeeze(K.dot(K.reshape(Hi, [-1, 2*dim_k]), self.w_Dense_Pi_0), axis=-1)
        # Hi_w_b = K.reshape(Hi_w, [-1, num_imgRegion]) + self.b_Dense_Pi_0
        Hi_w_b = K.squeeze(K.dot(Hi, self.w_Dense_Pi_0), axis=-1) + self.b_Dense_Pi_0  # shape=(batchSize,num_imgRegion)
        Pi = K.softmax(Hi_w_b)
        Pi = K.permute_dimensions(K.repeat(Pi, output_dim), (0, 2, 1))  # shape=(batchSize,num_imgRegion,output_dim)
        Pi_Vi = Pi*ifeature
        Vi = K.sum(Pi_Vi, axis=1)  # shape=(batchSize,output_dim)

        # phase 1: image-guided text feature computation
        w_Vi_1 = K.repeat(K.dot(Vi, self.w_Dense_Vi_1), seq_len)    # shape=(batchSize,seq_len,dim_k)
        w_Vt_1 = K.dot(K.reshape(tfeature_h, [-1, output_dim]), self.w_Dense_Vt_1)   # shape=((batchSize*seq_len),dim_k)
        w_Vt_1 = K.reshape(w_Vt_1, (-1, seq_len, dim_k))    # shape= (batchSize, seq_len, dim_k)
        Vi_Vt_1 = K.concatenate([w_Vi_1, w_Vt_1], axis=-1)    # shape=(batchSize, seq_len, 2*dim_k)
        Ht = K.tanh(Vi_Vt_1)
        Ht_b = K.squeeze(K.dot(Ht, self.w_Dense_Pi_1), axis=-1) + self.b_Dense_Pi_1   # shape=(batch_size, seq_len)
        Pt = K.softmax(Ht_b)
        Pt = K.permute_dimensions(K.repeat(Pt, output_dim), (0, 2, 1))    # shape=(batchSize, seq_len, output_dim)
        Pt_Vt = Pt*tfeature_h
        Vt = K.sum(Pt_Vt, axis=1)    # shape=(batchSize, output_dim)

        return Vi+Vt

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][-1])
        return output_shape

    def get_config(self):
        return super(coAttention_alt, self).get_config()


class coAttention_para(Layer):
    """
    self-defined parallel co-attention layer.
    inputs: [tFeature, iFeature]
    outputs: [coFeature]

    dimension:
    input dimensions: [(batch_size, seq_length, embedding_size), (batch_size, num_img_region, 2*hidden_size)]
        considering subsequent operation, better to set embedding_size == 2*hidden_size
    output dimensions:[(batch_size, 2*hidden_size)]
    """
    def __init__(self, dim_k, **kwargs):
        super(coAttention_para, self).__init__(**kwargs)
        self.dim_k = dim_k  # internal tensor dimension
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A Co-Attention_para layer should be called '
                             'on a list of inputs.')
        if len(input_shape) != 2:
            raise ValueError('A Co-Attention_para layer should be called on a list of 2 inputs.'
                             'Got '+str(len(input_shape))+'inputs.')
        self.embedding_size = input_shape[0][-1]
        self.num_region = input_shape[1][1]
        self.seq_len = input_shape[0][1]
        """
        naming variables following the VQA paper
        """
        self.Wb = self.add_weight(name="Wb",
                                  initializer="random_normal",
                                  # initializer="ones",
                                  shape=(self.embedding_size, self.embedding_size),
                                  trainable=True)
        self.Wq = self.add_weight(name="Wq",
                                  initializer="random_normal",
                                  # initializer="ones",
                                  shape=(self.embedding_size, self.dim_k),
                                  trainable=True)
        self.Wv = self.add_weight(name="Wv",
                                  initializer="random_normal",
                                  # initializer="ones",
                                  shape=(self.embedding_size, self.dim_k),
                                  trainable=True)
        self.Whv = self.add_weight(name="Whv",
                                   initializer="random_normal",
                                   # initializer="ones",
                                   shape=(self.dim_k, 1),
                                   trainable=True)
        self.Whq = self.add_weight(name="Whq",
                                   initializer="random_normal",
                                   # initializer="ones",
                                   shape=(self.dim_k, 1),
                                   trainable=True)

        super(coAttention_para, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        tFeature = inputs[0]
        iFeature = inputs[1]
        # affinity matrix C
        affi_mat = K.dot(tFeature, self.Wb)
        affi_mat = K.batch_dot(affi_mat, K.permute_dimensions(iFeature, (0, 2, 1)))  # (batch_size, seq_len, num_region)
        # Hq, Hv, av, aq
        tmp_Hv = K.dot(tFeature, self.Wq)
        Hv = K.dot(iFeature, self.Wv) + K.batch_dot(K.permute_dimensions(affi_mat, (0, 2, 1)), tmp_Hv)
        Hv = K.tanh(Hv)
        av = K.softmax(K.squeeze(K.dot(Hv, self.Whv), axis=-1))

        tmp_Hq = K.dot(iFeature, self.Wv)
        Hq = K.dot(tFeature, self.Wq) + K.batch_dot(affi_mat, tmp_Hq)
        Hq = K.tanh(Hq)
        aq = K.softmax(K.squeeze(K.dot(Hq, self.Whq), axis=-1))

        av = K.permute_dimensions(K.repeat(av, self.embedding_size), (0, 2, 1))
        aq = K.permute_dimensions(K.repeat(aq, self.embedding_size), (0, 2, 1))

        tfeature = K.sum(aq * tFeature, axis=1)
        ifeature = K.sum(av * iFeature, axis=1)

        return tfeature+ifeature

    def get_config(self):
        return super(coAttention_para, self).get_config()


class encoding(Layer):
    """
    self defined encoding layer, summarize total co-feature based on three level co-features
    input: [co_feature_word, co_feature_phrase, co_feature_text]
    output: total co_feature

    dimension :
    input dimensions : [(batch_size, embedding_size)]*3
    output dimension: (batch_size, embedding_size)
    """
    def __init__(self, **kwargs):
        super(encoding, self).__init__(**kwargs)
        # self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A Co-Attention_alt layer should be called '
                             'on a list of inputs.')
        if len(input_shape) != 3:
            raise ValueError('A Co-Attention_alt layer should be called on a list of 3 inputs.'
                             'Got '+str(len(input_shape))+'inputs.')
        self.output_dim = input_shape[0][-1]

        self.w_word = self.add_weight(name='w_word',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        self.w_phrase = self.add_weight(name="w_phrase",
                                        shape=(self.output_dim*2, self.output_dim),
                                        initializer="random_normal",
                                        trainable=True)
        self.w_text = self.add_weight(name="w_text",
                                      shape=(self.output_dim*2, self.output_dim),
                                      initializer="random_normal",
                                      trainable=True)
        super(encoding, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        feature_word = inputs[0]
        feature_phrase = inputs[1]
        feature_text = inputs[2]

        h_w = K.tanh(K.dot(feature_word, self.w_word))
        h_p = K.tanh(K.dot(K.concatenate([feature_phrase, h_w]), self.w_phrase))
        h_t = K.tanh(K.dot(K.concatenate([feature_text, h_p]), self.w_text))

        return h_t

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1])
        return output_shape

    def get_config(self):
        return super(encoding, self).get_config()


class self_conv1d(Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format=None,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_bias=True,
                 ** kwargs):
        super(self_conv1d, self).__init__(**kwargs)
        self.rank = 1
        self.filters = filters
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = True
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('A Co-Attention_alt layer should be called on a tensor of 3 dims.'
                             'Got '+str(len(input_shape))+'dims.')
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      # initializer='ones',
                                      name='kernel',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        # initializer='zeros',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, mask=None):
        # print(K.get_value(self.kernel))
        outputs = K.conv1d(
            inputs,
            self.kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape

    def get_config(self):
        return super(self_conv1d, self).get_config()


class self_maxpooling(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(self_maxpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A Co-Attention_alt layer should be called '
                             'on a list of inputs.')
        self.num_inputs = len(input_shape)

    def call(self, inputs, mask=None):
        tmp = K.stack([inputs[0], inputs[1], inputs[2]], axis=1)

        outputs = K.max(tmp, axis=1)

        return outputs

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        return output_shape

    def get_config(self):
        return super(self_maxpooling, self).get_config()


def myLossFunc(y_true, y_pred):
    probs_log = -K.log(y_pred)
    loss = K.mean(K.sum(probs_log*y_true, axis=-1))
    return loss



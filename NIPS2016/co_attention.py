from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Input, Reshape, Dense, Embedding, Bidirectional, GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers

from selfDef import coAttention_alt, coAttention_para, myLossFunc, self_conv1d, \
    self_maxpooling, text_attention, encoding


def textFeature(X):
    embeddings = Embedding(input_dim=num_words + 3, output_dim=embedding_size,
                weights=[init_embeddings], mask_zero=True, input_length=seq_length)(X)

    word_level = embeddings
    phrase_level_1 = self_conv1d(filters=embedding_size,
                                 kernel_size=1,
                                 padding="same",
                                 activation="tanh")(word_level)
    phrase_level_2 = self_conv1d(filters=embedding_size,
                                 kernel_size=2,
                                 padding="same",
                                 activation="tanh")(word_level)
    phrase_level_3 = self_conv1d(filters=embedding_size,
                                 kernel_size=3,
                                 padding="same",
                                 activation="tanh")(word_level)

    phrase_level = self_maxpooling()([phrase_level_1, phrase_level_2, phrase_level_3])

    text_level = Bidirectional(GRU(units=hidden_size, return_sequences=True))(phrase_level)
    return word_level, phrase_level, text_level


def imageFeature(inputs):
    imageModel = InceptionV3(weights='imagenet', include_top=False, )
    for layer in imageModel.layers:
        layer.trainable = False
    features = imageModel(inputs)
    features = Reshape(target_shape=(num_region, 2048))(features)
    features = Dense(hidden_size * 2, activation="tanh", use_bias=False)(features)
    return features


def modelDef():
    inputs_img = Input(shape=(299, 299, 3,))
    inputs_text = Input(shape=(seq_length,))
    text_mask = Input(shape=(seq_length,))

    iFeature = imageFeature(inputs_img)
    tFeature_word, tFeature_phrase, tFeature_text = textFeature(inputs_text)

    sum_tFeature_word = text_attention(attention_size)(tFeature_word)
    sum_tFeature_phrase = text_attention(attention_size)(tFeature_phrase)
    sum_tFeature_text = text_attention(attention_size)(tFeature_text)

    co_feature_word = coAttention_alt(dim_k=dim_k)([iFeature, tFeature_word, sum_tFeature_word])
    co_feature_phrase = coAttention_alt(dim_k=dim_k)([iFeature, tFeature_phrase, sum_tFeature_phrase])
    co_feature_text = coAttention_alt(dim_k=dim_k)([iFeature, tFeature_text, sum_tFeature_text])

    h = encoding()([co_feature_word, co_feature_phrase, co_feature_text])
    dropout = Dropout(drop_rate)(h)

    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)

    model = Model(inputs=[inputs_img, inputs_text, text_mask],
                  outputs=[Softmax])
    sgd = optimizers.SGD(lr=0.15, momentum=0.9, clipnorm=1.0)
    model.compile(optimizer=sgd, loss=myLossFunc, metrics=[accuracy])
    # res = model.predict(x=..)
    return model







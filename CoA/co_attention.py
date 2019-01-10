'''
hierarchical Co-attention model based on IJCAI article
'''
from keras.models import Model
from keras.layers.core import Activation, Flatten, Reshape, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import AveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Dense, Embedding, Merge, Dropout, Lambda
import keras.backend as K

from selfDef import coAttention_alt, myLossFunc
import numpy as np

num_tags = 3896
num_words = 212000
index_from = 3
seq_length = 30
batch_size = 512
embedding_size = 200
hidden_size = 100
attention_size = 200
dim_k = 100
num_region = 7*7
drop_rate = 0.5

# prepare the following data. img data is the output of VGG-16
img_train, text_train, tag_train, img_test, text_test, tag_test


def imageFeature(inputs):
    features = Reshape(target_shape=(num_region, 512))(inputs)
    features = Dense(embedding_size, activation="tanh", use_bias=False)(features)
    features_pooling = AveragePooling1D(pool_size=num_region, padding="same")(features)
    features_pooling = Lambda(lambda x: K.squeeze(x, axis=1))(features_pooling)

    return features, features_pooling


def textFeature(X):
    embeddings = Embedding(input_dim=num_words + index_from, output_dim=embedding_size,
                           mask_zero=True, input_length=seq_length)(X)
    tFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)

    return tFeature


def modelDef():
    inputs_img = Input(shape=(7, 7, 512))
    inputs_text = Input(shape=(seq_length,))

    iFeature, iFeature_pooling = imageFeature(inputs_img)
    tFeature = textFeature(inputs_text)
    co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature])
    dropout = Dropout(drop_rate)(co_feature)
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
    model = Model(inputs=[inputs_img, inputs_text],
                  outputs=[Softmax])
    # adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    model.compile(optimizer="adam", loss=myLossFunc)
    return model


def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:]
        if np.sum(y_true[i, top_indices]) >= 1:
            acc_count += 1
        p = np.sum(y_true[i, top_indices])/top_K
        r = np.sum(y_true[i, top_indices])/np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
        if p != 0 or r != 0:
            f1_K.append(2 * p * r / (p + r))
        else:
            f1_K.append(0)

    acc_K = acc_count * 1.0 / y_pred.shape[0]

    return acc_K, np.mean(np.array(precision_K)), np.mean(np.array(recall_K)), np.mean(np.array(f1_K))


if __name__ == "__main__":
    model = modelDef()
    history = model.fit(x=[img_train, text_train],
                        y=tag_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1,)
    y_pred = model.predict(x=[test_img, test_text])
    acc_K, precision_K, recall_K, f1_K = evaluation(test_tag, y_pred, TopK)





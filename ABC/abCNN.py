from keras.models import Model
from keras.layers import Input, Reshape, Dense, Embedding, Dropout, LSTM, MaxPooling1D, Lambda, Concatenate, \
    Multiply, RepeatVector, Flatten, Activation, Permute, Conv1D
import keras.backend as K
from localAtt import LocalAttention
import numpy as np


num_tags = 2207
num_words = 20000
index_from = 3
seq_length = 30
batch_size = 256
embedding_size = 100
drop_rate = 0.75
num_epoch = 35

# prepare the following data. img data is the output of VGG-16
img_train, text_train, tag_train, img_test, text_test, tag_test


def myLossFunc(y_true, y_pred):
    probs_log = -K.log(y_pred)
    loss = K.mean(K.sum(probs_log*y_true, axis=-1))
    return loss


def modelDef():
    input_text = Input(shape=(seq_length, ))
    embeddings = Embedding(input_dim=num_words + index_from, output_dim=embedding_size,
                           mask_zero=False, input_length=seq_length)(input_text)
    #Global channel
    gc1 = Conv1D(filters=embedding_size, kernel_size=1, activation="tanh", use_bias=True)(embeddings)
    gc2 = Conv1D(filters=embedding_size, kernel_size=2, activation="tanh", use_bias=True)(embeddings)
    gc3 = Conv1D(filters=embedding_size, kernel_size=3, activation="tanh", use_bias=True)(embeddings)

    gc1m = MaxPooling1D(pool_size=30)(gc1)
    gc2m = MaxPooling1D(pool_size=29)(gc2)
    gc3m = MaxPooling1D(pool_size=28)(gc3)
    gc = MaxPooling1D(pool_size=3)(Concatenate(axis=1)([gc1m, gc2m, gc3m]))
    gc = Lambda(lambda x:K.squeeze(x, axis=1))(gc)
    # print(gc)

    #Local channel
    lcs = Conv1D(filters=1, kernel_size=5, activation="tanh", use_bias=True, padding="same")(embeddings)
    lcs = Lambda(lambda x:K.squeeze(x, axis=-1))(lcs)
    bools = LocalAttention()(lcs)
    bools = Permute([2, 1])(RepeatVector(embedding_size)(bools))
    lca = Multiply()([embeddings, bools])
    lcf = Activation("tanh")(Lambda(lambda x:K.sum(x, axis=1))(lca))
    # print(lcf)

    gc = RepeatVector(1)(gc)
    lcf = RepeatVector(1)(lcf)
    h = Concatenate(axis=1)([gc, lcf])
    h = Conv1D(filters=embedding_size, kernel_size=2, activation="tanh", use_bias=True)(h)
    dropout = Dropout(drop_rate)(Lambda(lambda x:K.squeeze(x, axis=1))(h))

    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
    model = Model(inputs=input_text, outputs=Softmax)
    model.compile(optimizer="adam", loss=myLossFunc)
    return model


def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    correct = 0

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:]
        if np.sum(y_true[i, top_indices]) >= 1:
            acc_count += 1
        correct += np.sum(y_true[i, top_indices])

    acc_K = acc_count * 1.0 / y_pred.shape[0]
    precision_K = correct * 1.0 / (top_K * y_pred.shape[0])
    recall_K = correct * 1.0 / np.sum(y_true)
    f1_K = 2 * precision_K * recall_K / (precision_K + recall_K)

    return acc_K, precision_K, recall_K, f1_K


if __name__ == "__main__":
    myModel = modelDef()
    history = myModel.fit(x=texts_train,
                          y=tags_train,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,)
    y_pred = myModel.predict(x=[texts_test])
    acc, precision, recall, f1 = evaluation(tags_test, y_pred, 3)

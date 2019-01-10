
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Embedding, Dropout, LSTM, AveragePooling1D, Lambda, Concatenate, \
    Multiply, RepeatVector, Flatten, Activation, Permute, merge
import keras.backend as K

from topicAtt import topicAttention
import numpy as np

num_tags = 3896
num_words = 212000
index_from = 3
seq_length = 30
batch_size = 512
embedding_size = 300
attention_size = 200
topic_num = 100
dim_k = 100
drop_rate = 0.75

# prepare the following data. img data is the output of VGG-16
topics_train, text_train, tag_train, topics_test, text_test, tag_test


def myLossFunc(y_true, y_pred):
    probs_log = -K.log(y_pred)
    loss = K.mean(K.sum(probs_log*y_true, axis=-1))
    # loss = K.mean(K.sum(K.clip(probs_log * y_true, -1e40, 100), axis=-1))
    return loss


def modelDef():
    input_text = Input(shape=(seq_length, ))
    input_topic = Input(shape=(topic_num,))

    embeddings = Embedding(input_dim=num_words+index_from, output_dim=embedding_size,
                           mask_zero=True, input_length=seq_length)(input_text)
    tFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)
    topic_h = topicAttention()([tFeature, input_topic])
    dropout = Dropout(drop_rate)(topic_h)
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)

    model = Model(inputs=[input_text, input_topic], outputs=[Softmax])
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
    history = myModel.fit(x=[texts_train, topics_train],
                          y=tags_train,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1, )
    y_pred = myModel.predict(x=[texts_test, topics_test])
    acc, precision, recall, f1 = evaluation(tags_test, y_pred, top_K)


